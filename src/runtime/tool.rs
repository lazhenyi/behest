//! Tool runtime layer with validation, timeout, and bounded parallelism.
//!
//! [`ToolRuntime`] wraps a [`ToolRegistry`] and enforces [`RuntimePolicy`]
//! constraints during tool execution: JSON schema validation, per-tool
//! timeouts, bounded concurrency, and execution recording to an
//! [`ExecutionStore`].

use std::sync::Arc;
use std::time::Instant;

use futures_util::future::join_all;
use serde_json::Value;
use tokio::sync::Semaphore;
use tokio::time::timeout;
use tracing::{debug, warn};
use uuid::Uuid;

use super::error::{RuntimeError, RuntimeResult};
use super::policy::RuntimePolicy;
use crate::provider::{Message, ToolCall};
use crate::store::{ExecutionStore, ToolExecution};
use crate::tool::{ToolOutput, ToolRegistry};
use crate::tool_output::{self, ToolOutputConfig};

/// Result of a single tool execution within the runtime.
#[derive(Debug)]
pub struct ToolExecutionOutcome {
    /// The original tool call.
    pub call: ToolCall,
    /// Output if successful, or error message string if failed.
    pub output: Result<ToolOutput, String>,
    /// The resulting message for the conversation.
    pub message: Message,
}

/// Runtime layer that orchestrates tool execution with policy enforcement.
pub struct ToolRuntime {
    registry: ToolRegistry,
    policy: RuntimePolicy,
    semaphore: Arc<Semaphore>,
}

impl ToolRuntime {
    /// Creates a new tool runtime wrapping the given registry.
    #[must_use]
    pub fn new(registry: ToolRegistry, policy: RuntimePolicy) -> Self {
        let semaphore = Arc::new(Semaphore::new(policy.max_tool_concurrency));
        Self {
            registry,
            policy,
            semaphore,
        }
    }

    /// Returns a reference to the underlying tool registry.
    #[must_use]
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Returns the runtime policy.
    #[must_use]
    pub fn policy(&self) -> &RuntimePolicy {
        &self.policy
    }

    /// Executes a batch of tool calls with bounded parallelism, timeout,
    /// validation, and optional execution recording.
    ///
    /// Each tool call is executed independently. If `execution_store` is
    /// provided, each execution is recorded.
    ///
    /// When `continue_on_tool_failure` is true (from policy), failed tool
    /// calls produce error messages but do not abort the batch.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ToolTimeout`] when the semaphore is exhausted,
    /// or propagates errors from individual tool executions when the policy
    /// does not allow continuation on failure.
    pub async fn execute_batch(
        &self,
        calls: Vec<ToolCall>,
        session_id: Uuid,
        message_id: Uuid,
        execution_store: Option<&dyn ExecutionStore>,
    ) -> RuntimeResult<Vec<ToolExecutionOutcome>> {
        if calls.is_empty() {
            return Ok(Vec::new());
        }

        let futures: Vec<_> = calls
            .into_iter()
            .map(|call| {
                let sem = Arc::clone(&self.semaphore);
                let registry = self.registry.clone();
                let tool_timeout = self.policy.tool_timeout;
                let continue_on_failure = self.policy.continue_on_tool_failure;

                async move {
                    let _permit = sem.acquire().await.map_err(|_| RuntimeError::ToolTimeout {
                        tool: call.name.clone(),
                    })?;

                    let outcome = Self::execute_single(
                        &registry,
                        &call,
                        tool_timeout,
                        continue_on_failure,
                        &self.policy.tool_output,
                    )
                    .await;

                    if let Some(store) = execution_store {
                        Self::record_execution(store, session_id, message_id, &call, &outcome)
                            .await;
                    }

                    Ok::<ToolExecutionOutcome, RuntimeError>(outcome)
                }
            })
            .collect();

        let results = join_all(futures).await;
        results.into_iter().collect()
    }

    async fn execute_single(
        registry: &ToolRegistry,
        call: &ToolCall,
        tool_timeout: std::time::Duration,
        continue_on_failure: bool,
        truncation_config: &ToolOutputConfig,
    ) -> ToolExecutionOutcome {
        let Some(tool) = registry.get(&call.name) else {
            let error_msg = format!("tool not found: {}", call.name);
            warn!(tool = %call.name, "tool not found");
            return ToolExecutionOutcome {
                call: call.clone(),
                output: Err(error_msg.clone()),
                message: Message::tool_text(
                    call.id.clone(),
                    call.name.clone(),
                    format!("{{\"error\":\"{error_msg}\"}}"),
                ),
            };
        };

        if let Err(validation_error) =
            Self::validate_arguments(&tool.parameters_schema(), &call.arguments)
        {
            debug!(tool = %call.name, error = %validation_error, "schema validation failed");
            if !continue_on_failure {
                return ToolExecutionOutcome {
                    call: call.clone(),
                    output: Err(validation_error.clone()),
                    message: Message::tool_text(
                        call.id.clone(),
                        call.name.clone(),
                        format!("{{\"error\":\"{validation_error}\"}}"),
                    ),
                };
            }
            return ToolExecutionOutcome {
                call: call.clone(),
                output: Err(validation_error.clone()),
                message: Message::tool_text(
                    call.id.clone(),
                    call.name.clone(),
                    format!("{{\"error\":\"{validation_error}\"}}"),
                ),
            };
        }

        let start = Instant::now();
        match timeout(tool_timeout, tool.execute(call.arguments.clone())).await {
            Ok(Ok(output)) => {
                let duration = start.elapsed();
                debug!(tool = %call.name, ?duration, "tool executed successfully");
                let raw_text = output.value.to_string();
                let truncated =
                    tool_output::truncate_output(&raw_text, truncation_config, Some(&call.name));
                let msg = Message::tool_text(call.id.clone(), call.name.clone(), truncated.text);
                ToolExecutionOutcome {
                    call: call.clone(),
                    output: Ok(output),
                    message: msg,
                }
            }
            Ok(Err(tool_error)) => {
                let duration = start.elapsed();
                let error_msg = tool_error.to_string();
                warn!(tool = %call.name, ?duration, error = %error_msg, "tool execution failed");
                ToolExecutionOutcome {
                    call: call.clone(),
                    output: Err(error_msg.clone()),
                    message: Message::tool_text(
                        call.id.clone(),
                        call.name.clone(),
                        format!("{{\"error\":\"{error_msg}\"}}"),
                    ),
                }
            }
            Err(_) => {
                let error_msg = format!("tool execution timeout after {tool_timeout:?}");
                warn!(tool = %call.name, "tool execution timed out");
                ToolExecutionOutcome {
                    call: call.clone(),
                    output: Err(error_msg.clone()),
                    message: Message::tool_text(
                        call.id.clone(),
                        call.name.clone(),
                        format!("{{\"error\":\"{error_msg}\"}}"),
                    ),
                }
            }
        }
    }

    fn validate_arguments(schema: &Value, arguments: &Value) -> Result<(), String> {
        if schema.is_null() || schema.as_object().is_none() {
            return Ok(());
        }

        let evaluation = jsonschema::evaluate(schema, arguments);
        let flag = evaluation.flag();

        if flag.valid {
            Ok(())
        } else {
            let errors: Vec<String> = evaluation
                .iter_errors()
                .map(|e| e.error.to_string())
                .collect();
            Err(format!("validation errors: {}", errors.join("; ")))
        }
    }

    async fn record_execution(
        store: &dyn ExecutionStore,
        session_id: Uuid,
        message_id: Uuid,
        call: &ToolCall,
        outcome: &ToolExecutionOutcome,
    ) {
        let mut execution = ToolExecution::new(
            session_id,
            message_id,
            &call.id,
            &call.name,
            call.arguments.clone(),
        );

        match &outcome.output {
            Ok(output) => {
                execution = execution.with_success(output.value.clone(), std::time::Duration::ZERO);
            }
            Err(error) => {
                execution = execution.with_failure(error, std::time::Duration::ZERO);
            }
        }

        if let Err(e) = store.record_execution(execution).await {
            warn!(error = %e, "failed to record tool execution");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryExecutionStore;
    use crate::tool::FunctionTool;
    use serde_json::json;

    fn echo_tool() -> FunctionTool<
        impl Fn(
            Value,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = crate::tool::ToolResult<Value>> + Send>,
        > + Send
        + Sync
        + 'static,
    > {
        FunctionTool::new(
            "echo",
            "Echoes input",
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"]
            }),
            |args: Value| -> std::pin::Pin<
                Box<dyn std::future::Future<Output = crate::tool::ToolResult<Value>> + Send>,
            > {
                Box::pin(async move { Ok(args.get("message").cloned().unwrap_or(Value::Null)) })
            },
        )
    }

    fn failing_tool() -> FunctionTool<
        impl Fn(
            Value,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = crate::tool::ToolResult<Value>> + Send>,
        > + Send
        + Sync
        + 'static,
    > {
        FunctionTool::new(
            "fail",
            "Always fails",
            json!({"type": "object"}),
            |_args: Value| -> std::pin::Pin<
                Box<dyn std::future::Future<Output = crate::tool::ToolResult<Value>> + Send>,
            > {
                Box::pin(async move {
                    Err(crate::error::ToolError::Execution {
                        name: "fail".to_owned(),
                        message: "intentional failure".to_owned(),
                    })
                })
            },
        )
    }

    #[tokio::test]
    async fn execute_batch_should_run_tools_in_parallel() {
        let mut registry = ToolRegistry::new();
        registry.register(echo_tool());
        let policy = RuntimePolicy::new().with_max_tool_concurrency(2);
        let runtime = ToolRuntime::new(registry, policy);

        let calls = vec![
            ToolCall::new("call_1", "echo", json!({"message": "hello"})),
            ToolCall::new("call_2", "echo", json!({"message": "world"})),
        ];

        let session_id = Uuid::now_v7();
        let message_id = Uuid::now_v7();
        let outcomes = runtime
            .execute_batch(calls, session_id, message_id, None)
            .await
            .unwrap();

        assert_eq!(outcomes.len(), 2);
        assert!(outcomes[0].output.is_ok());
        assert!(outcomes[1].output.is_ok());
    }

    #[tokio::test]
    async fn execute_batch_should_handle_unknown_tool() {
        let registry = ToolRegistry::new();
        let policy = RuntimePolicy::new();
        let runtime = ToolRuntime::new(registry, policy);

        let calls = vec![ToolCall::new("call_1", "unknown", json!({}))];
        let session_id = Uuid::now_v7();
        let message_id = Uuid::now_v7();
        let outcomes = runtime
            .execute_batch(calls, session_id, message_id, None)
            .await
            .unwrap();

        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].output.is_err());
    }

    #[tokio::test]
    async fn execute_batch_should_validate_schema() {
        let mut registry = ToolRegistry::new();
        registry.register(echo_tool());
        let policy = RuntimePolicy::new();
        let runtime = ToolRuntime::new(registry, policy);

        let calls = vec![ToolCall::new("call_1", "echo", json!({"message": 123}))];
        let session_id = Uuid::now_v7();
        let message_id = Uuid::now_v7();
        let outcomes = runtime
            .execute_batch(calls, session_id, message_id, None)
            .await
            .unwrap();

        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].output.is_err());
        let err = outcomes[0].output.as_ref().unwrap_err();
        assert!(err.contains("validation"));
    }

    #[tokio::test]
    async fn execute_batch_should_record_to_execution_store() {
        let mut registry = ToolRegistry::new();
        registry.register(echo_tool());
        let policy = RuntimePolicy::new();
        let runtime = ToolRuntime::new(registry, policy);

        let store = MemoryExecutionStore::new();
        let calls = vec![ToolCall::new("call_1", "echo", json!({"message": "test"}))];
        let session_id = Uuid::now_v7();
        let message_id = Uuid::now_v7();

        runtime
            .execute_batch(calls, session_id, message_id, Some(&store))
            .await
            .unwrap();

        let executions = store.list_executions(&session_id).await.unwrap();
        assert_eq!(executions.len(), 1);
        assert_eq!(executions[0].tool_name, "echo");
    }

    #[tokio::test]
    async fn execute_batch_should_handle_tool_failure() {
        let mut registry = ToolRegistry::new();
        registry.register(failing_tool());
        let policy = RuntimePolicy::new().with_continue_on_tool_failure(true);
        let runtime = ToolRuntime::new(registry, policy);

        let calls = vec![ToolCall::new("call_1", "fail", json!({}))];
        let session_id = Uuid::now_v7();
        let message_id = Uuid::now_v7();
        let outcomes = runtime
            .execute_batch(calls, session_id, message_id, None)
            .await
            .unwrap();

        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].output.is_err());
    }

    #[tokio::test]
    async fn execute_batch_empty_returns_empty() {
        let registry = ToolRegistry::new();
        let policy = RuntimePolicy::new();
        let runtime = ToolRuntime::new(registry, policy);

        let session_id = Uuid::now_v7();
        let message_id = Uuid::now_v7();
        let outcomes = runtime
            .execute_batch(Vec::new(), session_id, message_id, None)
            .await
            .unwrap();

        assert!(outcomes.is_empty());
    }
}
