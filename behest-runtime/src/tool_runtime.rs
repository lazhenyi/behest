//! Tool runtime layer with validation, timeout, and bounded parallelism.
//!
//! [`ToolRuntime`] wraps a [`ToolRegistry`] and enforces [`RuntimePolicy`]
//! constraints during tool execution: JSON schema validation, per-tool
//! timeouts, bounded concurrency, and execution recording to an
//! [`ExecutionStore`].

use std::sync::Arc;
use std::time::Instant;

use behest_tool::Tool;
use futures_util::future::join_all;
use serde_json::Value;
use tokio::sync::Semaphore;
use tokio::time::timeout;
use tracing::{debug, warn};
use uuid::Uuid;

use super::error::{RuntimeError, RuntimeResult};
use super::policy::RuntimePolicy;
use super::tool_output::{self, ToolOutputConfig};
use super::tool_scope::ScopedToolRegistry;
use behest_core::error::ToolError;
use behest_provider::{Message, ToolCall};
use behest_store::{ExecutionStore, ToolExecution};
use behest_tool::{ToolOutput, ToolRegistry};

/// Outcome of running a single tool call within the runtime.
///
/// Carries the original [`ToolCall`], the execution result (success or error),
/// and the [`Message`] produced for the conversation history.
#[derive(Debug)]
pub struct ToolExecutionOutcome {
    /// The original tool call.
    pub call: ToolCall,
    /// Output if successful, or the tool error if failed.
    pub output: Result<ToolOutput, ToolError>,
    /// The resulting message for the conversation.
    pub message: Message,
}

/// Runtime layer that orchestrates tool execution with policy enforcement.
pub struct ToolRuntime {
    registry: Arc<ScopedToolRegistry>,
    policy: RuntimePolicy,
    semaphore: Arc<Semaphore>,
}

impl ToolRuntime {
    /// Creates a new tool runtime wrapping the given registry.
    #[must_use]
    pub fn new(registry: ToolRegistry, policy: RuntimePolicy) -> Self {
        let mut policy = policy;
        policy.max_tool_concurrency = policy.max_tool_concurrency.max(1);
        let semaphore = Arc::new(Semaphore::new(policy.max_tool_concurrency));
        Self {
            registry: Arc::new(ScopedToolRegistry::new(registry)),
            policy,
            semaphore,
        }
    }

    /// Returns a reference to the underlying scoped tool registry.
    #[must_use]
    pub fn registry(&self) -> &Arc<ScopedToolRegistry> {
        &self.registry
    }

    /// Returns a mutable reference to the underlying scoped tool registry.
    pub fn registry_mut(&mut self) -> &mut Arc<ScopedToolRegistry> {
        &mut self.registry
    }

    /// Returns the runtime policy.
    #[must_use]
    pub fn policy(&self) -> &RuntimePolicy {
        &self.policy
    }

    /// Registers a tool at the base level of the scoped registry.
    ///
    /// Returns the previously registered tool with the same name, if any.
    pub fn register_tool(&self, tool: Arc<dyn Tool>) -> Option<Arc<dyn Tool>> {
        self.registry.base().register_arc(tool)
    }

    /// Unregisters a tool from the base level of the scoped registry.
    ///
    /// Returns the removed tool if it existed, or [`None`] if no tool with
    /// `name` was registered.
    #[must_use]
    pub fn unregister_tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.registry.unregister_from_base(name)
    }

    /// Executes a batch of tool calls with bounded parallelism, timeout,
    /// validation, and optional execution recording.
    ///
    /// Tool calls are partitioned by [`Tool::is_concurrency_safe`]:
    /// concurrent-safe tools execute in parallel (bounded by semaphore),
    /// while exclusive tools execute sequentially, one at a time.
    /// Results are merged in the original call order.
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

        let call_count = calls.len();
        let indexed_calls: Vec<(usize, ToolCall)> = calls.into_iter().enumerate().collect();

        let (concurrent_group, exclusive_group): (Vec<_>, Vec<_>) =
            indexed_calls.into_iter().partition(|(_, call)| {
                self.registry
                    .get(&call.name)
                    .is_some_and(|tool| tool.is_concurrency_safe())
            });

        let mut results: Vec<Option<ToolExecutionOutcome>> =
            (0..call_count).map(|_| None).collect();

        if !concurrent_group.is_empty() {
            let concurrent_results = {
                let futures: Vec<_> = concurrent_group
                    .into_iter()
                    .map(|(idx, call)| {
                        let sem = Arc::clone(&self.semaphore);
                        let registry = self.registry.clone();
                        let tool_timeout = self.policy.tool_timeout;
                        let continue_on_failure = self.policy.continue_on_tool_failure;
                        let truncation_config = self.policy.tool_output.clone();

                        async move {
                            let _permit =
                                sem.acquire().await.map_err(|_| RuntimeError::ToolTimeout {
                                    tool: call.name.clone(),
                                })?;

                            let outcome = Self::execute_single(
                                &registry,
                                &call,
                                tool_timeout,
                                continue_on_failure,
                                &truncation_config,
                            )
                            .await?;
                            Ok::<(usize, ToolExecutionOutcome), RuntimeError>((idx, outcome))
                        }
                    })
                    .collect();

                join_all(futures).await
            };

            for res in concurrent_results {
                let (idx, outcome) = res?;
                if let Some(store) = execution_store {
                    Self::record_execution(store, session_id, message_id, &outcome.call, &outcome)
                        .await;
                }
                results[idx] = Some(outcome);
            }
        }

        for (idx, call) in exclusive_group {
            let _permit =
                self.semaphore
                    .acquire()
                    .await
                    .map_err(|_| RuntimeError::ToolTimeout {
                        tool: call.name.clone(),
                    })?;

            let outcome = Self::execute_single(
                &self.registry,
                &call,
                self.policy.tool_timeout,
                self.policy.continue_on_tool_failure,
                &self.policy.tool_output,
            )
            .await?;

            if let Some(store) = execution_store {
                Self::record_execution(store, session_id, message_id, &call, &outcome).await;
            }

            results[idx] = Some(outcome);
        }

        Ok(results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| unreachable!("all indices populated")))
            .collect())
    }

    async fn execute_single(
        registry: &ScopedToolRegistry,
        call: &ToolCall,
        tool_timeout: std::time::Duration,
        continue_on_failure: bool,
        truncation_config: &ToolOutputConfig,
    ) -> RuntimeResult<ToolExecutionOutcome> {
        let Some(tool) = registry.get(&call.name) else {
            let error_msg = format!("tool not found: {}", call.name);
            warn!(tool = %call.name, "tool not found");
            let error = ToolError::NotFound {
                name: call.name.clone(),
            };
            if !continue_on_failure {
                return Err(error.into());
            }

            return Ok(ToolExecutionOutcome {
                call: call.clone(),
                output: Err(error),
                message: Message::tool_text(
                    call.id.clone(),
                    call.name.clone(),
                    format!("{{\"error\":\"{error_msg}\"}}"),
                ),
            });
        };

        if let Err(validation_error) =
            Self::validate_arguments(&tool.parameters_schema(), &call.arguments)
        {
            debug!(tool = %call.name, error = %validation_error, "schema validation failed");
            let err = ToolError::InvalidArguments {
                name: call.name.clone(),
                message: validation_error.clone(),
            };
            if !continue_on_failure {
                return Err(err.into());
            }
            return Ok(ToolExecutionOutcome {
                call: call.clone(),
                output: Err(err),
                message: Message::tool_text(
                    call.id.clone(),
                    call.name.clone(),
                    format!("{{\"error\":\"{validation_error}\"}}"),
                ),
            });
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
                Ok(ToolExecutionOutcome {
                    call: call.clone(),
                    output: Ok(output),
                    message: msg,
                })
            }
            Ok(Err(tool_error)) => {
                let duration = start.elapsed();
                let error_msg = tool_error.to_string();
                warn!(tool = %call.name, ?duration, error = %error_msg, "tool execution failed");
                if !continue_on_failure {
                    return Err(tool_error.into());
                }

                Ok(ToolExecutionOutcome {
                    call: call.clone(),
                    output: Err(tool_error),
                    message: Message::tool_text(
                        call.id.clone(),
                        call.name.clone(),
                        format!("{{\"error\":\"{error_msg}\"}}"),
                    ),
                })
            }
            Err(_) => {
                let error_msg = format!("tool execution timeout after {tool_timeout:?}");
                warn!(tool = %call.name, "tool execution timed out");
                if !continue_on_failure {
                    return Err(RuntimeError::ToolTimeout {
                        tool: call.name.clone(),
                    });
                }

                Ok(ToolExecutionOutcome {
                    call: call.clone(),
                    output: Err(ToolError::Execution {
                        name: call.name.clone(),
                        message: error_msg.clone(),
                    }),
                    message: Message::tool_text(
                        call.id.clone(),
                        call.name.clone(),
                        format!("{{\"error\":\"{error_msg}\"}}"),
                    ),
                })
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
                execution = execution.with_failure(error.to_string(), std::time::Duration::ZERO);
            }
        }

        if let Err(e) = store.record_execution(execution).await {
            warn!(error = %e, "failed to record tool execution");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::type_complexity, clippy::expect_used)]
mod tests {
    use super::*;
    use behest_store::memory::MemoryExecutionStore;
    use behest_tool::FunctionTool;
    use serde_json::json;

    fn echo_tool() -> FunctionTool {
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
                Box<dyn std::future::Future<Output = behest_tool::ToolResult<Value>> + Send>,
            > {
                Box::pin(async move { Ok(args.get("message").cloned().unwrap_or(Value::Null)) })
            },
        )
    }

    fn failing_tool() -> FunctionTool {
        FunctionTool::new(
            "fail",
            "Always fails",
            json!({"type": "object"}),
            |_args: Value| -> std::pin::Pin<
                Box<dyn std::future::Future<Output = behest_tool::ToolResult<Value>> + Send>,
            > {
                Box::pin(async move {
                    Err(ToolError::Execution {
                        name: "fail".to_owned(),
                        message: "intentional failure".to_owned(),
                    })
                })
            },
        )
    }

    #[tokio::test]
    async fn execute_batch_should_run_tools_in_parallel() {
        let registry = ToolRegistry::new();
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
        let registry = ToolRegistry::new();
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
        assert!(err.to_string().contains("validation"));
    }

    #[tokio::test]
    async fn execute_batch_should_record_to_execution_store() {
        let registry = ToolRegistry::new();
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
        let registry = ToolRegistry::new();
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

    #[tokio::test]
    async fn execute_batch_partitions_concurrent_and_exclusive() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let concurrent_flag = Arc::new(AtomicBool::new(false));
        let exclusive_flag = Arc::new(AtomicBool::new(false));

        let con_flag = Arc::clone(&concurrent_flag);
        let concurrent_tool = FunctionTool::new(
            "concurrent",
            "Safe for concurrency",
            json!({"type": "object"}),
            move |_args: Value| -> std::pin::Pin<
                Box<dyn std::future::Future<Output = behest_tool::ToolResult<Value>> + Send>,
            > {
                let flag = Arc::clone(&con_flag);
                Box::pin(async move {
                    flag.store(true, Ordering::SeqCst);
                    Ok(Value::Null)
                })
            },
        )
        .concurrency_safe();

        let exc_flag = Arc::clone(&exclusive_flag);
        let exclusive_tool = FunctionTool::new(
            "exclusive",
            "Not safe for concurrency",
            json!({"type": "object"}),
            move |_args: Value| -> std::pin::Pin<
                Box<dyn std::future::Future<Output = behest_tool::ToolResult<Value>> + Send>,
            > {
                let flag = Arc::clone(&exc_flag);
                Box::pin(async move {
                    flag.store(true, Ordering::SeqCst);
                    Ok(Value::Null)
                })
            },
        );
        // exclusive_tool is NOT .concurrency_safe() — defaults to false

        let registry = ToolRegistry::new();
        registry.register(concurrent_tool);
        registry.register(exclusive_tool);

        let policy = RuntimePolicy::new().with_max_tool_concurrency(4);
        let runtime = ToolRuntime::new(registry, policy);

        let calls = vec![
            ToolCall::new("call_1", "concurrent", json!({})),
            ToolCall::new("call_2", "exclusive", json!({})),
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
        assert!(concurrent_flag.load(Ordering::SeqCst));
        assert!(exclusive_flag.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn execute_batch_preserves_result_order() {
        let tool_a = FunctionTool::new(
            "a",
            "Tool A",
            json!({"type": "object"}),
            |_args: Value| -> std::pin::Pin<
                Box<dyn std::future::Future<Output = behest_tool::ToolResult<Value>> + Send>,
            > { Box::pin(async move { Ok(Value::String("A".into())) }) },
        )
        .concurrency_safe();

        let tool_b = FunctionTool::new(
            "b",
            "Tool B",
            json!({"type": "object"}),
            |_args: Value| -> std::pin::Pin<
                Box<dyn std::future::Future<Output = behest_tool::ToolResult<Value>> + Send>,
            > { Box::pin(async move { Ok(Value::String("B".into())) }) },
        );
        // tool_b is NOT concurrency_safe

        let registry = ToolRegistry::new();
        registry.register(tool_a);
        registry.register(tool_b);

        let policy = RuntimePolicy::new().with_max_tool_concurrency(4);
        let runtime = ToolRuntime::new(registry, policy);

        let calls = vec![
            ToolCall::new("call_1", "a", json!({})),
            ToolCall::new("call_2", "b", json!({})),
        ];

        let session_id = Uuid::now_v7();
        let message_id = Uuid::now_v7();
        let outcomes = runtime
            .execute_batch(calls, session_id, message_id, None)
            .await
            .unwrap();

        assert_eq!(outcomes.len(), 2);
        // Results must be in original call order: a first, then b
        assert_eq!(outcomes[0].call.name, "a");
        assert_eq!(outcomes[1].call.name, "b");
    }

    #[tokio::test]
    async fn execute_batch_should_propagate_tool_failure_when_continue_disabled() {
        let registry = ToolRegistry::new();
        registry.register(failing_tool());
        let policy = RuntimePolicy::new().with_continue_on_tool_failure(false);
        let runtime = ToolRuntime::new(registry, policy);

        let calls = vec![ToolCall::new("call_1", "fail", json!({}))];
        let session_id = Uuid::now_v7();
        let message_id = Uuid::now_v7();
        let result = runtime
            .execute_batch(calls, session_id, message_id, None)
            .await;

        assert!(matches!(result, Err(RuntimeError::Tool(_))));
    }

    #[tokio::test]
    async fn execute_batch_should_propagate_invalid_arguments_when_continue_disabled() {
        let registry = ToolRegistry::new();
        registry.register(echo_tool());
        let policy = RuntimePolicy::new().with_continue_on_tool_failure(false);
        let runtime = ToolRuntime::new(registry, policy);

        let calls = vec![ToolCall::new("call_1", "echo", json!({"message": 123}))];
        let session_id = Uuid::now_v7();
        let message_id = Uuid::now_v7();
        let result = runtime
            .execute_batch(calls, session_id, message_id, None)
            .await;

        assert!(matches!(result, Err(RuntimeError::Tool(_))));
    }

    #[tokio::test]
    async fn execute_batch_with_zero_concurrency_should_not_hang() {
        let registry = ToolRegistry::new();
        registry.register(echo_tool());
        let policy = RuntimePolicy::new().with_max_tool_concurrency(0);
        let runtime = ToolRuntime::new(registry, policy);

        let calls = vec![ToolCall::new("call_1", "echo", json!({"message": "hello"}))];
        let session_id = Uuid::now_v7();
        let message_id = Uuid::now_v7();
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            runtime.execute_batch(calls, session_id, message_id, None),
        )
        .await;

        let outcomes = result.expect("tool runtime should not hang").unwrap();
        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].output.is_ok());
    }
}
