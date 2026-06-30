//! Async runtime glue connecting the sans-IO state machine to real I/O.
//!
//! The [`Runtime`] struct orchestrates the full agent execution loop:
//!
//! 1. Receives a user message
//! 2. Builds context (system prompt + history)
//! 3. Drives the sans-IO state machine via [`transition`]
//! 4. Executes actions: model calls, tool execution, memory compaction, approval
//! 5. Emits events through the hook stack
//! 6. Returns when the run finishes or is aborted
//!
//! # Architecture
//!
//! ```text
//! User Message
//!     │
//!     ▼
//! Runtime::run()
//!     │
//!     ├─► build context (system prompt + history + RAG)
//!     ├─► transition(state, input) → (new_state, actions)
//!     │       │
//!     │       ├─► RequestModel  → provider.stream() → events → ModelCompleted
//!     │       ├─► ExecuteTool   → tool_registry.execute() → ToolResultReceived
//!     │       ├─► RequestToolApproval → approval_gate.request() → await decision
//!     │       ├─► CompactMemory → memory compaction → MemoryCompactionCompleted
//!     │       ├─► FinishRun     → return RunOutput
//!     │       └─► AbortRun      → return error
//!     │
//!     └─► emit events → HookStack.dispatch() → EventActions
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]

use behest_approval::{ApprovalGate, ApprovalPolicy};
use behest_context::{
    EventSink, RunBudget, RunContextImpl, SessionContextImpl, SessionState, ToolContextImpl,
};
use behest_core::error::ProviderError;
use behest_core::id::RunId;
use behest_core::message::{ChatRequest, ContentPart, FinishReason, Message, TokenUsage};
use behest_core::run::{RunConfig, RunInput, RunState, transition};
use behest_core::tool_types::ToolCall;
use behest_event::{AgentEvent, HookStack};
use behest_memory::ConversationMemory;
use behest_tool::{ToolExecutionStrategy, ToolRegistry};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

mod error;
pub use error::{RuntimeError, RuntimeResult};

/// Configuration for the runtime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Maximum number of tool-calling iterations.
    pub max_iterations: usize,
    /// Maximum tool execution concurrency.
    pub max_tool_concurrency: usize,
    /// Tool execution strategy (sequential, parallel, auto).
    pub tool_execution_strategy: ToolExecutionStrategy,
    /// Timeout for individual tool executions.
    pub tool_timeout: Duration,
    /// Timeout for provider calls.
    pub provider_timeout: Duration,
    /// Whether to continue after tool failures.
    pub continue_on_tool_failure: bool,
    /// Whether to retry on provider errors.
    pub retry_on_provider_error: bool,
    /// Maximum number of retries for provider errors.
    pub max_retries: usize,
    /// Approval policy for tool calls.
    pub approval_policy: ApprovalPolicy,
    /// Whether to enable memory compaction.
    pub enable_compaction: bool,
    /// Event channel capacity.
    pub event_channel_capacity: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            max_tool_concurrency: 4,
            tool_execution_strategy: ToolExecutionStrategy::Auto,
            tool_timeout: Duration::from_secs(30),
            provider_timeout: Duration::from_secs(60),
            continue_on_tool_failure: true,
            retry_on_provider_error: true,
            max_retries: 2,
            approval_policy: ApprovalPolicy::AutoForReadOnly,
            enable_compaction: true,
            event_channel_capacity: 256,
        }
    }
}

/// Output from a completed run.
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// The run identifier.
    pub run_id: RunId,
    /// The session identifier.
    pub session_id: String,
    /// Number of model-calling iterations completed.
    pub iterations: usize,
    /// Why the run finished.
    pub finish_reason: FinishReason,
    /// Total token usage across all model calls.
    pub total_usage: TokenUsage,
    /// The final assistant message.
    pub final_message: Option<Message>,
}

/// The central runtime orchestrator.
///
/// Connects the sans-IO state machine to real I/O: model providers,
/// tool execution, memory management, and approval workflows.
pub struct Runtime {
    config: RuntimeConfig,
    tools: Arc<ToolRegistry>,
    memory: Arc<dyn ConversationMemory>,
    approval_gate: Arc<ApprovalGate>,
    hooks: HookStack,
    event_tx: broadcast::Sender<AgentEvent>,
}

impl Runtime {
    /// Creates a new runtime with default configuration.
    ///
    /// The runtime is parameterized by:
    /// - `tools`: registry of callable tools
    /// - `memory`: conversation memory backend
    #[must_use]
    pub fn new(tools: Arc<ToolRegistry>, memory: Arc<dyn ConversationMemory>) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            config: RuntimeConfig::default(),
            tools,
            memory,
            approval_gate: Arc::new(ApprovalGate::new()),
            hooks: HookStack::new(),
            event_tx,
        }
    }

    /// Sets the runtime configuration.
    #[must_use]
    pub fn with_config(mut self, config: RuntimeConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the approval gate.
    #[must_use]
    pub fn with_approval_gate(mut self, gate: Arc<ApprovalGate>) -> Self {
        self.approval_gate = gate;
        self
    }

    /// Adds a hook to the runtime.
    pub fn add_hook(&mut self, hook: Box<dyn behest_event::Hook>) {
        self.hooks.push(hook);
    }

    /// Subscribes to runtime events.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
    }

    /// Runs an agent execution loop.
    ///
    /// Takes a user message text, builds context, and drives the
    /// sans-IO state machine until completion or error.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] if the provider fails unrecoverably or
    /// the run is aborted.
    pub async fn run(
        &self,
        provider: &(dyn behest::provider::ChatProvider + Sync),
        session_id: &str,
        user_id: &str,
        user_message: &str,
        system_prompt: Option<&str>,
    ) -> RuntimeResult<RunOutput> {
        let run_id = RunId::new();
        let cancel = CancellationToken::new();
        let sink = EventSink::new();
        let _budget = RunBudget::new(self.config.max_iterations.checked_mul(100_000));

        info!(
            run_id = %run_id,
            session_id = %session_id,
            "Starting agent run"
        );

        // Build initial messages
        let mut messages: Vec<Message> = Vec::new();
        if let Some(prompt) = system_prompt {
            messages.push(Message::system_text(prompt));
        }
        messages.push(Message::user_text(user_message));

        // Load conversation history
        if let Ok(history) = self.memory.load(session_id).await
            && !history.is_empty()
        {
            debug!(count = history.len(), "Loaded conversation history");
            // Insert history before the current user message
            let user_msg = messages.pop();
            messages = history;
            if let Some(msg) = user_msg {
                messages.push(msg);
            }
        }

        let tool_specs = self.tools.specs();

        // State machine loop
        let mut state = RunState::Idle;
        let mut iterations: usize = 0;
        let mut total_usage = TokenUsage::new(0, 0);
        let mut final_message: Option<Message> = None;

        loop {
            iterations += 1;
            if iterations > self.config.max_iterations {
                warn!(iterations, "Exceeded max iterations");
                break;
            }

            // Build context and create chat request
            let model_name = provider.capabilities().max_output_tokens.map_or_else(
                || behest_core::id::ModelName::new("default"),
                |_| behest_core::id::ModelName::new("default"),
            );

            let mut request = ChatRequest::new(model_name);
            for msg in &messages {
                request = request.with_message(msg.clone());
            }
            for spec in &tool_specs {
                request = request.with_tool(spec.clone());
            }

            // Transition: BuildingContext → CallingModel
            let (new_state, _actions) = transition(
                &state,
                RunInput::UserMessageReceived {
                    content: vec![ContentPart::text(user_message)],
                },
                &RunConfig {
                    max_iterations: self.config.max_iterations,
                    max_tokens: None,
                    continue_on_tool_failure: self.config.continue_on_tool_failure,
                },
            );
            state = new_state;

            // Call model (with retry)
            let response = self
                .call_model_with_retry(provider, &request, &cancel)
                .await?;

            // Emit model completed event
            let event = AgentEvent::ModelCompleted {
                run_id,
                usage: response.usage.unwrap_or(TokenUsage::new(0, 0)),
            };
            self.emit_event(&event, &run_id, session_id).await;

            total_usage = TokenUsage::new(
                total_usage.input_tokens
                    + response.usage.unwrap_or(TokenUsage::new(0, 0)).input_tokens,
                total_usage.output_tokens
                    + response
                        .usage
                        .unwrap_or(TokenUsage::new(0, 0))
                        .output_tokens,
            );

            let assistant_msg = response.message.clone();
            messages.push(assistant_msg.clone());

            // Check for tool calls
            if !response.message.tool_calls().is_empty() {
                let calls = response.message.tool_calls().to_vec();

                // Execute tools
                let tool_results = self
                    .execute_tools(&calls, &run_id, session_id, user_id, &sink, &cancel)
                    .await;

                // Feed tool results back as messages
                for (call, result) in calls.iter().zip(tool_results.iter()) {
                    let result_text = match result {
                        Ok(output) => output.value.to_string(),
                        Err(e) => {
                            warn!(tool = %call.name, error = %e, "Tool execution failed");
                            format!("Error: {e}")
                        }
                    };
                    messages.push(Message::Tool {
                        tool_call_id: call.id.clone(),
                        name: call.name.clone(),
                        content: vec![ContentPart::text(&result_text)],
                    });
                }
            } else {
                // No tool calls — run is complete
                final_message = Some(assistant_msg);
                break;
            }

            // Persist to memory
            if let Err(e) = self.memory.append(session_id, messages.clone()).await {
                warn!(error = %e, "Failed to persist conversation");
            }
        }

        // Emit finish event
        let finish_event = AgentEvent::RunFinished {
            run_id,
            reason: FinishReason::Stop,
            usage: total_usage,
        };
        self.emit_event(&finish_event, &run_id, session_id).await;

        info!(
            run_id = %run_id,
            iterations,
            input_tokens = total_usage.input_tokens,
            output_tokens = total_usage.output_tokens,
            "Agent run complete"
        );

        Ok(RunOutput {
            run_id,
            session_id: session_id.to_string(),
            iterations,
            finish_reason: FinishReason::Stop,
            total_usage,
            final_message,
        })
    }

    /// Calls the model with retry logic.
    async fn call_model_with_retry(
        &self,
        provider: &(dyn behest::provider::ChatProvider + Sync),
        request: &ChatRequest,
        cancel: &CancellationToken,
    ) -> RuntimeResult<behest::provider::ChatResponse> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if cancel.is_cancelled() {
                return Err(RuntimeError::Cancelled);
            }

            let result = tokio::time::timeout(
                self.config.provider_timeout,
                provider.complete(request.clone()),
            )
            .await;

            match result {
                Ok(Ok(response)) => return Ok(response),
                Ok(Err(e)) => {
                    if !self.config.retry_on_provider_error || !e.is_retryable() {
                        return Err(RuntimeError::Provider(e));
                    }
                    warn!(attempt, error = %e, "Provider error, retrying");
                    last_error = Some(e);
                    tokio::time::sleep(Duration::from_millis(500 * 2_u64.pow(attempt as u32)))
                        .await;
                }
                Err(_elapsed) => {
                    last_error = Some(ProviderError::Timeout {
                        provider: provider.id(),
                    });
                    if !self.config.retry_on_provider_error {
                        return Err(RuntimeError::Provider(last_error.unwrap()));
                    }
                }
            }
        }

        Err(RuntimeError::Provider(last_error.unwrap_or_else(|| {
            ProviderError::Timeout {
                provider: provider.id(),
            }
        })))
    }

    /// Executes a batch of tool calls.
    async fn execute_tools(
        &self,
        calls: &[ToolCall],
        run_id: &RunId,
        session_id: &str,
        user_id: &str,
        sink: &EventSink,
        cancel: &CancellationToken,
    ) -> Vec<Result<behest_tool::ToolOutput, behest_core::error::ToolError>> {
        use std::collections::HashMap;

        let tool_arcs: Vec<_> = calls
            .iter()
            .filter_map(|c| self.tools.get(&c.name))
            .collect();

        let plan = self.config.tool_execution_strategy.plan(calls, &tool_arcs);

        let mut results: HashMap<
            String,
            Result<behest_tool::ToolOutput, behest_core::error::ToolError>,
        > = HashMap::new();

        for group in &plan.groups {
            if group.parallel && group.calls.len() > 1 {
                let handles: Vec<_> = group
                    .calls
                    .iter()
                    .map(|call| {
                        let tools = Arc::clone(&self.tools);
                        let call = call.clone();
                        let run_id = *run_id;
                        let session_id = session_id.to_string();
                        let user_id = user_id.to_string();
                        let sink = sink.clone();
                        let cancel = cancel.clone();
                        let timeout = self.config.tool_timeout;
                        let approval_gate = Arc::clone(&self.approval_gate);
                        let approval_policy = self.config.approval_policy.clone();

                        tokio::spawn(async move {
                            execute_single_tool(
                                &tools,
                                &call,
                                &run_id,
                                &session_id,
                                &user_id,
                                &sink,
                                &cancel,
                                timeout,
                                &approval_gate,
                                &approval_policy,
                            )
                            .await
                        })
                    })
                    .collect();

                for handle in handles {
                    match handle.await {
                        Ok(Ok((call_id, result))) => {
                            results.insert(call_id, result);
                        }
                        Ok(Err(_)) => {} // Task failed, skip
                        Err(e) => {
                            warn!(error = %e, "Tool execution task panicked");
                        }
                    }
                }
            } else {
                for call in &group.calls {
                    if cancel.is_cancelled() {
                        break;
                    }
                    if let Ok((call_id, result)) = execute_single_tool(
                        &self.tools,
                        call,
                        run_id,
                        session_id,
                        user_id,
                        sink,
                        cancel,
                        self.config.tool_timeout,
                        &self.approval_gate,
                        &self.config.approval_policy,
                    )
                    .await
                    {
                        results.insert(call_id, result);
                    }
                }
            }
        }

        // Return results in original call order
        calls
            .iter()
            .map(|c| {
                results.remove(&c.id).unwrap_or_else(|| {
                    Err(behest_core::error::ToolError::Execution {
                        name: c.name.clone(),
                        message: "Tool execution was skipped".to_string(),
                    })
                })
            })
            .collect()
    }

    /// Emits an event through the hook stack and broadcast channel.
    async fn emit_event(&self, event: &AgentEvent, run_id: &RunId, session_id: &str) {
        // Dispatch to hooks
        let hook_ctx = behest_context::HookContextImpl {
            app: behest_context::AppContext {
                invocation_id: run_id.to_string(),
                session_id: session_id.to_string(),
                user_id: String::new(),
                app_name: "behest".to_string(),
            },
            state: RunState::Idle,
            run_id: *run_id,
            iteration: 0,
            tokens_used: 0,
        };

        let _hook_actions = self.hooks.dispatch(event, &hook_ctx);

        // Broadcast to subscribers
        let _ = self.event_tx.send(event.clone());
    }
}

/// Executes a single tool call with timeout, approval check, and progress emission.
#[allow(clippy::too_many_arguments)]
async fn execute_single_tool(
    tools: &ToolRegistry,
    call: &ToolCall,
    run_id: &RunId,
    session_id: &str,
    user_id: &str,
    sink: &EventSink,
    cancel: &CancellationToken,
    timeout: Duration,
    approval_gate: &ApprovalGate,
    approval_policy: &ApprovalPolicy,
) -> Result<
    (
        String,
        Result<behest_tool::ToolOutput, behest_core::error::ToolError>,
    ),
    String,
> {
    let tool = match tools.get(&call.name) {
        Some(t) => t,
        None => {
            return Ok((
                call.id.clone(),
                Err(behest_core::error::ToolError::NotFound {
                    name: call.name.clone(),
                }),
            ));
        }
    };

    // Check approval
    if tool.requires_approval() && approval_policy.requires_approval(true, tool.is_read_only()) {
        let reason = tool
            .approval_reason()
            .unwrap_or_else(|| format!("Tool '{}' requires approval", call.name));
        approval_gate
            .request(call.clone(), reason, Some(timeout))
            .await;

        // Wait for approval decision
        let approval_timeout = timeout.min(Duration::from_secs(60));
        let start = tokio::time::Instant::now();
        loop {
            if start.elapsed() > approval_timeout {
                approval_gate.expire_timed_out().await;
                return Ok((
                    call.id.clone(),
                    Err(behest_core::error::ToolError::Execution {
                        name: call.name.clone(),
                        message: "Approval timed out".to_string(),
                    }),
                ));
            }
            if approval_gate.is_approved(&call.id).await {
                break;
            }
            if cancel.is_cancelled() {
                return Err("cancelled".to_string());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    // Emit tool execution started
    let start_event = AgentEvent::ToolExecutionStarted {
        run_id: *run_id,
        call_id: call.id.clone(),
        name: call.name.clone(),
    };
    let _ = sink.emit(serde_json::to_value(&start_event).unwrap_or_default());

    // Execute with timeout
    let ctx = ToolContextImpl {
        run: RunContextImpl {
            session: SessionContextImpl {
                app: behest_context::AppContext {
                    invocation_id: run_id.to_string(),
                    session_id: session_id.to_string(),
                    user_id: user_id.to_string(),
                    app_name: "behest".to_string(),
                },
                state: SessionState::new(),
            },
            run_id: *run_id,
            cancel: cancel.clone(),
            deadline: Some(tokio::time::Instant::now().into_std() + timeout),
            sink: sink.clone(),
            budget: RunBudget::new(None),
        },
        tool_call: call.clone(),
    };

    let result = tokio::time::timeout(timeout, tool.execute(&ctx, call.arguments.clone())).await;

    match result {
        Ok(Ok(output)) => {
            let completed_event = AgentEvent::ToolExecutionCompleted {
                run_id: *run_id,
                call_id: call.id.clone(),
                output: output.value.clone(),
            };
            let _ = sink.emit(serde_json::to_value(&completed_event).unwrap_or_default());
            Ok((call.id.clone(), Ok(output)))
        }
        Ok(Err(e)) => {
            let failed_event = AgentEvent::ToolExecutionFailed {
                run_id: *run_id,
                call_id: call.id.clone(),
                error: e.to_string(),
            };
            let _ = sink.emit(serde_json::to_value(&failed_event).unwrap_or_default());
            Ok((call.id.clone(), Err(e)))
        }
        Err(_elapsed) => Ok((
            call.id.clone(),
            Err(behest_core::error::ToolError::Execution {
                name: call.name.clone(),
                message: "Tool execution timed out".to_string(),
            }),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock provider for testing.
    struct MockProvider {
        responses: std::sync::Mutex<Vec<behest::provider::ChatResponse>>,
    }

    impl MockProvider {
        fn new(responses: Vec<behest::provider::ChatResponse>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
            }
        }
    }

    #[async_trait::async_trait]
    impl behest::provider::ChatProvider for MockProvider {
        fn id(&self) -> behest_core::id::ProviderId {
            behest_core::id::ProviderId::new("mock")
        }

        fn capabilities(&self) -> behest_core::capabilities::ProviderCapabilities {
            behest_core::capabilities::ProviderCapabilities {
                chat: true,
                chat_stream: false,
                tool_calling: true,
                ..behest_core::capabilities::ProviderCapabilities::empty()
            }
        }

        async fn complete(
            &self,
            _request: ChatRequest,
        ) -> behest::provider::ProviderResult<behest::provider::ChatResponse> {
            let mut responses = self.responses.lock().unwrap();
            Ok(responses.remove(0))
        }
    }

    #[tokio::test]
    async fn runtime_simple_text_response() {
        let tools = Arc::new(ToolRegistry::new());
        let memory = Arc::new(behest_memory::InMemoryConversationMemory::new());

        let provider = MockProvider::new(vec![behest::provider::ChatResponse {
            provider: behest_core::id::ProviderId::new("mock"),
            model: behest_core::id::ModelName::new("mock"),
            message: Message::assistant_text("Hello! How can I help?"),
            finish_reason: FinishReason::Stop,
            usage: Some(TokenUsage::new(10, 5)),
            raw: None,
        }]);

        let runtime = Runtime::new(tools, memory);

        let output = runtime
            .run(&provider, "test-session", "test-user", "Hi!", None)
            .await
            .unwrap();

        assert_eq!(output.iterations, 1);
        assert_eq!(output.total_usage.input_tokens, 10);
        assert_eq!(output.total_usage.output_tokens, 5);
    }

    #[tokio::test]
    async fn runtime_emits_events() {
        let tools = Arc::new(ToolRegistry::new());
        let memory = Arc::new(behest_memory::InMemoryConversationMemory::new());

        let provider = MockProvider::new(vec![behest::provider::ChatResponse {
            provider: behest_core::id::ProviderId::new("mock"),
            model: behest_core::id::ModelName::new("mock"),
            message: Message::assistant_text("Done"),
            finish_reason: FinishReason::Stop,
            usage: Some(TokenUsage::new(5, 2)),
            raw: None,
        }]);

        let runtime = Runtime::new(tools, memory);
        let mut rx = runtime.subscribe();

        let _output = runtime
            .run(&provider, "sess", "user", "Hello", None)
            .await
            .unwrap();

        // Should receive at least ModelCompleted and RunFinished events
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        assert!(!events.is_empty(), "Should emit events");
    }
}
