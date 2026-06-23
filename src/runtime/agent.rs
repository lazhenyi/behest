//! Agent runtime — streaming-first execution kernel.
//!
//! [`AgentRuntime`] orchestrates the full agent loop: context building,
//! model invocation (streaming with non-streaming fallback), tool execution,
//! session persistence, and event emission.

use std::sync::Arc;

#[cfg(feature = "queue")]
use crate::queue::EventPublisher;

use chrono::Utc;
use futures_util::StreamExt;
use tokio::sync::broadcast;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::provider::{
    ChatRequest, ChatResponse, ChatStreamEvent, FinishReason, Message, TokenUsage, ToolCall,
};

use super::accumulator::StreamAccumulator;
use super::compaction::CompactionService;
use super::context::ContextPipeline;
use super::error::{RuntimeError, RuntimeResult};
use super::event::{
    AgentEvent, ContextBuilt, MessageCommitted, ModelStarted, RunCompleted, RunFailed, RunStarted,
    TextDelta, ToolCallCompleted, ToolCallDelta, ToolCallStarted as ToolCallStartedEvent,
    UsageRecorded,
};
use super::policy::RuntimePolicy;
use super::run::{RunId, RunRecord, RunRequest, RunStatus};
use super::store::{RunEventRecord, RuntimeStore};
use super::tool::ToolRuntime;

/// Streaming-first agent runtime kernel.
///
/// Ties together provider registry, context pipeline, tool runtime,
/// compaction service, and persistent stores into a complete agent
/// execution loop.
pub struct AgentRuntime {
    providers: crate::provider::ProviderRegistry,
    context: ContextPipeline,
    tools: ToolRuntime,
    store: Arc<RuntimeStore>,
    policy: RuntimePolicy,
    compaction: CompactionService,
    event_tx: broadcast::Sender<AgentEvent>,
    #[cfg(feature = "queue")]
    event_publisher: Option<Arc<dyn EventPublisher>>,
}

impl AgentRuntime {
    /// Creates a new agent runtime.
    #[must_use]
    pub fn new(
        providers: crate::provider::ProviderRegistry,
        context: ContextPipeline,
        tools: ToolRuntime,
        store: Arc<RuntimeStore>,
        policy: RuntimePolicy,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        let compaction = CompactionService::new(providers.clone(), policy.compaction.clone());
        Self {
            providers,
            context,
            tools,
            store,
            policy,
            compaction,
            event_tx,
            #[cfg(feature = "queue")]
            event_publisher: None,
        }
    }

    /// Sets an external event publisher for the agent runtime.
    ///
    /// When set, every [`AgentEvent`] emitted during a run will also be
    /// published to the configured [`EventPublisher`] via fire-and-forget.
    #[cfg(feature = "queue")]
    #[must_use]
    pub fn with_event_publisher(mut self, publisher: Arc<dyn EventPublisher>) -> Self {
        self.event_publisher = Some(publisher);
        self
    }

    /// Subscribes to runtime events.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
    }

    /// Returns the runtime policy.
    #[must_use]
    pub fn policy(&self) -> &RuntimePolicy {
        &self.policy
    }

    /// Returns the tool runtime.
    #[must_use]
    pub fn tools(&self) -> &ToolRuntime {
        &self.tools
    }

    /// Returns the provider registry.
    #[must_use]
    pub fn providers(&self) -> &crate::provider::ProviderRegistry {
        &self.providers
    }

    /// Returns the context pipeline.
    #[must_use]
    pub fn context(&self) -> &ContextPipeline {
        &self.context
    }

    /// Returns the runtime store.
    #[must_use]
    pub fn store(&self) -> &Arc<RuntimeStore> {
        &self.store
    }

    /// Executes an agent run to completion.
    ///
    /// The run loop:
    /// 1. Creates or loads a session
    /// 2. Persists the user message
    /// 3. Iterates: build context → call model → persist response → execute tools → repeat
    /// 4. Returns the final run ID and finish reason
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` on provider, store, context, or policy violations.
    #[allow(clippy::too_many_lines)]
    pub async fn run(&self, request: RunRequest) -> RuntimeResult<RunOutput> {
        let run_id = RunId::new();
        let session_id = self.store.ensure_session(request.session_id).await?;

        let run_record = RunRecord::new(
            run_id,
            session_id,
            request.provider.clone(),
            request.model.clone(),
            request.metadata.clone(),
        );
        self.store.runs().create_run(run_record).await?;

        self.emit(&AgentEvent::RunStarted(RunStarted {
            run_id,
            session_id,
            timestamp: Utc::now(),
        }));
        self.update_status(run_id, RunStatus::SessionLoaded).await?;

        let user_message = Message::user_text(&request.input);
        let user_msg_id = self.store.append_message(session_id, &user_message).await?;
        debug!(%run_id, %user_msg_id, "user message persisted");

        let provider = self
            .providers
            .chat(&request.provider)
            .ok_or_else(|| RuntimeError::ProviderNotFound(request.provider.to_string()))?;

        let tool_specs = self.tools.registry().specs();
        let has_tools = !tool_specs.is_empty();

        let mut iteration = 0usize;
        let mut total_usage = TokenUsage::new(0, 0);
        let mut last_finish;

        loop {
            iteration += 1;
            if iteration > self.policy.max_iterations {
                let err = RuntimeError::IterationLimitExceeded(self.policy.max_iterations);
                self.fail_run(run_id, &err).await;
                return Err(err);
            }

            if let Some(budget) = self.policy.max_tokens {
                let budget_u64 = budget as u64;
                if total_usage.total_tokens >= budget_u64 {
                    #[allow(clippy::cast_possible_truncation)]
                    let err = RuntimeError::TokenBudgetExceeded {
                        used: total_usage.total_tokens as usize,
                        limit: budget,
                    };
                    self.fail_run(run_id, &err).await;
                    return Err(err);
                }
            }

            self.update_status(run_id, RunStatus::BuildingContext)
                .await?;

            // --- Compaction: proactive overflow check ---
            if self.policy.compaction.auto {
                let caps = provider.capabilities();
                if let (Some(model_ctx), Some(max_out)) =
                    (caps.max_input_tokens, caps.max_output_tokens)
                {
                    let records = self
                        .store
                        .sessions()
                        .list_messages(&session_id)
                        .await
                        .map_err(RuntimeError::from)?;

                    if let Some(result) = self
                        .compaction
                        .compact_if_needed(
                            &records,
                            model_ctx,
                            max_out,
                            self.store.sessions(),
                            session_id,
                        )
                        .await?
                    {
                        debug!(
                            run_id = %run_id,
                            tokens_saved = result.tokens_saved,
                            "proactive compaction completed"
                        );
                    }
                }
            }

            let chat_request = self
                .context
                .build(
                    &self.store,
                    session_id,
                    request.model.clone(),
                    if iteration == 1 {
                        Some(&request.input)
                    } else {
                        None
                    },
                    if has_tools { Some(&tool_specs) } else { None },
                )
                .await?;

            self.emit(&AgentEvent::ContextBuilt(ContextBuilt {
                run_id,
                message_count: chat_request.messages.len(),
                timestamp: Utc::now(),
            }));

            self.update_status(run_id, RunStatus::CallingModel).await?;

            self.emit(&AgentEvent::ModelStarted(ModelStarted {
                run_id,
                provider: request.provider.clone(),
                model: request.model.clone(),
                iteration,
                timestamp: Utc::now(),
            }));

            let (assistant_message, finish_reason, usage) =
                match self.call_model(&provider, chat_request, run_id).await {
                    Ok(result) => result,
                    Err(RuntimeError::Provider(ref e)) if e.is_context_overflow() => {
                        // Provider reported context overflow — compact and retry
                        if self.policy.compaction.auto {
                            let caps = provider.capabilities();
                            let model_ctx = caps.max_input_tokens.unwrap_or(128_000);
                            let max_out = caps.max_output_tokens.unwrap_or(16_384);
                            let records = self
                                .store
                                .sessions()
                                .list_messages(&session_id)
                                .await
                                .map_err(RuntimeError::from)?;

                            let result = self
                                .compaction
                                .compact_after_overflow(
                                    &records,
                                    model_ctx,
                                    max_out,
                                    self.store.sessions(),
                                    session_id,
                                )
                                .await?;
                            debug!(
                                run_id = %run_id,
                                tokens_saved = result.tokens_saved,
                                "reactive compaction after provider overflow"
                            );
                        }
                        continue;
                    }
                    Err(e) => return Err(e),
                };

            if let Some(u) = &usage {
                total_usage = TokenUsage::new(
                    total_usage.input_tokens + u.input_tokens,
                    total_usage.output_tokens + u.output_tokens,
                );
                self.emit(&AgentEvent::UsageRecorded(UsageRecorded {
                    run_id,
                    usage: *u,
                    timestamp: Utc::now(),
                }));
            }

            last_finish = finish_reason.clone();

            let assistant_msg_id = self
                .store
                .append_message(session_id, &assistant_message)
                .await?;

            self.emit(&AgentEvent::AssistantMessageCommitted(MessageCommitted {
                run_id,
                message_id: assistant_msg_id,
                timestamp: Utc::now(),
            }));

            let tool_calls = match &assistant_message {
                Message::Assistant { tool_calls, .. } if !tool_calls.is_empty() => {
                    tool_calls.clone()
                }
                _ => Vec::new(),
            };

            if tool_calls.is_empty() {
                break;
            }

            if !matches!(finish_reason, FinishReason::ToolCalls) {
                break;
            }

            self.update_status(run_id, RunStatus::WaitingForTools)
                .await?;

            let outcomes = self
                .tools
                .execute_batch(
                    tool_calls,
                    session_id,
                    assistant_msg_id,
                    Some(self.store.executions()),
                )
                .await?;

            for outcome in &outcomes {
                let tool_msg_id = self
                    .store
                    .append_message(session_id, &outcome.message)
                    .await?;

                self.emit(&AgentEvent::ToolMessageCommitted(MessageCommitted {
                    run_id,
                    message_id: tool_msg_id,
                    timestamp: Utc::now(),
                }));
            }

            self.update_status(run_id, RunStatus::Persisting).await?;

            if !matches!(last_finish, FinishReason::ToolCalls) {
                break;
            }
        }

        self.update_status(run_id, RunStatus::Completed).await?;

        self.emit(&AgentEvent::RunCompleted(RunCompleted {
            run_id,
            finish_reason: last_finish.clone(),
            iterations: iteration,
            timestamp: Utc::now(),
        }));

        info!(%run_id, iterations = iteration, "run completed");

        Ok(RunOutput {
            run_id,
            session_id,
            iterations: iteration,
            finish_reason: last_finish,
            total_usage,
        })
    }

    async fn call_model(
        &self,
        provider: &Arc<dyn crate::provider::ChatProvider>,
        request: ChatRequest,
        run_id: RunId,
    ) -> RuntimeResult<(Message, FinishReason, Option<TokenUsage>)> {
        let caps = provider.capabilities();

        if caps.chat_stream {
            match self.call_streaming(provider, request.clone(), run_id).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    warn!(error = %e, "streaming failed, falling back to complete");
                }
            }
        }

        self.call_complete(provider, request, run_id).await
    }

    async fn call_streaming(
        &self,
        provider: &Arc<dyn crate::provider::ChatProvider>,
        request: ChatRequest,
        run_id: RunId,
    ) -> RuntimeResult<(Message, FinishReason, Option<TokenUsage>)> {
        let stream = timeout(self.policy.provider_timeout, provider.stream(request))
            .await
            .map_err(|_| RuntimeError::ProviderNotFound("provider timeout".to_owned()))?
            .map_err(RuntimeError::from)?;

        let mut accumulator = StreamAccumulator::new();
        let mut finish_reason = FinishReason::Stop;
        let mut usage: Option<TokenUsage> = None;

        tokio::pin!(stream);

        while let Some(event_result) = stream.next().await {
            let event = event_result.map_err(RuntimeError::from)?;

            match &event {
                ChatStreamEvent::TextDelta { delta } => {
                    accumulator.append_text(delta);
                    self.emit(&AgentEvent::TextDelta(TextDelta {
                        run_id,
                        delta: delta.clone(),
                        timestamp: Utc::now(),
                    }));
                }
                ChatStreamEvent::ToolCallStarted { id, name } => {
                    accumulator.start_tool_call(id.clone(), name.clone());
                    self.emit(&AgentEvent::ToolCallStarted(ToolCallStartedEvent {
                        run_id,
                        call_id: id.clone(),
                        tool_name: name.clone(),
                        timestamp: Utc::now(),
                    }));
                }
                ChatStreamEvent::ToolCallArgumentsDelta { id, delta } => {
                    accumulator.append_tool_arguments(id, delta);
                    self.emit(&AgentEvent::ToolCallDelta(ToolCallDelta {
                        run_id,
                        call_id: id.clone(),
                        delta: delta.clone(),
                        timestamp: Utc::now(),
                    }));
                }
                ChatStreamEvent::ToolCallCompleted { call } => {
                    self.emit(&AgentEvent::ToolCallCompleted(ToolCallCompleted {
                        run_id,
                        call: call.clone(),
                        timestamp: Utc::now(),
                    }));
                }
                ChatStreamEvent::Finished {
                    finish_reason: fr,
                    usage: u,
                } => {
                    finish_reason = fr.clone();
                    usage = *u;
                }
                ChatStreamEvent::Started { .. } => {}
            }
        }

        let message = accumulator.to_message();
        Ok((message, finish_reason, usage))
    }

    async fn call_complete(
        &self,
        provider: &Arc<dyn crate::provider::ChatProvider>,
        request: ChatRequest,
        run_id: RunId,
    ) -> RuntimeResult<(Message, FinishReason, Option<TokenUsage>)> {
        let response: ChatResponse =
            timeout(self.policy.provider_timeout, provider.complete(request))
                .await
                .map_err(|_| RuntimeError::ProviderNotFound("provider timeout".to_owned()))?
                .map_err(RuntimeError::from)?;

        if let Some(text) = extract_assistant_text(&response.message) {
            self.emit(&AgentEvent::TextDelta(TextDelta {
                run_id,
                delta: text,
                timestamp: Utc::now(),
            }));
        }

        for tc in extract_tool_calls(&response.message) {
            self.emit(&AgentEvent::ToolCallCompleted(ToolCallCompleted {
                run_id,
                call: tc,
                timestamp: Utc::now(),
            }));
        }

        Ok((response.message, response.finish_reason, response.usage))
    }

    fn emit(&self, event: &AgentEvent) {
        let _ = self.event_tx.send(event.clone());

        // Persist event to RunStore (fire-and-forget).
        {
            let event = event.clone();
            let store = Arc::clone(&self.store);
            tokio::spawn(async move {
                let record = RunEventRecord::new(0, event.run_id(), event);
                if let Err(e) = store.runs().append_event(record).await {
                    tracing::warn!(error = %e, "failed to persist agent event");
                }
            });
        }

        #[cfg(feature = "queue")]
        {
            if let Some(publisher) = &self.event_publisher {
                let publisher = Arc::clone(publisher);
                let event = event.clone();
                tokio::spawn(async move {
                    if let Err(e) = publisher.publish(event).await {
                        tracing::warn!(error = %e, "failed to publish event externally");
                    }
                });
            }
        }
    }

    async fn update_status(&self, run_id: RunId, status: RunStatus) -> RuntimeResult<()> {
        self.store.runs().update_run_status(run_id, status).await
    }

    async fn fail_run(&self, run_id: RunId, err: &RuntimeError) {
        let error_msg = err.to_string();
        error!(%run_id, error = %error_msg, "run failed");
        let _ = self.update_status(run_id, RunStatus::Failed).await;
        self.emit(&AgentEvent::RunFailed(RunFailed {
            run_id,
            error: error_msg,
            timestamp: Utc::now(),
        }));
    }
}

/// Output of a completed agent run.
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// Run identifier.
    pub run_id: RunId,
    /// Session identifier.
    pub session_id: Uuid,
    /// Number of model call iterations.
    pub iterations: usize,
    /// Final finish reason.
    pub finish_reason: FinishReason,
    /// Aggregated token usage across all iterations.
    pub total_usage: TokenUsage,
}

fn extract_assistant_text(message: &Message) -> Option<String> {
    match message {
        Message::Assistant { content, .. } => {
            let text: String = content
                .iter()
                .filter_map(|p| match p {
                    crate::provider::ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            if text.is_empty() { None } else { Some(text) }
        }
        _ => None,
    }
}

fn extract_tool_calls(message: &Message) -> Vec<ToolCall> {
    match message {
        Message::Assistant { tool_calls, .. } => tool_calls.clone(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::provider::{
        ChatProvider, ChatResponse, ModelName, ProviderCapabilities, ProviderId, ProviderResult,
    };
    use crate::runtime::memory::MemoryRunStore;
    use crate::store::memory::{MemoryExecutionStore, MemorySessionStore};
    use crate::tool::{FunctionTool, ToolRegistry};
    use async_trait::async_trait;
    use serde_json::json;

    struct MockProvider {
        responses: std::sync::Mutex<Vec<ChatResponse>>,
    }

    impl MockProvider {
        fn new(responses: Vec<ChatResponse>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
            }
        }

        fn text_response(text: &str) -> ChatResponse {
            ChatResponse {
                provider: ProviderId::new("mock"),
                model: ModelName::new("test"),
                message: Message::assistant_text(text),
                finish_reason: FinishReason::Stop,
                usage: Some(TokenUsage::new(10, 20)),
                raw: None,
            }
        }

        fn tool_call_response(
            call_id: &str,
            tool_name: &str,
            args: serde_json::Value,
        ) -> ChatResponse {
            ChatResponse {
                provider: ProviderId::new("mock"),
                model: ModelName::new("test"),
                message: Message::Assistant {
                    content: vec![],
                    tool_calls: vec![ToolCall::new(call_id, tool_name, args)],
                },
                finish_reason: FinishReason::ToolCalls,
                usage: Some(TokenUsage::new(15, 25)),
                raw: None,
            }
        }
    }

    #[async_trait]
    impl ChatProvider for MockProvider {
        fn id(&self) -> ProviderId {
            ProviderId::new("mock")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::chat()
        }

        async fn complete(&self, _request: ChatRequest) -> ProviderResult<ChatResponse> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Ok(Self::text_response("no more responses"))
            } else {
                Ok(responses.remove(0))
            }
        }
    }

    fn make_runtime(provider: MockProvider, tools: ToolRegistry) -> AgentRuntime {
        let mut registry = crate::provider::ProviderRegistry::new();
        registry.register_chat(provider);

        let sessions = MemorySessionStore::new();
        let executions = MemoryExecutionStore::new();
        let runs = MemoryRunStore::new();
        let store = Arc::new(RuntimeStore::new(
            Box::new(sessions),
            Box::new(executions),
            Box::new(runs),
        ));

        let policy = RuntimePolicy::new().with_max_iterations(5);
        let tool_runtime = ToolRuntime::new(tools, policy.clone());
        let context = ContextPipeline::new();

        AgentRuntime::new(registry, context, tool_runtime, store, policy)
    }

    #[tokio::test]
    async fn run_should_complete_with_text_response() {
        let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
        let runtime = make_runtime(provider, ToolRegistry::new());

        let request = RunRequest::new(ProviderId::new("mock"), ModelName::new("test"), "Hi there");

        let output = runtime.run(request).await.unwrap();
        assert_eq!(output.iterations, 1);
        assert!(matches!(output.finish_reason, FinishReason::Stop));
        assert_eq!(output.total_usage.input_tokens, 10);
        assert_eq!(output.total_usage.output_tokens, 20);
    }

    #[tokio::test]
    async fn run_should_execute_tools_and_loop() {
        let provider = MockProvider::new(vec![
            MockProvider::tool_call_response("call_1", "echo", json!({"message": "hello"})),
            MockProvider::text_response("Done!"),
        ]);

        let mut tools = ToolRegistry::new();
        tools.register(FunctionTool::new(
            "echo",
            "Echoes input",
            json!({"type": "object", "properties": {"message": {"type": "string"}}}),
            |args: serde_json::Value| -> std::pin::Pin<
                Box<
                    dyn std::future::Future<Output = crate::tool::ToolResult<serde_json::Value>>
                        + Send,
                >,
            > {
                Box::pin(async move {
                    Ok(args
                        .get("message")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null))
                })
            },
        ));

        let runtime = make_runtime(provider, tools);

        let request = RunRequest::new(
            ProviderId::new("mock"),
            ModelName::new("test"),
            "Echo hello",
        );

        let output = runtime.run(request).await.unwrap();
        assert_eq!(output.iterations, 2);
        assert!(matches!(output.finish_reason, FinishReason::Stop));
    }

    #[tokio::test]
    async fn run_should_respect_iteration_limit() {
        let responses: Vec<ChatResponse> = (0..10)
            .map(|i| {
                MockProvider::tool_call_response(
                    &format!("call_{i}"),
                    "echo",
                    json!({"message": format!("msg_{i}")}),
                )
            })
            .collect();

        let provider = MockProvider::new(responses);

        let mut tools = ToolRegistry::new();
        tools.register(FunctionTool::new(
            "echo",
            "Echoes",
            json!({"type": "object"}),
            |_args: serde_json::Value| -> std::pin::Pin<
                Box<
                    dyn std::future::Future<Output = crate::tool::ToolResult<serde_json::Value>>
                        + Send,
                >,
            > { Box::pin(async move { Ok(json!("ok")) }) },
        ));

        let runtime = make_runtime(provider, tools);

        let request = RunRequest::new(ProviderId::new("mock"), ModelName::new("test"), "loop");

        let result = runtime.run(request).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RuntimeError::IterationLimitExceeded(_)
        ));
    }

    #[tokio::test]
    async fn run_should_emit_events() {
        let provider = MockProvider::new(vec![MockProvider::text_response("Hello!")]);
        let runtime = make_runtime(provider, ToolRegistry::new());
        let mut rx = runtime.subscribe();

        let request = RunRequest::new(ProviderId::new("mock"), ModelName::new("test"), "Hi");

        let _output = runtime.run(request).await.unwrap();

        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }

        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::RunStarted(_)))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ContextBuilt(_)))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ModelStarted(_)))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::RunCompleted(_)))
        );
    }

    #[tokio::test]
    async fn run_should_create_session_when_none_provided() {
        let provider = MockProvider::new(vec![MockProvider::text_response("Hi")]);
        let runtime = make_runtime(provider, ToolRegistry::new());

        let request = RunRequest::new(ProviderId::new("mock"), ModelName::new("test"), "Hello");

        let output = runtime.run(request).await.unwrap();
        assert_ne!(output.session_id, Uuid::nil());
    }

    #[tokio::test]
    async fn run_should_fail_for_unknown_provider() {
        let provider = MockProvider::new(vec![]);
        let runtime = make_runtime(provider, ToolRegistry::new());

        let request = RunRequest::new(
            ProviderId::new("nonexistent"),
            ModelName::new("test"),
            "Hello",
        );

        let result = runtime.run(request).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RuntimeError::ProviderNotFound(_)
        ));
    }
}
