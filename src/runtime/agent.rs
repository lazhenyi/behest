//! Agent runtime — streaming-first execution kernel.
//!
//! [`AgentRuntime`] orchestrates the full agent loop: context building,
//! model invocation (streaming with non-streaming fallback), tool execution,
//! session persistence, and event emission.

use std::sync::Arc;

#[cfg(feature = "queue")]
use crate::queue::EventPublisher;

use chrono::Utc;
use tokio::sync::broadcast;
use tracing::{debug, error, warn};
use uuid::Uuid;

use crate::provider::{FinishReason, Message, TokenUsage};

use super::compaction::{CompactionCircuitBreaker, CompactionService};
use super::context::ContextPipeline;
use super::doom_loop::DoomLoopDetector;
use super::error::{RuntimeError, RuntimeResult};
use super::event::{AgentEvent, RunStarted};
use super::input::{InputAdmission, InputRecord};
use super::job::BackgroundJobPool;
use super::policy::RuntimePolicy;
use super::run::{RunId, RunRecord, RunRequest, RunStatus};
use super::session_gate::SessionGate;
use super::snapshot::{Snapshot, SnapshotStore};
use super::store::RuntimeStore;
use super::tool::ToolRuntime;
use super::turn::{TurnState, TurnTransition};
use crate::tool_scope::ScopeGuard;

/// Streaming-first agent runtime kernel.
///
/// Ties together provider registry, context pipeline, tool runtime,
/// compaction service, persistent stores, and background job pool into a
/// complete agent execution loop.
pub struct AgentRuntime {
    providers: crate::provider::ProviderRegistry,
    pub(super) context: ContextPipeline,
    pub(super) tools: ToolRuntime,
    pub(super) store: Arc<RuntimeStore>,
    pub(super) policy: RuntimePolicy,
    pub(super) compaction: CompactionService,
    session_gate: SessionGate,
    input_admission: InputAdmission,
    pub(super) event_tx: broadcast::Sender<AgentEvent>,
    #[cfg(feature = "queue")]
    pub(super) event_publisher: Option<Arc<dyn EventPublisher>>,
    pub(super) background_jobs: Option<Arc<BackgroundJobPool>>,
    snapshot_store: Option<Arc<dyn SnapshotStore>>,
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
        let input_admission = InputAdmission::new(policy.input_admission.clone());
        Self {
            providers,
            context,
            tools,
            store,
            policy,
            compaction,
            session_gate: SessionGate::new(),
            input_admission,
            event_tx,
            #[cfg(feature = "queue")]
            event_publisher: None,
            background_jobs: None,
            snapshot_store: None,
        }
    }

    /// Injects a background job pool for event persistence and publishing.
    ///
    /// Callers are responsible for calling [`BackgroundJobPool::start`]
    /// on the pool before passing it in.
    #[must_use]
    pub fn with_background_jobs(mut self, pool: Arc<BackgroundJobPool>) -> Self {
        self.background_jobs = Some(pool);
        self
    }

    /// Sets an external event publisher for the agent runtime.
    ///
    /// When set, every [`AgentEvent`] emitted during a run will also be
    /// published to the configured [`EventPublisher`] via fire-and-forget.
    #[cfg(feature = "queue")]
    #[must_use]
    pub fn with_event_publisher(mut self, publisher: Arc<dyn EventPublisher>) -> Self {
        if let Some(ref jobs) = self.background_jobs {
            jobs.set_event_publisher(Arc::clone(&publisher));
        }
        self.event_publisher = Some(publisher);
        self
    }

    /// Sets an optional snapshot store for FSM run recovery.
    #[must_use]
    pub fn with_snapshot_store(mut self, snapshot_store: Arc<dyn SnapshotStore>) -> Self {
        self.snapshot_store = Some(snapshot_store);
        self
    }

    /// Returns a reference to the background job pool, if configured.
    #[must_use]
    pub fn background_jobs(&self) -> Option<&Arc<BackgroundJobPool>> {
        self.background_jobs.as_ref()
    }

    /// Returns the session gate.
    #[must_use]
    pub fn session_gate(&self) -> &SessionGate {
        &self.session_gate
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
        let run_id = request.run_id.unwrap_or_default();
        let session_id = self.store.ensure_session(request.session_id).await?;

        // Acquire per-session lock — prevents concurrent runs from
        // interleaving writes to the same session.
        let _session_guard = self
            .session_gate
            .acquire(session_id)
            .await
            .map_err(|busy| RuntimeError::SessionBusy(busy.session_id))?;

        // Admit the input before allocating any run resources.
        let mut input_record = InputRecord::new(session_id, request.input.clone());
        let admission_events = self
            .input_admission
            .admit(&mut input_record)
            .map_err(|e| RuntimeError::InputAdmissionFailed(e.to_string()))?;
        if input_record.state == super::input::InputState::Rejected {
            let reason = input_record.rejection_reason.clone().unwrap_or_default();
            return Err(RuntimeError::InputRejected {
                input_id: input_record.id,
                reason,
            });
        }
        debug!(
            input_id = %input_record.id,
            events = admission_events.len(),
            "input admitted"
        );

        // Push a Run-level tool scope. The RAII guard ensures cleanup
        // on every exit path, including early returns and panics.
        let _run_scope: ScopeGuard = self.tools.registry().push_scope_guarded();

        let run_record = RunRecord::new(
            run_id,
            session_id,
            request.provider.clone(),
            request.model.clone(),
            request.metadata.clone(),
            request.client_request_id.clone(),
        );
        self.store.runs().create_run(run_record).await?;

        // Create doom loop detector for this run.
        let mut doom_detector = DoomLoopDetector::new(self.policy.doom_loop.clone());

        // Create compaction circuit breaker for this run.
        let mut compaction_breaker =
            CompactionCircuitBreaker::new(self.policy.compaction.circuit_breaker_threshold);

        self.emit(&AgentEvent::RunStarted(RunStarted {
            run_id,
            session_id,
            provider: request.provider.clone(),
            model: request.model.clone(),
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

        self.run_loop(
            run_id,
            session_id,
            provider,
            request,
            tool_specs,
            has_tools,
            0,
            TokenUsage::new(0, 0),
            None,
            None,
            None,
            TurnState::CheckingPolicy,
            &mut doom_detector,
            &mut compaction_breaker,
            0,
        )
        .await
    }

    /// Resumes a crashed or halted agent run from its last saved snapshot.
    ///
    /// Loads the snapshot by `run_id`, re-acquires the session lock, pushes
    /// a Run-level tool scope, and restarts the turn loop from the saved state.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` if the snapshot is not found, the session is busy,
    /// or resuming fails.
    pub async fn resume(&self, run_id: RunId) -> RuntimeResult<RunOutput> {
        let snapshot_store = self.snapshot_store.as_ref().ok_or_else(|| {
            RuntimeError::RecoveryFailed("snapshot store not configured".to_string())
        })?;

        let snapshot = snapshot_store
            .load(run_id)
            .await?
            .ok_or_else(|| RuntimeError::RunNotFound(run_id))?;

        // Re-acquire per-session lock
        let _session_guard = self
            .session_gate
            .acquire(snapshot.session_id)
            .await
            .map_err(|busy| RuntimeError::SessionBusy(busy.session_id))?;

        // Re-push a Run-level tool scope
        let _run_scope: ScopeGuard = self.tools.registry().push_scope_guarded();

        let provider = self
            .providers
            .chat(&snapshot.request.provider)
            .ok_or_else(|| RuntimeError::ProviderNotFound(snapshot.request.provider.to_string()))?;

        let tool_specs = self.tools.registry().specs();
        let has_tools = !tool_specs.is_empty();

        // Resume the run in the database/store status as well
        self.update_status(run_id, TurnTransition::status_for(snapshot.current_state))
            .await?;

        let mut doom_detector = DoomLoopDetector::new(self.policy.doom_loop.clone());

        let mut compaction_breaker =
            CompactionCircuitBreaker::new(self.policy.compaction.circuit_breaker_threshold);

        self.run_loop(
            run_id,
            snapshot.session_id,
            provider,
            snapshot.request,
            tool_specs,
            has_tools,
            snapshot.iteration,
            snapshot.total_usage,
            snapshot.last_finish,
            snapshot.assistant_message,
            snapshot.assistant_msg_id,
            snapshot.current_state,
            &mut doom_detector,
            &mut compaction_breaker,
            snapshot.output_recovery_count,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn save_snapshot_helper(
        &self,
        run_id: RunId,
        session_id: Uuid,
        iteration: usize,
        state: TurnState,
        total_usage: TokenUsage,
        last_finish: Option<&FinishReason>,
        assistant_message: Option<&Message>,
        assistant_msg_id: Option<Uuid>,
        request: &RunRequest,
        output_recovery_count: u32,
    ) -> RuntimeResult<()> {
        if let Some(store) = &self.snapshot_store {
            let snapshot = Snapshot {
                run_id,
                session_id,
                status: TurnTransition::status_for(state),
                iteration,
                current_state: state,
                total_usage,
                last_finish: last_finish.cloned(),
                assistant_message: assistant_message.cloned(),
                assistant_msg_id,
                request: request.clone(),
                output_recovery_count,
                timestamp: Utc::now(),
            };
            store.save(&snapshot).await?;
        }
        Ok(())
    }

    pub(super) async fn delete_snapshot_helper(&self, run_id: RunId) -> RuntimeResult<()> {
        if let Some(store) = &self.snapshot_store {
            store.delete(run_id).await?;
        }
        Ok(())
    }

    pub(super) fn emit(&self, event: &AgentEvent) {
        if let Err(e) = self.event_tx.send(event.clone()) {
            warn!(lag = ?e, "event channel full, consumer too slow — event dropped");
        }

        if let Some(ref jobs) = self.background_jobs {
            let jobs = Arc::clone(jobs);
            let event = event.clone();
            tokio::spawn(async move {
                jobs.schedule(
                    super::job::JobPriority::Normal,
                    super::job::JobType::PersistEvent {
                        run_id: event.run_id(),
                        event: event.clone(),
                    },
                    super::job::JobConditions::default(),
                )
                .await;
            });
        }

        #[cfg(feature = "queue")]
        if self.event_publisher.is_some() {
            if let Some(ref jobs) = self.background_jobs {
                let jobs = Arc::clone(jobs);
                let event = event.clone();
                tokio::spawn(async move {
                    jobs.schedule(
                        super::job::JobPriority::High,
                        super::job::JobType::PublishExternalEvent {
                            event: event.clone(),
                        },
                        super::job::JobConditions::default(),
                    )
                    .await;
                });
            }
        }
    }

    pub(super) async fn update_status(
        &self,
        run_id: RunId,
        status: RunStatus,
    ) -> RuntimeResult<()> {
        self.store.runs().update_run_status(run_id, status).await
    }

    pub(super) async fn fail_run(&self, run_id: RunId, err: &RuntimeError) {
        let error_msg = err.to_string();
        error!(%run_id, error = %error_msg, "run failed");
        let _ = self.update_status(run_id, RunStatus::Failed).await;
        self.emit(&AgentEvent::RunFailed(super::event::RunFailed {
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::provider::{
        ChatProvider, ChatRequest, ChatResponse, ModelName, ProviderCapabilities, ProviderId,
        ProviderResult, ToolCall,
    };
    use crate::runtime::memory::MemoryRunStore;
    use crate::runtime::snapshot::{FileSnapshotStore, Snapshot};
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

        fn length_response(text: &str) -> ChatResponse {
            ChatResponse {
                provider: ProviderId::new("mock"),
                model: ModelName::new("test"),
                message: Message::assistant_text(text),
                finish_reason: FinishReason::Length,
                usage: Some(TokenUsage::new(10, 20)),
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

    #[tokio::test]
    async fn run_should_create_snapshots_and_resume_successfully() {
        let temp_dir = tempfile::tempdir().unwrap();
        let snapshot_store = Arc::new(FileSnapshotStore::new(temp_dir.path().to_path_buf()));

        let provider = MockProvider::new(vec![
            MockProvider::tool_call_response("call_rec", "echo", json!({"message": "rec"})),
            MockProvider::text_response("Done after resume!"),
        ]);

        let mut tools = ToolRegistry::new();
        tools.register(FunctionTool::new(
            "echo",
            "Echoes message",
            json!({"type": "object"}),
            |args: serde_json::Value| -> std::pin::Pin<
                Box<
                    dyn std::future::Future<Output = crate::tool::ToolResult<serde_json::Value>>
                        + Send,
                >,
            > {
                Box::pin(async move { Ok(args.get("message").cloned().unwrap_or_default()) })
            },
        ));

        let runtime = make_runtime(provider, tools).with_snapshot_store(snapshot_store.clone());

        let request = RunRequest::new(
            ProviderId::new("mock"),
            ModelName::new("test"),
            "test snapshot and resume",
        );

        let run_id = RunId::new();
        let session_id = runtime.store().ensure_session(None).await.unwrap();

        // Real runs would already have a run record in the store before crashing/suspending.
        let run_record = RunRecord::new(
            run_id,
            session_id,
            ProviderId::new("mock"),
            ModelName::new("test"),
            serde_json::Value::Null,
            None,
        );
        runtime.store().runs().create_run(run_record).await.unwrap();

        let snapshot = Snapshot {
            run_id,
            session_id,
            status: RunStatus::CallingModel,
            iteration: 1,
            current_state: TurnState::CallingModel,
            total_usage: TokenUsage::new(5, 5),
            last_finish: Some(FinishReason::ToolCalls),
            assistant_message: Some(Message::Assistant {
                content: vec![],
                tool_calls: vec![ToolCall::new("call_rec", "echo", json!({"message": "rec"}))],
            }),
            assistant_msg_id: Some(Uuid::new_v4()),
            request: request.clone(),
            output_recovery_count: 0,
            timestamp: Utc::now(),
        };

        snapshot_store.save(&snapshot).await.unwrap();

        let output = runtime.resume(run_id).await.unwrap();

        assert_eq!(output.run_id, run_id);
        assert_eq!(output.session_id, session_id);
        assert!(matches!(output.finish_reason, FinishReason::Stop));
    }

    #[tokio::test]
    async fn run_should_recover_from_length_finish() {
        let provider = MockProvider::new(vec![
            MockProvider::length_response("First half..."),
            MockProvider::length_response("Second half..."),
            MockProvider::text_response("Complete response."),
        ]);
        let mut policy = RuntimePolicy::new();
        policy.max_output_recovery_attempts = 2;
        let runtime = make_runtime_with_policy(provider, ToolRegistry::new(), policy);

        let request = RunRequest::new(
            ProviderId::new("mock"),
            ModelName::new("test"),
            "Long story",
        );
        let output = runtime.run(request).await.unwrap();

        assert_eq!(output.iterations, 3);
        assert!(matches!(output.finish_reason, FinishReason::Stop));
    }

    #[tokio::test]
    async fn run_should_stop_recovery_after_max_attempts() {
        let provider = MockProvider::new(vec![
            MockProvider::length_response("Try 1..."),
            MockProvider::length_response("Try 2..."),
            MockProvider::length_response("Still truncated..."),
        ]);
        let mut policy = RuntimePolicy::new();
        policy.max_output_recovery_attempts = 2;
        let runtime = make_runtime_with_policy(provider, ToolRegistry::new(), policy);

        let request = RunRequest::new(
            ProviderId::new("mock"),
            ModelName::new("test"),
            "Even longer story",
        );
        let output = runtime.run(request).await.unwrap();

        assert_eq!(output.iterations, 3);
        assert!(matches!(output.finish_reason, FinishReason::Length));
    }

    fn make_runtime_with_policy(
        provider: MockProvider,
        tools: ToolRegistry,
        policy: RuntimePolicy,
    ) -> AgentRuntime {
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
        let tool_runtime = ToolRuntime::new(tools, policy.clone());
        let context = ContextPipeline::new();
        AgentRuntime::new(registry, context, tool_runtime, store, policy)
    }
}
