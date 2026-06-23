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

use crate::provider::{ChatRequest, ChatStreamEvent, FinishReason, Message, TokenUsage, ToolCall};

use super::accumulator::StreamAccumulator;
use super::compaction::CompactionService;
use super::context::ContextPipeline;
use super::doom_loop::{DoomLoopDetector, DoomLoopType};
use super::error::{RuntimeError, RuntimeResult};
use super::event::{
    AgentEvent, ContextBuilt, DoomLoopDetected, MessageCommitted, ModelStarted, RunCompleted,
    RunFailed, RunStarted, TextDelta, ToolCallCompleted, ToolCallDelta,
    ToolCallStarted as ToolCallStartedEvent, UsageRecorded,
};
use super::job::{BackgroundJobPool, JobConditions, JobPriority, JobType};
use super::policy::RuntimePolicy;
use super::run::{RunId, RunRecord, RunRequest, RunStatus};
use super::session_gate::SessionGate;
use super::snapshot::{Snapshot, SnapshotStore};
use super::store::RuntimeStore;
use super::tool::ToolRuntime;
use super::turn::{TurnAction, TurnOutcome, TurnState, TurnTransition};
use crate::tool_scope::ScopeGuard;

/// Streaming-first agent runtime kernel.
///
/// Ties together provider registry, context pipeline, tool runtime,
/// compaction service, persistent stores, and background job pool into a
/// complete agent execution loop.
pub struct AgentRuntime {
    providers: crate::provider::ProviderRegistry,
    context: ContextPipeline,
    tools: ToolRuntime,
    store: Arc<RuntimeStore>,
    policy: RuntimePolicy,
    compaction: CompactionService,
    session_gate: SessionGate,
    event_tx: broadcast::Sender<AgentEvent>,
    #[cfg(feature = "queue")]
    event_publisher: Option<Arc<dyn EventPublisher>>,
    background_jobs: Arc<BackgroundJobPool>,
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
        let background_jobs = BackgroundJobPool::new(
            store.clone(),
            #[cfg(feature = "queue")]
            None,
            None,
        );
        background_jobs.start();
        Self {
            providers,
            context,
            tools,
            store,
            policy,
            compaction,
            session_gate: SessionGate::new(),
            event_tx,
            #[cfg(feature = "queue")]
            event_publisher: None,
            background_jobs,
            snapshot_store: None,
        }
    }

    /// Sets an external event publisher for the agent runtime.
    ///
    /// When set, every [`AgentEvent`] emitted during a run will also be
    /// published to the configured [`EventPublisher`] via fire-and-forget.
    #[cfg(feature = "queue")]
    #[must_use]
    pub fn with_event_publisher(mut self, publisher: Arc<dyn EventPublisher>) -> Self {
        self.background_jobs
            .set_event_publisher(Arc::clone(&publisher));
        self.event_publisher = Some(publisher);
        self
    }

    /// Sets an optional snapshot store for FSM run recovery.
    #[must_use]
    pub fn with_snapshot_store(mut self, snapshot_store: Arc<dyn SnapshotStore>) -> Self {
        self.snapshot_store = Some(snapshot_store);
        self
    }

    /// Returns a reference to the background job pool.
    #[must_use]
    pub fn background_jobs(&self) -> &Arc<BackgroundJobPool> {
        &self.background_jobs
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
        let run_id = RunId::new();
        let session_id = self.store.ensure_session(request.session_id).await?;

        // Acquire per-session lock — prevents concurrent runs from
        // interleaving writes to the same session.
        let _session_guard = self
            .session_gate
            .acquire(session_id)
            .await
            .map_err(|busy| RuntimeError::SessionBusy(busy.session_id))?;

        // Push a Run-level tool scope. The RAII guard ensures cleanup
        // on every exit path, including early returns and panics.
        let _run_scope: ScopeGuard = self.tools.registry().push_scope_guarded();

        let run_record = RunRecord::new(
            run_id,
            session_id,
            request.provider.clone(),
            request.model.clone(),
            request.metadata.clone(),
        );
        self.store.runs().create_run(run_record).await?;

        // Create doom loop detector for this run.
        let mut doom_detector = DoomLoopDetector::new(self.policy.doom_loop.clone());

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
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn save_snapshot_helper(
        &self,
        run_id: RunId,
        session_id: Uuid,
        iteration: usize,
        state: TurnState,
        total_usage: TokenUsage,
        last_finish: &Option<FinishReason>,
        assistant_message: &Option<Message>,
        assistant_msg_id: Option<Uuid>,
        request: &RunRequest,
    ) -> RuntimeResult<()> {
        if let Some(store) = &self.snapshot_store {
            let snapshot = Snapshot {
                run_id,
                session_id,
                status: TurnTransition::status_for(state),
                iteration,
                current_state: state,
                total_usage,
                last_finish: last_finish.clone(),
                assistant_message: assistant_message.clone(),
                assistant_msg_id,
                request: request.clone(),
                timestamp: Utc::now(),
            };
            store.save(&snapshot).await?;
        }
        Ok(())
    }

    async fn delete_snapshot_helper(&self, run_id: RunId) -> RuntimeResult<()> {
        if let Some(store) = &self.snapshot_store {
            store.delete(run_id).await?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_loop(
        &self,
        run_id: RunId,
        session_id: Uuid,
        provider: Arc<dyn crate::provider::ChatProvider>,
        request: RunRequest,
        tool_specs: Vec<crate::provider::ToolSpec>,
        has_tools: bool,
        iteration: usize,
        total_usage: TokenUsage,
        last_finish: Option<FinishReason>,
        assistant_message: Option<Message>,
        assistant_msg_id: Option<Uuid>,
        start_state: TurnState,
        doom_detector: &mut DoomLoopDetector,
    ) -> RuntimeResult<RunOutput> {
        let result = self
            .run_loop_inner(
                run_id,
                session_id,
                provider,
                request,
                tool_specs,
                has_tools,
                iteration,
                total_usage,
                last_finish,
                assistant_message,
                assistant_msg_id,
                start_state,
                doom_detector,
            )
            .await;

        // Clean up snapshot on exit since the run has finished (either Completed or Failed).
        let _ = self.delete_snapshot_helper(run_id).await;

        result
    }

    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    async fn run_loop_inner(
        &self,
        run_id: RunId,
        session_id: Uuid,
        provider: Arc<dyn crate::provider::ChatProvider>,
        request: RunRequest,
        tool_specs: Vec<crate::provider::ToolSpec>,
        has_tools: bool,
        mut iteration: usize,
        mut total_usage: TokenUsage,
        mut last_finish: Option<FinishReason>,
        mut assistant_message: Option<Message>,
        mut assistant_msg_id: Option<Uuid>,
        start_state: TurnState,
        doom_detector: &mut DoomLoopDetector,
    ) -> RuntimeResult<RunOutput> {
        let mut resume_from = start_state;

        loop {
            // Increment iteration ONLY if we are starting a normal iteration (from CheckingPolicy)
            if resume_from == TurnState::CheckingPolicy {
                iteration += 1;
            }

            // ── TurnState::CheckingPolicy ──────────────────────────
            if resume_from == TurnState::CheckingPolicy {
                self.save_snapshot_helper(
                    run_id,
                    session_id,
                    iteration,
                    TurnState::CheckingPolicy,
                    total_usage,
                    &last_finish,
                    &assistant_message,
                    assistant_msg_id,
                    &request,
                )
                .await?;

                let outcome = if iteration > self.policy.max_iterations {
                    TurnOutcome::PolicyExceeded {
                        reason: format!(
                            "iteration {iteration} exceeds limit {}",
                            self.policy.max_iterations
                        ),
                    }
                } else if let Some(budget) = self.policy.max_tokens {
                    let budget_u64 = budget as u64;
                    if total_usage.total_tokens >= budget_u64 {
                        #[allow(clippy::cast_possible_truncation)]
                        TurnOutcome::PolicyExceeded {
                            reason: format!(
                                "token budget {budget} exceeded: {} used",
                                total_usage.total_tokens
                            ),
                        }
                    } else {
                        TurnOutcome::Success
                    }
                } else {
                    TurnOutcome::Success
                };

                match TurnTransition::resolve(TurnState::CheckingPolicy, &outcome) {
                    TurnAction::Fail { reason: _ } => {
                        if iteration > self.policy.max_iterations {
                            let err =
                                RuntimeError::IterationLimitExceeded(self.policy.max_iterations);
                            self.fail_run(run_id, &err).await;
                            return Err(err);
                        }
                        #[allow(clippy::cast_possible_truncation)]
                        let err = RuntimeError::TokenBudgetExceeded {
                            used: total_usage.total_tokens as usize,
                            limit: self.policy.max_tokens.unwrap_or(0),
                        };
                        self.fail_run(run_id, &err).await;
                        return Err(err);
                    }
                    TurnAction::Continue { .. } => {}
                    _ => unreachable!("CheckingPolicy only produces Fail or Continue"),
                }
            }

            // ── TurnState::BuildingContext ──────────────────────────
            let mut chat_request = None;
            if resume_from == TurnState::CheckingPolicy
                || resume_from == TurnState::BuildingContext
                || resume_from == TurnState::CallingModel
            {
                self.save_snapshot_helper(
                    run_id,
                    session_id,
                    iteration,
                    TurnState::BuildingContext,
                    total_usage,
                    &last_finish,
                    &assistant_message,
                    assistant_msg_id,
                    &request,
                )
                .await?;

                self.update_status(
                    run_id,
                    TurnTransition::status_for(TurnState::BuildingContext),
                )
                .await?;

                // Proactive compaction
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

                let req = self
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
                    message_count: req.messages.len(),
                    timestamp: Utc::now(),
                }));

                chat_request = Some(req);
            }

            // ── TurnState::CallingModel ─────────────────────────────
            let (next_assistant_message, next_finish_reason, _usage) = if resume_from
                == TurnState::CheckingPolicy
                || resume_from == TurnState::CallingModel
            {
                resume_from = TurnState::CheckingPolicy;

                self.save_snapshot_helper(
                    run_id,
                    session_id,
                    iteration,
                    TurnState::CallingModel,
                    total_usage,
                    &last_finish,
                    &assistant_message,
                    assistant_msg_id,
                    &request,
                )
                .await?;

                self.update_status(run_id, TurnTransition::status_for(TurnState::CallingModel))
                    .await?;

                self.emit(&AgentEvent::ModelStarted(ModelStarted {
                    run_id,
                    provider: request.provider.clone(),
                    model: request.model.clone(),
                    iteration,
                    timestamp: Utc::now(),
                }));

                let req = chat_request.take().ok_or_else(|| {
                    RuntimeError::RecoveryFailed("chat request missing in CallingModel".to_string())
                })?;

                let (msg, finish, usg) = {
                    let model_result = self.call_model(&provider, req, run_id).await;
                    let outcome = match &model_result {
                        Ok(_) => TurnOutcome::Success,
                        Err(RuntimeError::Provider(e)) if e.is_context_overflow() => {
                            TurnOutcome::ContextOverflow
                        }
                        Err(e) => TurnOutcome::ProviderError {
                            message: e.to_string(),
                        },
                    };

                    match TurnTransition::resolve(TurnState::CallingModel, &outcome) {
                        TurnAction::Continue { .. } => model_result?,
                        TurnAction::CompactAndRetry => {
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
                        TurnAction::Fail { .. } => match model_result {
                            Err(e) => return Err(e),
                            Ok(_) => unreachable!("Fail action but model call succeeded"),
                        },
                        TurnAction::BreakLoop => {
                            unreachable!("CallingModel never produces BreakLoop")
                        }
                    }
                };

                if let Some(u) = &usg {
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

                last_finish = Some(finish.clone());

                let msg_id = self.store.append_message(session_id, &msg).await?;

                self.emit(&AgentEvent::AssistantMessageCommitted(MessageCommitted {
                    run_id,
                    message_id: msg_id,
                    timestamp: Utc::now(),
                }));

                assistant_message = Some(msg);
                assistant_msg_id = Some(msg_id);

                (assistant_message.clone(), last_finish.clone(), usg)
            } else {
                (assistant_message.clone(), last_finish.clone(), None)
            };

            // ── TurnState::ProcessingResponse ───────────────────────
            let mut tool_calls = Vec::new();
            if resume_from == TurnState::CheckingPolicy
                || resume_from == TurnState::ProcessingResponse
            {
                resume_from = TurnState::CheckingPolicy;

                let msg = next_assistant_message.as_ref().ok_or_else(|| {
                    RuntimeError::RecoveryFailed(
                        "assistant message missing in ProcessingResponse".to_string(),
                    )
                })?;
                let finish = next_finish_reason.as_ref().ok_or_else(|| {
                    RuntimeError::RecoveryFailed(
                        "last finish missing in ProcessingResponse".to_string(),
                    )
                })?;

                let calls = match msg {
                    Message::Assistant { tool_calls, .. } if !tool_calls.is_empty() => {
                        tool_calls.clone()
                    }
                    _ => Vec::new(),
                };

                let response_outcome = if calls.is_empty() {
                    TurnOutcome::NoToolCalls
                } else if !matches!(finish, FinishReason::ToolCalls) {
                    TurnOutcome::NotToolCalls {
                        finish_reason: finish.clone(),
                    }
                } else {
                    TurnOutcome::Success
                };

                match TurnTransition::resolve(TurnState::ProcessingResponse, &response_outcome) {
                    TurnAction::BreakLoop => break,
                    TurnAction::Continue { .. } => {
                        tool_calls = calls;
                    }
                    _ => unreachable!("ProcessingResponse only produces BreakLoop or Continue"),
                }
            } else if let Some(Message::Assistant {
                tool_calls: calls, ..
            }) = &assistant_message
            {
                tool_calls = calls.clone();
            }

            // ── TurnState::ExecutingTools ───────────────────────────
            if resume_from == TurnState::CheckingPolicy || resume_from == TurnState::ExecutingTools
            {
                resume_from = TurnState::CheckingPolicy;

                self.save_snapshot_helper(
                    run_id,
                    session_id,
                    iteration,
                    TurnState::ExecutingTools,
                    total_usage,
                    &last_finish,
                    &assistant_message,
                    assistant_msg_id,
                    &request,
                )
                .await?;

                self.update_status(
                    run_id,
                    TurnTransition::status_for(TurnState::ExecutingTools),
                )
                .await?;

                let msg_id = assistant_msg_id.ok_or_else(|| {
                    RuntimeError::RecoveryFailed(
                        "assistant message ID missing in ExecutingTools".to_string(),
                    )
                })?;

                let outcomes = self
                    .tools
                    .execute_batch(
                        tool_calls,
                        session_id,
                        msg_id,
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
            }

            // ── TurnState::Persisting ───────────────────────────────
            if resume_from == TurnState::CheckingPolicy || resume_from == TurnState::Persisting {
                resume_from = TurnState::CheckingPolicy;

                self.save_snapshot_helper(
                    run_id,
                    session_id,
                    iteration,
                    TurnState::Persisting,
                    total_usage,
                    &last_finish,
                    &assistant_message,
                    assistant_msg_id,
                    &request,
                )
                .await?;

                self.update_status(run_id, TurnTransition::status_for(TurnState::Persisting))
                    .await?;

                let finish = last_finish.as_ref().ok_or_else(|| {
                    RuntimeError::RecoveryFailed("last finish missing in Persisting".to_string())
                })?;

                let persisting_outcome = if matches!(finish, FinishReason::ToolCalls) {
                    TurnOutcome::Success
                } else {
                    TurnOutcome::NotToolCalls {
                        finish_reason: finish.clone(),
                    }
                };

                match TurnTransition::resolve(TurnState::Persisting, &persisting_outcome) {
                    TurnAction::BreakLoop => break,
                    TurnAction::Continue { .. } => {}
                    _ => unreachable!(),
                }
            }

            // Clear assistant message info for the next iteration
            assistant_message = None;
            assistant_msg_id = None;
        }

        self.update_status(run_id, RunStatus::Completed).await?;

        let final_finish = last_finish.clone().unwrap_or(FinishReason::Stop);
        self.emit(&AgentEvent::RunCompleted(RunCompleted {
            run_id,
            finish_reason: final_finish.clone(),
            iterations: iteration,
            timestamp: Utc::now(),
        }));

        info!(%run_id, iterations = iteration, "run completed");

        Ok(RunOutput {
            run_id,
            session_id,
            iterations: iteration,
            finish_reason: final_finish,
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
        let response = timeout(self.policy.provider_timeout, provider.complete(request))
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
                call: tc.clone(),
                timestamp: Utc::now(),
            }));
        }

        Ok((response.message, response.finish_reason, response.usage))
    }

    fn emit(&self, event: &AgentEvent) {
        let _ = self.event_tx.send(event.clone());

        let jobs = Arc::clone(&self.background_jobs);
        let event = event.clone();
        tokio::spawn(async move {
            jobs.schedule(
                JobPriority::Normal,
                JobType::PersistEvent {
                    run_id: event.run_id(),
                    event: event.clone(),
                },
                JobConditions::default(),
            )
            .await;
        });

        #[cfg(feature = "queue")]
        if self.event_publisher.is_some() {
            let jobs = Arc::clone(&self.background_jobs);
            let event = event.clone();
            tokio::spawn(async move {
                jobs.schedule(
                    JobPriority::High,
                    JobType::PublishExternalEvent {
                        event: event.clone(),
                    },
                    JobConditions::default(),
                )
                .await;
            });
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
            timestamp: Utc::now(),
        };

        snapshot_store.save(&snapshot).await.unwrap();

        let output = runtime.resume(run_id).await.unwrap();

        assert_eq!(output.run_id, run_id);
        assert_eq!(output.session_id, session_id);
        assert!(matches!(output.finish_reason, FinishReason::Stop));
    }
}
