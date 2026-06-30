//! Agent run loop — the core iterative execution engine.
//!
//! Orchestrates the turn-by-turn lifecycle of an agent invocation:
//! policy checking, context building, model calling (streaming or
//! complete), response processing, tool execution, and persisting.
//! Extracted from [`super::agent::AgentRuntime`] to keep that file
//! under 1000 lines.
//!
//! # States
//!
//! The loop is driven by [`TurnState`] transitions via [`TurnTransition`]:
//!
//! ```text
//! CheckingPolicy → BuildingContext → CallingModel
//!                                               ↓
//! ProcessingResponse ←────────────────── ←── ←──┘
//!        ↓ (tool calls)
//! ExecutingTools → Persisting → CheckingPolicy (next iteration)
//! ```
//!
//! # Recovery
//!
//! On provider context overflow (`FinishReason::Length`) the loop falls
//! back to a streaming-only model call, and on output truncation it
//! prompts the model to continue where it left off. Snapshot-based
//! recovery allows resuming from any intermediate state.

use std::sync::Arc;

use chrono::Utc;
use futures_util::StreamExt;
use tokio::time::timeout;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::provider::{ChatRequest, ChatStreamEvent, FinishReason, Message, TokenUsage, ToolCall};

use super::AgentRuntime;
use super::accumulator::StreamAccumulator;
use super::compaction::CompactionCircuitBreaker;
use super::doom_loop::{DoomLoopDetector, DoomLoopType};
use super::error::{RuntimeError, RuntimeResult};
use super::event::{
    AgentEvent, CompactionCircuitOpened, ContextBuilt, DoomLoopDetected, MessageCommitted,
    ModelStarted, RunCompleted, TextDelta, ToolCallCompleted, ToolCallDelta,
    ToolCallStarted as ToolCallStartedEvent, UsageRecorded,
};
use super::run::{RunId, RunRequest, RunStatus};
use super::turn::{TurnAction, TurnOutcome, TurnState, TurnTransition};

impl AgentRuntime {
    /// Runs the agent turn loop — the main iterative execution driver.
    ///
    /// Drives the state machine through policy checking, context building,
    /// model calling, response processing, tool execution, and persisting.
    /// Wraps [`Self::run_loop_inner`] with snapshot cleanup on return.
    ///
    /// # Arguments
    /// * `run_id` — Identifies the current run.
    /// * `session_id` — Identifies the conversation session.
    /// * `provider` — The chat provider for model calls.
    /// * `request` — The original run request parameters.
    /// * `tool_specs` — Tool definitions available to the model.
    /// * `has_tools` — Whether tools are configured for this run.
    /// * `iteration` — Current iteration counter (1-based).
    /// * `total_usage` — Accumulated token usage across all iterations.
    /// * `last_finish` — Finish reason from the previous model call, if any.
    /// * `assistant_message` — Most recent assistant message, for resumption.
    /// * `assistant_msg_id` — Store ID of the most recent assistant message.
    /// * `start_state` — Turn state to resume from (for snapshot recovery).
    /// * `doom_detector` — Shared doom-loop detector for this run.
    /// * `breaker` — Compaction circuit breaker for this run.
    /// * `output_recovery_count` — Number of truncation recoveries so far.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::IterationLimitExceeded`] when the iteration
    /// budget is exhausted, [`RuntimeError::TokenBudgetExceeded`] when the
    /// token budget is exceeded, [`RuntimeError::DoomLoopDetected`] when a
    /// repetitive tool-call pattern is found, or other [`RuntimeError`]
    /// variants from provider or storage failures.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn run_loop(
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
        breaker: &mut CompactionCircuitBreaker,
        output_recovery_count: u32,
    ) -> RuntimeResult<super::RunOutput> {
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
                breaker,
                output_recovery_count,
            )
            .await;

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
        breaker: &mut CompactionCircuitBreaker,
        mut output_recovery_count: u32,
    ) -> RuntimeResult<super::RunOutput> {
        let mut resume_from = start_state;

        loop {
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
                    last_finish.as_ref(),
                    assistant_message.as_ref(),
                    assistant_msg_id,
                    &request,
                    output_recovery_count,
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
                    last_finish.as_ref(),
                    assistant_message.as_ref(),
                    assistant_msg_id,
                    &request,
                    output_recovery_count,
                )
                .await?;

                self.update_status(
                    run_id,
                    TurnTransition::status_for(TurnState::BuildingContext),
                )
                .await?;

                if self.policy.compaction.auto && !breaker.is_open() {
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

                        let compact_result = self
                            .compaction
                            .compact_if_needed(
                                &records,
                                model_ctx,
                                max_out,
                                self.store.sessions(),
                                session_id,
                            )
                            .await;

                        match compact_result {
                            Ok(Some(result)) => {
                                breaker.record_success();
                                debug!(
                                    run_id = %run_id,
                                    tokens_saved = result.tokens_saved,
                                    "proactive compaction completed"
                                );
                            }
                            Ok(None) => {}
                            Err(e) => {
                                if breaker.record_failure() {
                                    warn!(
                                        run_id = %run_id,
                                        failures = breaker.consecutive_failures(),
                                        "compaction circuit breaker opened"
                                    );
                                    self.emit(&AgentEvent::CompactionCircuitOpened(
                                        CompactionCircuitOpened {
                                            run_id,
                                            consecutive_failures: breaker.consecutive_failures(),
                                            timestamp: Utc::now(),
                                        },
                                    ));
                                }
                                return Err(e);
                            }
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
                    last_finish.as_ref(),
                    assistant_message.as_ref(),
                    assistant_msg_id,
                    &request,
                    output_recovery_count,
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

                                let compact_result = self
                                    .compaction
                                    .compact_after_overflow(
                                        &records,
                                        model_ctx,
                                        max_out,
                                        self.store.sessions(),
                                        session_id,
                                    )
                                    .await;

                                match compact_result {
                                    Ok(result) => {
                                        breaker.record_success();
                                        debug!(
                                            run_id = %run_id,
                                            tokens_saved = result.tokens_saved,
                                            "reactive compaction after provider overflow"
                                        );
                                    }
                                    Err(e) => {
                                        if breaker.record_failure() {
                                            warn!(
                                                run_id = %run_id,
                                                failures = breaker.consecutive_failures(),
                                                "compaction circuit breaker opened"
                                            );
                                            self.emit(&AgentEvent::CompactionCircuitOpened(
                                                CompactionCircuitOpened {
                                                    run_id,
                                                    consecutive_failures: breaker
                                                        .consecutive_failures(),
                                                    timestamp: Utc::now(),
                                                },
                                            ));
                                        }
                                        return Err(e);
                                    }
                                }
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

                if matches!(finish, FinishReason::Length)
                    && (output_recovery_count as usize) < self.policy.max_output_recovery_attempts
                {
                    output_recovery_count += 1;
                    let continue_msg = Message::user_text(
                        "Your previous response was truncated due to output length limit. \
                         Please continue from where you left off.",
                    );
                    self.store.append_message(session_id, &continue_msg).await?;
                    let outcome = TurnOutcome::OutputTruncated;
                    match TurnTransition::resolve(TurnState::ProcessingResponse, &outcome) {
                        TurnAction::Continue { .. } => continue,
                        _ => unreachable!(),
                    }
                }

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
                    _ => unreachable!(
                        "ProcessingResponse only produces BreakLoop, Continue, or OutputTruncated"
                    ),
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
                    last_finish.as_ref(),
                    assistant_message.as_ref(),
                    assistant_msg_id,
                    &request,
                    output_recovery_count,
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

                for call in &tool_calls {
                    if let Some(loop_type) =
                        doom_detector.record_and_check(&call.name, &call.arguments)
                    {
                        let description = match &loop_type {
                            DoomLoopType::ConsecutiveDuplicate { tool_name, count } => {
                                format!("consecutive duplicate: {tool_name} called {count} times")
                            }
                            DoomLoopType::Cycle {
                                pattern,
                                repetitions,
                            } => {
                                format!(
                                    "cycle detected: [{}] repeated {repetitions} times",
                                    pattern.join(", ")
                                )
                            }
                        };
                        self.emit(&AgentEvent::DoomLoopDetected(DoomLoopDetected {
                            run_id,
                            description: description.clone(),
                            timestamp: Utc::now(),
                        }));
                        let err = RuntimeError::DoomLoopDetected { description };
                        self.fail_run(run_id, &err).await;
                        return Err(err);
                    }
                }

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
                    last_finish.as_ref(),
                    assistant_message.as_ref(),
                    assistant_msg_id,
                    &request,
                    output_recovery_count,
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

        Ok(super::RunOutput {
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
            .map_err(|_| {
                RuntimeError::Provider(crate::error::ProviderError::Timeout {
                    provider: provider.id(),
                })
            })?
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
                _ => {}
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
            .map_err(|_| {
                RuntimeError::Provider(crate::error::ProviderError::Timeout {
                    provider: provider.id(),
                })
            })?
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
