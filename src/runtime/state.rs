//! Event-sourced run state reconstruction.
//!
//! [`RunState`] is a materialized view built by folding [`AgentEvent`] records.
//! It supports replayable state reconstruction from event logs and batch
//! reload of multiple runs.

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::provider::{FinishReason, ModelName, ProviderId, TokenUsage};

use super::event::AgentEvent;
use super::run::{RunId, RunRecord, RunStatus};
use super::store::RunEventRecord;

/// Projected state of a run, reconstructed from its event log.
///
/// Unlike [`RunRecord`] which is a static metadata snapshot,
/// `RunState` is a fully materialized view derived by folding every
/// recorded event via [`apply`](Self::apply). This enables replay-based
/// state recovery and batch inspection of run progress without scanning
/// raw events.
#[derive(Debug, Clone)]
pub struct RunState {
    /// Run identifier.
    pub run_id: RunId,
    /// Session this run belongs to.
    pub session_id: uuid::Uuid,
    /// Current status.
    pub status: RunStatus,
    /// Provider used for model calls.
    pub provider: ProviderId,
    /// Model used for generation.
    pub model: ModelName,
    /// Run metadata.
    pub metadata: Value,
    /// Current iteration (0 before first model call).
    pub iteration: usize,
    /// Aggregated token usage across all provider calls.
    pub total_usage: TokenUsage,
    /// Finish reason from the last model response.
    pub last_finish: Option<FinishReason>,
    /// Error message if the run failed.
    pub last_error: Option<String>,
    /// Number of events folded into this state.
    pub event_count: usize,
    /// When the run was created.
    pub created_at: DateTime<Utc>,
    /// When the last event was recorded.
    pub updated_at: DateTime<Utc>,
}

impl RunState {
    /// Creates a [`RunState`] by folding the run record metadata with its
    /// event log.
    ///
    /// The `record` provides static metadata (provider, model, created_at)
    /// while `events` drive the dynamic state projection by applying each
    /// event in order via [`apply`](Self::apply).
    #[must_use]
    pub fn create(record: &RunRecord, events: &[RunEventRecord]) -> Self {
        let updated_at = events.last().map_or(record.updated_at, |e| e.timestamp);

        let mut state = RunState {
            run_id: record.id,
            session_id: record.session_id,
            status: RunStatus::Pending,
            provider: record.provider.clone(),
            model: record.model.clone(),
            metadata: record.metadata.clone(),
            iteration: 0,
            total_usage: TokenUsage::new(0, 0),
            last_finish: None,
            last_error: None,
            event_count: events.len(),
            created_at: record.created_at,
            updated_at,
        };

        for event_record in events {
            state.apply(&event_record.event);
        }

        state
    }

    /// Incrementally updates the projected state by applying a single event.
    ///
    /// Handles all [`AgentEvent`] variants: `RunStarted` sets provider/model,
    /// `ModelStarted` advances iteration, `UsageRecorded` accumulates tokens,
    /// terminal events set the final status, while informational events
    /// only update the timestamp.
    pub fn apply(&mut self, event: &AgentEvent) {
        match event {
            AgentEvent::RunStarted(e) => {
                self.status = RunStatus::SessionLoaded;
                // RunStarted carries provider/model for self-contained reconstruction.
                self.provider = e.provider.clone();
                self.model = e.model.clone();
                self.updated_at = e.timestamp;
            }
            AgentEvent::ModelStarted(e) => {
                self.iteration = e.iteration;
                self.status = RunStatus::CallingModel;
                self.updated_at = e.timestamp;
            }
            AgentEvent::UsageRecorded(e) => {
                self.total_usage = TokenUsage::new(
                    self.total_usage.input_tokens + e.usage.input_tokens,
                    self.total_usage.output_tokens + e.usage.output_tokens,
                );
                self.updated_at = e.timestamp;
            }
            AgentEvent::RunCompleted(e) => {
                self.status = RunStatus::Completed;
                self.last_finish = Some(e.finish_reason.clone());
                self.updated_at = e.timestamp;
            }
            AgentEvent::RunFailed(e) => {
                self.status = RunStatus::Failed;
                self.last_error = Some(e.error.clone());
                self.updated_at = e.timestamp;
            }
            AgentEvent::RunCancelled(e) => {
                self.status = RunStatus::Cancelled;
                self.updated_at = e.timestamp;
            }
            AgentEvent::ContextBuilt(e) => {
                self.updated_at = e.timestamp;
            }
            AgentEvent::TextDelta(e) => {
                self.updated_at = e.timestamp;
            }
            AgentEvent::ToolCallStarted(e) => {
                self.updated_at = e.timestamp;
            }
            AgentEvent::ToolCallDelta(e) => {
                self.updated_at = e.timestamp;
            }
            AgentEvent::ToolCallCompleted(e) => {
                self.updated_at = e.timestamp;
            }
            AgentEvent::ToolExecutionStarted(e) => {
                self.updated_at = e.timestamp;
            }
            AgentEvent::ToolExecutionFinished(e) => {
                self.updated_at = e.timestamp;
            }
            AgentEvent::AssistantMessageCommitted(e) | AgentEvent::ToolMessageCommitted(e) => {
                self.updated_at = e.timestamp;
            }
            AgentEvent::DoomLoopDetected(e) => {
                self.updated_at = e.timestamp;
            }
            AgentEvent::CompactionCircuitOpened(e) => {
                self.updated_at = e.timestamp;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::event::{
        ModelStarted as ModelStartedEvent, RunCompleted as RunCompletedEvent,
        RunFailed as RunFailedEvent, RunStarted as RunStartedEvent,
        UsageRecorded as UsageRecordedEvent,
    };
    use super::*;

    fn make_record() -> RunRecord {
        RunRecord::new(
            RunId::new(),
            uuid::Uuid::new_v4(),
            ProviderId::new("test-provider"),
            ModelName::new("test-model"),
            Value::Null,
            None,
        )
    }

    #[test]
    fn rebuilds_completed_run_from_events() {
        let record = make_record();
        let run_id = record.id;
        let session_id = record.session_id;

        let events = vec![
            RunEventRecord::new(
                0,
                run_id,
                AgentEvent::RunStarted(RunStartedEvent {
                    run_id,
                    session_id,
                    provider: record.provider.clone(),
                    model: record.model.clone(),
                    timestamp: Utc::now(),
                }),
            ),
            RunEventRecord::new(
                1,
                run_id,
                AgentEvent::ModelStarted(ModelStartedEvent {
                    run_id,
                    provider: record.provider.clone(),
                    model: record.model.clone(),
                    iteration: 1,
                    timestamp: Utc::now(),
                }),
            ),
            RunEventRecord::new(
                2,
                run_id,
                AgentEvent::UsageRecorded(UsageRecordedEvent {
                    run_id,
                    usage: TokenUsage::new(100, 50),
                    timestamp: Utc::now(),
                }),
            ),
            RunEventRecord::new(
                3,
                run_id,
                AgentEvent::RunCompleted(RunCompletedEvent {
                    run_id,
                    finish_reason: FinishReason::Stop,
                    iterations: 1,
                    timestamp: Utc::now(),
                }),
            ),
        ];

        let state = RunState::create(&record, &events);

        assert_eq!(state.run_id, run_id);
        assert_eq!(state.session_id, session_id);
        assert_eq!(state.status, RunStatus::Completed);
        assert_eq!(state.provider, record.provider);
        assert_eq!(state.model, record.model);
        assert_eq!(state.iteration, 1);
        assert_eq!(state.total_usage, TokenUsage::new(100, 50));
        assert_eq!(state.last_finish, Some(FinishReason::Stop));
        assert!(state.last_error.is_none());
        assert_eq!(state.event_count, 4);
    }

    #[test]
    fn rebuilds_failed_run_from_events() {
        let record = make_record();
        let run_id = record.id;
        let session_id = record.session_id;

        let events = vec![
            RunEventRecord::new(
                0,
                run_id,
                AgentEvent::RunStarted(RunStartedEvent {
                    run_id,
                    session_id,
                    provider: record.provider.clone(),
                    model: record.model.clone(),
                    timestamp: Utc::now(),
                }),
            ),
            RunEventRecord::new(
                1,
                run_id,
                AgentEvent::RunFailed(RunFailedEvent {
                    run_id,
                    error: "something broke".to_string(),
                    timestamp: Utc::now(),
                }),
            ),
        ];

        let state = RunState::create(&record, &events);

        assert_eq!(state.status, RunStatus::Failed);
        assert_eq!(state.last_error, Some("something broke".to_string()));
        assert!(state.last_finish.is_none());
    }

    #[test]
    fn empty_events_returns_pending() {
        let record = make_record();
        let state = RunState::create(&record, &[]);

        assert_eq!(state.status, RunStatus::Pending);
        assert_eq!(state.iteration, 0);
        assert_eq!(state.event_count, 0);
    }
}
