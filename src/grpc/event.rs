//! Event mapping between runtime [`AgentEvent`] and gRPC event proto.

use crate::provider::FinishReason;
use crate::runtime::AgentEvent;

use super::pb::{
    AgentEvent as PbAgentEvent, AssistantMessageCommittedEvent, CompactionCircuitOpenedEvent,
    ContextBuiltEvent, DoomLoopDetectedEvent, ModelStartedEvent, RunCancelledEvent,
    RunCompletedEvent, RunFailedEvent, RunStartedEvent, TextDeltaEvent, ToolCallCompletedEvent,
    ToolCallDeltaEvent, ToolCallStartedEvent, ToolExecutionFinishedEvent,
    ToolExecutionStartedEvent, ToolMessageCommittedEvent, UsageRecordedEvent,
    agent_event::Event as EventKind,
};

/// Converts a runtime [`AgentEvent`] to the protobuf representation.
#[must_use]
pub fn to_proto(event: &AgentEvent, sequence: u64, run_id: &str) -> PbAgentEvent {
    let event_kind = match event {
        AgentEvent::RunStarted(e) => EventKind::RunStarted(RunStartedEvent {
            session_id: e.session_id.to_string(),
            provider: e.provider.to_string(),
            model: e.model.to_string(),
        }),
        AgentEvent::ContextBuilt(e) => EventKind::ContextBuilt(ContextBuiltEvent {
            message_count: u32::try_from(e.message_count).unwrap_or(u32::MAX),
        }),
        AgentEvent::ModelStarted(e) => EventKind::ModelStarted(ModelStartedEvent {
            provider: e.provider.to_string(),
            model: e.model.to_string(),
            iteration: u32::try_from(e.iteration).unwrap_or(u32::MAX),
        }),
        AgentEvent::TextDelta(e) => EventKind::TextDelta(TextDeltaEvent {
            delta: e.delta.clone(),
        }),
        AgentEvent::ToolCallStarted(e) => EventKind::ToolCallStarted(ToolCallStartedEvent {
            call_id: e.call_id.clone(),
            tool_name: e.tool_name.clone(),
        }),
        AgentEvent::ToolCallDelta(e) => EventKind::ToolCallDelta(ToolCallDeltaEvent {
            call_id: e.call_id.clone(),
            delta: e.delta.clone(),
        }),
        AgentEvent::ToolCallCompleted(e) => EventKind::ToolCallCompleted(ToolCallCompletedEvent {
            call: Some(super::pb::ToolCall {
                id: e.call.id.clone(),
                name: e.call.name.clone(),
                arguments: e.call.arguments.to_string(),
            }),
        }),
        AgentEvent::ToolExecutionStarted(e) => {
            EventKind::ToolExecutionStarted(ToolExecutionStartedEvent {
                call_id: e.call_id.clone(),
                tool_name: e.tool_name.clone(),
            })
        }
        AgentEvent::ToolExecutionFinished(e) => {
            EventKind::ToolExecutionFinished(ToolExecutionFinishedEvent {
                call_id: e.call_id.clone(),
                tool_name: e.tool_name.clone(),
                duration_ms: e.duration_ms,
            })
        }
        AgentEvent::AssistantMessageCommitted(e) => {
            EventKind::AssistantMessageCommitted(AssistantMessageCommittedEvent {
                message_id: e.message_id.to_string(),
            })
        }
        AgentEvent::ToolMessageCommitted(e) => {
            EventKind::ToolMessageCommitted(ToolMessageCommittedEvent {
                message_id: e.message_id.to_string(),
            })
        }
        AgentEvent::UsageRecorded(e) => EventKind::UsageRecorded(UsageRecordedEvent {
            usage: Some(super::pb::TokenUsage {
                input_tokens: e.usage.input_tokens,
                output_tokens: e.usage.output_tokens,
                total_tokens: e.usage.total_tokens,
            }),
        }),
        AgentEvent::RunCompleted(e) => EventKind::RunCompleted(RunCompletedEvent {
            finish_reason: finish_reason_to_proto(&e.finish_reason),
            iterations: u32::try_from(e.iterations).unwrap_or(u32::MAX),
        }),
        AgentEvent::RunFailed(e) => EventKind::RunFailed(RunFailedEvent {
            error: e.error.clone(),
        }),
        AgentEvent::RunCancelled(_) => EventKind::RunCancelled(RunCancelledEvent {}),
        AgentEvent::DoomLoopDetected(e) => EventKind::DoomLoopDetected(DoomLoopDetectedEvent {
            description: e.description.clone(),
        }),
        AgentEvent::CompactionCircuitOpened(e) => {
            EventKind::CompactionCircuitOpened(CompactionCircuitOpenedEvent {
                consecutive_failures: e.consecutive_failures,
            })
        }
    };

    PbAgentEvent {
        sequence,
        run_id: run_id.to_owned(),
        event: Some(event_kind),
        timestamp: Some(crate::grpc::to_prost_timestamp(chrono::Utc::now())),
    }
}

/// Maps a runtime [`FinishReason`] to the protobuf enum value.
pub(super) fn finish_reason_to_proto(fr: &FinishReason) -> i32 {
    match fr {
        FinishReason::Stop => 1,
        FinishReason::ToolCalls => 2,
        FinishReason::Length => 3,
        FinishReason::ContentFilter => 4,
        FinishReason::Error => 5,
        FinishReason::Unknown(_) => 6,
    }
}
