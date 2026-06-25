//! Event mapping between runtime [`AgentEvent`] and gRPC event proto.
//!
//! Converts every variant of the runtime event model into its
//! corresponding protobuf representation, including optional W3C
//! trace context propagation via the `otel` feature flag.

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
///
/// Each event variant is mapped to its corresponding oneof kind.
/// A monotonically increasing `sequence` number and the originating
/// `run_id` are embedded in every event. W3C trace context from the
/// current tracing span is included when the `otel` feature is active.
///
/// # Parameters
///
/// * `event` - The runtime agent event to convert.
/// * `sequence` - Monotonic event sequence number for ordering.
/// * `run_id` - The originating run identifier.
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

    let (trace_id, span_id) = current_trace_context();

    PbAgentEvent {
        sequence,
        run_id: run_id.to_owned(),
        event: Some(event_kind),
        timestamp: Some(crate::grpc::to_prost_timestamp(chrono::Utc::now())),
        trace_id,
        span_id,
    }
}

/// Extracts W3C trace context from the current tracing span.
///
/// Returns `(trace_id, span_id)` as hex strings. Both are empty when the `otel`
/// feature is disabled or no active span carries OTel context.
fn current_trace_context() -> (String, String) {
    #[cfg(feature = "otel")]
    {
        use opentelemetry::trace::TraceContextExt as _;
        use tracing_opentelemetry::OpenTelemetrySpanExt as _;
        let ctx = tracing::Span::current().context();
        let span_ref = ctx.span();
        let sc = span_ref.span_context();
        if sc.is_valid() {
            return (sc.trace_id().to_string(), sc.span_id().to_string());
        }
    }
    (String::new(), String::new())
}

/// Maps a runtime [`FinishReason`] to the protobuf enum value.
///
/// Returns an `i32` corresponding to the proto enum:
/// 1 = Stop, 2 = ToolCalls, 3 = Length, 4 = ContentFilter,
/// 5 = Error, 6 = Unknown.
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
