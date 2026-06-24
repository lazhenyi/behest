//! Event mapping between runtime [`AgentEvent`] and gRPC event proto.

use crate::runtime::AgentEvent;

use super::pb::{AgentEvent as PbAgentEvent, Timestamp};

/// Converts a runtime [`AgentEvent`] to the protobuf representation.
#[must_use]
pub fn to_proto(event: &AgentEvent, sequence: u64, run_id: &str) -> PbAgentEvent {
    let (event_type, data) = event_payload(event);
    PbAgentEvent {
        sequence,
        run_id: run_id.to_owned(),
        event_type: event_type.to_owned(),
        data: serde_json::to_string(&data).unwrap_or_default(),
        timestamp: Some(Timestamp {
            value: chrono::Utc::now().to_rfc3339(),
        }),
    }
}

fn event_payload(event: &AgentEvent) -> (&'static str, serde_json::Value) {
    match event {
        AgentEvent::RunStarted(e) => (
            "run.started",
            serde_json::json!({"run_id": e.run_id.to_string(), "session_id": e.session_id.to_string()}),
        ),
        AgentEvent::ContextBuilt(e) => (
            "context.built",
            serde_json::json!({"run_id": e.run_id.to_string(), "message_count": e.message_count}),
        ),
        AgentEvent::ModelStarted(e) => (
            "model.started",
            serde_json::json!({"run_id": e.run_id.to_string(), "provider": e.provider.to_string(), "model": e.model.to_string(), "iteration": e.iteration}),
        ),
        AgentEvent::TextDelta(e) => (
            "text.delta",
            serde_json::json!({"run_id": e.run_id.to_string(), "delta": e.delta}),
        ),
        AgentEvent::ToolCallStarted(e) => (
            "tool_call.started",
            serde_json::json!({"run_id": e.run_id.to_string(), "call_id": e.call_id, "tool_name": e.tool_name}),
        ),
        AgentEvent::ToolCallDelta(e) => (
            "tool_call.delta",
            serde_json::json!({"run_id": e.run_id.to_string(), "call_id": e.call_id, "delta": e.delta}),
        ),
        AgentEvent::ToolCallCompleted(e) => (
            "tool_call.completed",
            serde_json::json!({"run_id": e.run_id.to_string(), "call": serde_json::to_value(&e.call).unwrap_or_default()}),
        ),
        AgentEvent::ToolExecutionStarted(e) => (
            "tool_execution.started",
            serde_json::json!({"run_id": e.run_id.to_string(), "call_id": e.call_id, "tool_name": e.tool_name}),
        ),
        AgentEvent::ToolExecutionFinished(e) => (
            "tool_execution.finished",
            serde_json::json!({"run_id": e.run_id.to_string(), "call_id": e.call_id, "tool_name": e.tool_name, "duration_ms": e.duration_ms}),
        ),
        AgentEvent::AssistantMessageCommitted(e) | AgentEvent::ToolMessageCommitted(e) => (
            "message.committed",
            serde_json::json!({"run_id": e.run_id.to_string(), "message_id": e.message_id.to_string()}),
        ),
        AgentEvent::UsageRecorded(e) => (
            "usage.recorded",
            serde_json::json!({"run_id": e.run_id.to_string(), "usage": e.usage}),
        ),
        AgentEvent::RunCompleted(e) => (
            "run.completed",
            serde_json::json!({"run_id": e.run_id.to_string(), "finish_reason": e.finish_reason, "iterations": e.iterations}),
        ),
        AgentEvent::RunFailed(e) => (
            "run.failed",
            serde_json::json!({"run_id": e.run_id.to_string(), "error": e.error}),
        ),
        AgentEvent::RunCancelled(e) => (
            "run.cancelled",
            serde_json::json!({"run_id": e.run_id.to_string()}),
        ),
        AgentEvent::DoomLoopDetected(e) => (
            "doom_loop.detected",
            serde_json::json!({"run_id": e.run_id.to_string(), "description": e.description}),
        ),
        AgentEvent::CompactionCircuitOpened(e) => (
            "compaction.circuit_opened",
            serde_json::json!({"run_id": e.run_id.to_string(), "consecutive_failures": e.consecutive_failures}),
        ),
    }
}
