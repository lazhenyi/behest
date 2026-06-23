//! Agent event types for streaming runtime execution.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::provider::{FinishReason, ModelName, ProviderId, TokenUsage, ToolCall};

use super::run::RunId;

/// Event emitted during agent runtime execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentEvent {
    /// Run has started.
    RunStarted(RunStarted),
    /// Context has been built.
    ContextBuilt(ContextBuilt),
    /// Model call has started.
    ModelStarted(ModelStarted),
    /// Text delta from model.
    TextDelta(TextDelta),
    /// Tool call has started.
    ToolCallStarted(ToolCallStarted),
    /// Tool call arguments delta.
    ToolCallDelta(ToolCallDelta),
    /// Tool call has completed.
    ToolCallCompleted(ToolCallCompleted),
    /// Tool execution has started.
    ToolExecutionStarted(ToolExecutionStarted),
    /// Tool execution has finished.
    ToolExecutionFinished(ToolExecutionFinished),
    /// Assistant message has been committed to store.
    AssistantMessageCommitted(MessageCommitted),
    /// Tool message has been committed to store.
    ToolMessageCommitted(MessageCommitted),
    /// Usage has been recorded.
    UsageRecorded(UsageRecorded),
    /// Run has completed successfully.
    RunCompleted(RunCompleted),
    /// Run has failed.
    RunFailed(RunFailed),
    /// Run has been cancelled.
    RunCancelled(RunCancelled),
    /// Doom loop has been detected.
    DoomLoopDetected(DoomLoopDetected),
}

impl AgentEvent {
    /// Returns the run identifier for any event variant.
    #[must_use]
    pub fn run_id(&self) -> RunId {
        match self {
            AgentEvent::RunStarted(e) => e.run_id,
            AgentEvent::ContextBuilt(e) => e.run_id,
            AgentEvent::ModelStarted(e) => e.run_id,
            AgentEvent::TextDelta(e) => e.run_id,
            AgentEvent::ToolCallStarted(e) => e.run_id,
            AgentEvent::ToolCallDelta(e) => e.run_id,
            AgentEvent::ToolCallCompleted(e) => e.run_id,
            AgentEvent::ToolExecutionStarted(e) => e.run_id,
            AgentEvent::ToolExecutionFinished(e) => e.run_id,
            AgentEvent::AssistantMessageCommitted(e) | AgentEvent::ToolMessageCommitted(e) => {
                e.run_id
            }
            AgentEvent::UsageRecorded(e) => e.run_id,
            AgentEvent::RunCompleted(e) => e.run_id,
            AgentEvent::RunFailed(e) => e.run_id,
            AgentEvent::RunCancelled(e) => e.run_id,
            AgentEvent::DoomLoopDetected(e) => e.run_id,
        }
    }

    /// Returns true when this event signals a terminal run state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            AgentEvent::RunCompleted(_) | AgentEvent::RunFailed(_) | AgentEvent::RunCancelled(_)
        )
    }
}

/// Run started event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStarted {
    /// Run identifier.
    pub run_id: RunId,
    /// Session identifier.
    pub session_id: Uuid,
    /// Provider used for model calls.
    pub provider: ProviderId,
    /// Model used for generation.
    pub model: ModelName,
    /// When the run started.
    pub timestamp: DateTime<Utc>,
}

/// Context built event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBuilt {
    /// Run identifier.
    pub run_id: RunId,
    /// Number of messages in context.
    pub message_count: usize,
    /// When context was built.
    pub timestamp: DateTime<Utc>,
}

/// Model started event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStarted {
    /// Run identifier.
    pub run_id: RunId,
    /// Provider being called.
    pub provider: ProviderId,
    /// Model being used.
    pub model: ModelName,
    /// Iteration number.
    pub iteration: usize,
    /// When the model call started.
    pub timestamp: DateTime<Utc>,
}

/// Text delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDelta {
    /// Run identifier.
    pub run_id: RunId,
    /// Text delta.
    pub delta: String,
    /// When the delta was emitted.
    pub timestamp: DateTime<Utc>,
}

/// Tool call started event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallStarted {
    /// Run identifier.
    pub run_id: RunId,
    /// Tool call ID.
    pub call_id: String,
    /// Tool name.
    pub tool_name: String,
    /// When the tool call started.
    pub timestamp: DateTime<Utc>,
}

/// Tool call arguments delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Run identifier.
    pub run_id: RunId,
    /// Tool call ID.
    pub call_id: String,
    /// Arguments delta.
    pub delta: String,
    /// When the delta was emitted.
    pub timestamp: DateTime<Utc>,
}

/// Tool call completed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallCompleted {
    /// Run identifier.
    pub run_id: RunId,
    /// Completed tool call.
    pub call: ToolCall,
    /// When the tool call completed.
    pub timestamp: DateTime<Utc>,
}

/// Tool execution started event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionStarted {
    /// Run identifier.
    pub run_id: RunId,
    /// Tool call ID.
    pub call_id: String,
    /// Tool name.
    pub tool_name: String,
    /// When execution started.
    pub timestamp: DateTime<Utc>,
}

/// Tool execution finished event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionFinished {
    /// Run identifier.
    pub run_id: RunId,
    /// Tool call ID.
    pub call_id: String,
    /// Tool name.
    pub tool_name: String,
    /// Execution result.
    pub result: ToolExecutionResult,
    /// Execution duration in milliseconds.
    pub duration_ms: u64,
    /// When execution finished.
    pub timestamp: DateTime<Utc>,
}

/// Result of tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolExecutionResult {
    /// Tool executed successfully.
    Success {
        /// Output value.
        output: Value,
    },
    /// Tool execution failed.
    Failure {
        /// Error message.
        error: String,
    },
}

/// Message committed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageCommitted {
    /// Run identifier.
    pub run_id: RunId,
    /// Message ID.
    pub message_id: Uuid,
    /// When the message was committed.
    pub timestamp: DateTime<Utc>,
}

/// Usage recorded event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecorded {
    /// Run identifier.
    pub run_id: RunId,
    /// Token usage.
    pub usage: TokenUsage,
    /// When usage was recorded.
    pub timestamp: DateTime<Utc>,
}

/// Run completed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunCompleted {
    /// Run identifier.
    pub run_id: RunId,
    /// Final finish reason.
    pub finish_reason: FinishReason,
    /// Total iterations.
    pub iterations: usize,
    /// When the run completed.
    pub timestamp: DateTime<Utc>,
}

/// Run failed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunFailed {
    /// Run identifier.
    pub run_id: RunId,
    /// Error message.
    pub error: String,
    /// When the run failed.
    pub timestamp: DateTime<Utc>,
}

/// Run cancelled event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunCancelled {
    /// Run identifier.
    pub run_id: RunId,
    /// When the run was cancelled.
    pub timestamp: DateTime<Utc>,
}

/// Doom loop detected event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoomLoopDetected {
    /// Run identifier.
    pub run_id: RunId,
    /// Description of the doom loop pattern detected.
    pub description: String,
    /// When the doom loop was detected.
    pub timestamp: DateTime<Utc>,
}
