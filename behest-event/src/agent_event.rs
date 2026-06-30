//! Canonical agent event definitions.
//!
//! This module owns the [`AgentEvent`] enum and its 17 variant payload
//! structs. It used to live in `behest::runtime::event` but was moved
//! here so the event contract lives next to `Hook` and `HookStack` in
//! the same crate.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use behest_core::id::RunId;
use behest_core::message::{FinishReason, TokenUsage};
use behest_core::tool_types::ToolCall;

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
    /// Prompt cache metrics from a single model call.
    CacheMetrics(CacheMetrics),
    /// Run has completed successfully.
    RunCompleted(RunCompleted),
    /// Run has failed.
    RunFailed(RunFailed),
    /// Run has been cancelled.
    RunCancelled(RunCancelled),
    /// Doom loop has been detected.
    DoomLoopDetected(DoomLoopDetected),
    /// Compaction circuit breaker has opened due to repeated failures.
    CompactionCircuitOpened(CompactionCircuitOpened),
}

impl AgentEvent {
    /// Returns the run identifier for any event variant without pattern matching.
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
            AgentEvent::CacheMetrics(e) => e.run_id,
            AgentEvent::RunCompleted(e) => e.run_id,
            AgentEvent::RunFailed(e) => e.run_id,
            AgentEvent::RunCancelled(e) => e.run_id,
            AgentEvent::DoomLoopDetected(e) => e.run_id,
            AgentEvent::CompactionCircuitOpened(e) => e.run_id,
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

/// Emitted when a run begins execution after session load and input admission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStarted {
    /// Run identifier.
    pub run_id: RunId,
    /// Session identifier.
    pub session_id: Uuid,
    /// Provider used for model calls.
    pub provider: behest_core::id::ProviderId,
    /// Model used for generation.
    pub model: behest_core::id::ModelName,
    /// When the run started.
    pub timestamp: DateTime<Utc>,
}

/// Emitted after context has been assembled from session history and adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBuilt {
    /// Run identifier.
    pub run_id: RunId,
    /// Number of messages in context.
    pub message_count: usize,
    /// When context was built.
    pub timestamp: DateTime<Utc>,
}

/// Emitted when a model invocation begins, carrying iteration count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStarted {
    /// Run identifier.
    pub run_id: RunId,
    /// Provider being called.
    pub provider: behest_core::id::ProviderId,
    /// Model being used.
    pub model: behest_core::id::ModelName,
    /// Iteration number.
    pub iteration: usize,
    /// When the model call started.
    pub timestamp: DateTime<Utc>,
}

/// Streaming text delta emitted during model response generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDelta {
    /// Run identifier.
    pub run_id: RunId,
    /// Text delta.
    pub delta: String,
    /// When the delta was emitted.
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the model requests a tool call.
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

/// Streaming delta for tool call arguments during model response.
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

/// Emitted when the model finishes emitting a complete tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallCompleted {
    /// Run identifier.
    pub run_id: RunId,
    /// Completed tool call.
    pub call: ToolCall,
    /// When the tool call completed.
    pub timestamp: DateTime<Utc>,
}

/// Emitted when a tool function begins executing.
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

/// Emitted when a tool function completes, carrying the result and duration.
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

/// Result of a single tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolExecutionResult {
    /// Tool executed successfully.
    Success {
        /// Output value returned by the tool.
        output: serde_json::Value,
    },
    /// Tool execution failed.
    Failure {
        /// Error message describing why the tool failed.
        error: String,
    },
}

/// Notification that a message has been persisted to the session store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageCommitted {
    /// Run identifier.
    pub run_id: RunId,
    /// Message ID.
    pub message_id: Uuid,
    /// When the message was committed.
    pub timestamp: DateTime<Utc>,
}

/// Emitted after each model invocation to record token consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecorded {
    /// Run identifier.
    pub run_id: RunId,
    /// Token usage.
    pub usage: TokenUsage,
    /// When usage was recorded.
    pub timestamp: DateTime<Utc>,
}

/// Emitted after each model call when the provider reported prompt cache
/// statistics. All fields are zero when the provider did not report cache
/// hits or writes for this call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Run identifier.
    pub run_id: RunId,
    /// Number of input tokens written to the cache by this call
    /// (Anthropic: `cache_creation_input_tokens`).
    pub cache_creation_input_tokens: u64,
    /// Number of input tokens served from the cache by this call
    /// (Anthropic: `cache_read_input_tokens`, DeepSeek: `prompt_cache_hit_tokens`).
    pub cache_read_input_tokens: u64,
    /// Number of input tokens served from the cache by this call
    /// (OpenAI: `prompt_tokens_details.cached_tokens`).
    pub cached_input_tokens: u64,
    /// When the metrics were recorded.
    pub timestamp: DateTime<Utc>,
}

impl CacheMetrics {
    /// Returns the total cache-related input tokens.
    #[must_use]
    pub const fn total_cache_tokens(&self) -> u64 {
        self.cache_creation_input_tokens + self.cache_read_input_tokens + self.cached_input_tokens
    }
}

/// Terminal event emitted when a run finishes successfully.
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

/// Terminal event emitted when a run terminates with an error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunFailed {
    /// Run identifier.
    pub run_id: RunId,
    /// Error message.
    pub error: String,
    /// When the run failed.
    pub timestamp: DateTime<Utc>,
}

/// Terminal event emitted when a run is cancelled before completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunCancelled {
    /// Run identifier.
    pub run_id: RunId,
    /// When the run was cancelled.
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the agent is detected to be in a repetitive tool call cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoomLoopDetected {
    /// Run identifier.
    pub run_id: RunId,
    /// Description of the doom loop pattern detected.
    pub description: String,
    /// When the doom loop was detected.
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the compaction circuit breaker opens due to repeated failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionCircuitOpened {
    /// Run identifier.
    pub run_id: RunId,
    /// Number of consecutive failures that triggered the breaker.
    pub consecutive_failures: u32,
    /// When the breaker opened.
    pub timestamp: DateTime<Utc>,
}
