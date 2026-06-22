//! Streaming event schema for chat providers.

use serde::{Deserialize, Serialize};

use crate::provider::{FinishReason, ModelName, ProviderId, TokenUsage, ToolCall};

/// Event emitted by a streaming chat provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
#[non_exhaustive]
pub enum ChatStreamEvent {
    /// Stream has started.
    Started {
        /// Provider serving the stream.
        provider: ProviderId,
        /// Model serving the stream.
        model: ModelName,
    },
    /// Text delta produced by the model.
    TextDelta {
        /// Incremental text chunk.
        delta: String,
    },
    /// Tool call has started.
    ToolCallStarted {
        /// Provider-generated call identifier.
        id: String,
        /// Tool name requested by the model.
        name: String,
    },
    /// Tool call argument JSON delta.
    ToolCallArgumentsDelta {
        /// Provider-generated call identifier.
        id: String,
        /// Incremental argument chunk.
        delta: String,
    },
    /// Tool call has completed and can be executed.
    ToolCallCompleted {
        /// Completed tool call.
        call: ToolCall,
    },
    /// Stream has finished.
    Finished {
        /// Reason the provider stopped generation.
        finish_reason: FinishReason,
        /// Token accounting, when supplied by the provider.
        usage: Option<TokenUsage>,
    },
}
