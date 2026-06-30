//! Anthropic wire types matching the `/v1/messages` API.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Request body for `POST /v1/messages`.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicRequest {
    /// Model identifier.
    pub model: String,
    /// Maximum output tokens.
    pub max_tokens: u32,
    /// System prompt, extracted from system messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Conversation messages (excluding system).
    pub messages: Vec<AnthropicMessage>,
    /// Tool definitions.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AnthropicToolDef>,
    /// Tool selection policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    /// Whether to stream the response as SSE.
    pub stream: bool,
}

/// A single message in an Anthropic conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    /// Message role (`user` or `assistant`).
    pub role: String,
    /// Content blocks.
    pub content: Vec<AnthropicContentBlock>,
}

/// Content block inside an Anthropic message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    /// Plain text content.
    #[serde(rename = "text")]
    Text {
        /// Text payload.
        text: String,
    },
    /// Image content.
    #[serde(rename = "image")]
    Image {
        /// Image source details.
        source: AnthropicImageSource,
    },
    /// Tool invocation by the assistant.
    #[serde(rename = "tool_use")]
    ToolUse {
        /// Tool call identifier.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input arguments.
        input: Value,
    },
    /// Tool result provided by the caller.
    #[serde(rename = "tool_result")]
    ToolResult {
        /// Tool call identifier being answered.
        tool_use_id: String,
        /// Result content (only text and image blocks are valid here).
        content: Vec<AnthropicToolResultContent>,
    },
}

/// Content block allowed inside a tool result.
///
/// Unlike [`AnthropicContentBlock`], this type deliberately excludes `ToolResult`
/// and `ToolUse` variants to prevent recursive nesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicToolResultContent {
    /// Plain text content.
    #[serde(rename = "text")]
    Text {
        /// Text payload.
        text: String,
    },
    /// Image content.
    #[serde(rename = "image")]
    Image {
        /// Image source details.
        source: AnthropicImageSource,
    },
}

/// Image source specification for Anthropic vision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicImageSource {
    /// Source type, always `"url"` for URL-based images.
    #[serde(rename = "type")]
    pub source_type: String,
    /// Image URL.
    pub url: String,
    /// Optional MIME type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

/// Tool definition for Anthropic function calling.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicToolDef {
    /// Tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON schema for accepted input.
    pub input_schema: Value,
}

/// Response body for `POST /v1/messages`.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicResponse {
    /// Response identifier.
    pub id: String,
    /// Model that served the response.
    pub model: String,
    /// Content blocks in the response.
    pub content: Vec<AnthropicContentBlock>,
    /// Reason generation stopped.
    pub stop_reason: Option<String>,
    /// Token usage accounting.
    pub usage: Option<AnthropicUsage>,
}

/// Token usage from an Anthropic response.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicUsage {
    /// Input token count.
    pub input_tokens: u64,
    /// Output token count.
    pub output_tokens: u64,
}

/// Streaming event from Anthropic SSE.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicStreamEvent {
    /// Stream has started, includes the full message metadata.
    #[serde(rename = "message_start")]
    MessageStart {
        /// Message metadata.
        message: AnthropicStreamMessage,
    },
    /// A new content block has started.
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        /// Content block index.
        index: usize,
        /// Initial content block data.
        content_block: AnthropicContentBlock,
    },
    /// Incremental content block delta.
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        /// Content block index.
        index: usize,
        /// Delta details.
        delta: AnthropicDelta,
    },
    /// A content block has ended.
    #[serde(rename = "content_block_stop")]
    ContentBlockStop {
        /// Content block index.
        index: usize,
    },
    /// Message-level delta (e.g., stop reason).
    #[serde(rename = "message_delta")]
    MessageDelta {
        /// Delta details.
        delta: AnthropicMessageDelta,
        /// Token usage for output.
        usage: Option<AnthropicUsage>,
    },
    /// Stream has ended.
    #[serde(rename = "message_stop")]
    MessageStop,
    /// Unknown or unhandled event type.
    #[serde(other)]
    Other,
}

/// Message metadata in a stream start event.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicStreamMessage {
    /// Model serving the stream.
    pub model: String,
}

/// Delta inside a content block.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicDelta {
    /// Text content delta.
    #[serde(rename = "text_delta")]
    TextDelta {
        /// Text increment.
        text: String,
    },
    /// Tool call input JSON delta.
    #[serde(rename = "input_json_delta")]
    InputJsonDelta {
        /// Partial JSON string.
        partial_json: String,
    },
    /// Unknown delta type.
    #[serde(other)]
    Other,
}

/// Message-level delta in a stream.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessageDelta {
    /// Stop reason, if generation has ended.
    pub stop_reason: Option<String>,
}

/// Anthropic API error response body, deserialized from non-2xx responses.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicErrorBody {
    /// Error details.
    #[serde(rename = "error")]
    pub detail: Option<AnthropicErrorDetail>,
}

/// Error detail inside an Anthropic error response.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicErrorDetail {
    /// Error type or code.
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// Error message.
    pub message: Option<String>,
}
