//! OpenAI wire types matching the `/v1/chat/completions` and `/v1/embeddings` API.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// -- Chat completion request --

/// Request body for `POST /v1/chat/completions`.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatRequest {
    /// Model identifier.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<OpenAiMessage>,
    /// Tool definitions available to the model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OpenAiToolDef>,
    /// Tool selection policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// Output format constraint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Maximum output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Whether to stream the response as SSE.
    pub stream: bool,
}

/// A single message in an OpenAI chat request or response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiMessage {
    /// Message role.
    pub role: String,
    /// Text content, or `null` for tool-only messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    /// Tool calls requested by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
    /// Tool call identifier being answered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Tool definition in OpenAI function-calling format.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAiToolDef {
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Function specification.
    pub function: OpenAiFunctionDef,
}

/// Function specification inside a tool definition.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAiFunctionDef {
    /// Function name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON schema for accepted arguments.
    pub parameters: Value,
}

/// Tool call in an OpenAI assistant message or stream delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolCall {
    /// Call identifier.
    pub id: Option<String>,
    /// Call index in a stream delta.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// Function call details.
    pub function: OpenAiFunctionCall,
}

/// Function invocation details inside a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunctionCall {
    /// Function name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// JSON arguments as a string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// -- Chat completion response --

/// Response body for `POST /v1/chat/completions`.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiChatResponse {
    /// Response identifier.
    pub id: String,
    /// Model that served the response.
    pub model: String,
    /// Response choices.
    pub choices: Vec<OpenAiChatChoice>,
    /// Token usage accounting.
    pub usage: Option<OpenAiUsage>,
}

/// One choice inside a chat completion response.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiChatChoice {
    /// Choice index.
    pub index: u32,
    /// Assistant message for this choice.
    pub message: OpenAiMessage,
    /// Reason generation stopped.
    pub finish_reason: Option<String>,
}

/// Token usage from an OpenAI response.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiUsage {
    /// Input token count.
    pub prompt_tokens: u64,
    /// Output token count.
    pub completion_tokens: u64,
    /// Total token count.
    pub total_tokens: u64,
    /// Optional breakdown of prompt tokens, including cached tokens.
    #[serde(default)]
    pub prompt_tokens_details: Option<OpenAiPromptTokensDetails>,
}

/// Breakdown of prompt tokens reported by OpenAI.
///
/// Populated when the API returns a `prompt_tokens_details` object.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OpenAiPromptTokensDetails {
    /// Number of input tokens served from the prompt cache.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

// -- Streaming delta --

/// A single SSE chunk from a streaming chat completion.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiStreamChunk {
    /// Chunk identifier.
    pub id: Option<String>,
    /// Model serving the stream.
    pub model: Option<String>,
    /// Delta choices.
    pub choices: Vec<OpenAiStreamChoice>,
    /// Token usage, only in the final chunk.
    pub usage: Option<OpenAiUsage>,
}

/// One choice inside a streaming delta.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiStreamChoice {
    /// Choice index.
    pub index: u32,
    /// Incremental content delta.
    pub delta: OpenAiDelta,
    /// Reason generation stopped, if this is the final chunk.
    pub finish_reason: Option<String>,
}

/// Content delta inside a streaming choice.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiDelta {
    /// Role, usually present only in the first delta.
    pub role: Option<String>,
    /// Text content delta.
    pub content: Option<String>,
    /// Tool call deltas.
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

// -- Embedding types --

/// Request body for `POST /v1/embeddings`.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAiEmbeddingRequest {
    /// Embedding model identifier.
    pub model: String,
    /// Text inputs to embed.
    pub input: Vec<String>,
    /// Target embedding dimension.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
}

/// Response body for `POST /v1/embeddings`.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiEmbeddingResponse {
    /// Model that produced the embeddings.
    pub model: String,
    /// Embedding vectors.
    pub data: Vec<OpenAiEmbeddingData>,
    /// Token usage accounting.
    pub usage: Option<OpenAiUsage>,
}

/// One embedding vector in the response.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiEmbeddingData {
    /// Input index this vector corresponds to.
    pub index: usize,
    /// Dense embedding vector.
    pub embedding: Vec<f32>,
}

/// OpenAI API error response body, deserialized from non-2xx responses.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiErrorBody {
    /// Error details.
    pub error: Option<OpenAiErrorDetail>,
}

/// Error detail inside an OpenAI error response.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiErrorDetail {
    /// Error message.
    pub message: Option<String>,
    /// Error type or code.
    #[serde(rename = "type")]
    pub kind: Option<String>,
}
