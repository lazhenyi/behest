//! Provider-neutral chat request and response data structures.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::cache::CacheControl;
use crate::id::{ModelName, ProviderId};
use crate::tool_types::{ToolCall, ToolChoice, ToolSpec};

/// Request for a complete or streamed chat response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Provider-specific model name.
    pub model: ModelName,
    /// Ordered conversation messages.
    pub messages: Vec<Message>,
    /// Tool definitions available to the model.
    pub tools: Vec<ToolSpec>,
    /// Tool selection policy.
    pub tool_choice: ToolChoice,
    /// Optional output format constraint.
    pub response_format: Option<ResponseFormat>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    pub top_p: Option<f32>,
    /// Maximum output tokens.
    pub max_output_tokens: Option<u32>,
    /// Stop sequences.
    pub stop: Vec<String>,
    /// Application metadata forwarded to provider adapters.
    pub metadata: Value,
}

impl ChatRequest {
    /// Creates a chat request for the given model with no messages.
    #[must_use]
    pub fn new(model: ModelName) -> Self {
        Self {
            model,
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: ToolChoice::default(),
            response_format: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stop: Vec::new(),
            metadata: Value::Null,
        }
    }

    /// Appends one message to the request.
    #[must_use]
    pub fn with_message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Appends a user text message to the request.
    #[must_use]
    pub fn with_user_text(self, text: impl Into<String>) -> Self {
        self.with_message(Message::user_text(text))
    }

    /// Adds a tool definition to the request.
    #[must_use]
    pub fn with_tool(mut self, tool: ToolSpec) -> Self {
        self.tools.push(tool);
        self
    }
}

/// Chat message role and content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "role")]
#[non_exhaustive]
pub enum Message {
    /// System instruction message.
    System {
        /// Message content parts.
        content: Vec<ContentPart>,
    },
    /// User message.
    User {
        /// Message content parts.
        content: Vec<ContentPart>,
    },
    /// Assistant message, optionally including tool calls.
    Assistant {
        /// Message content parts.
        content: Vec<ContentPart>,
        /// Tool calls requested by the assistant.
        tool_calls: Vec<ToolCall>,
    },
    /// Tool result message.
    Tool {
        /// Tool call identifier being answered.
        tool_call_id: String,
        /// Tool name that produced the result.
        name: String,
        /// Tool result content parts.
        content: Vec<ContentPart>,
    },
}

impl Message {
    /// Creates a system text message.
    #[must_use]
    pub fn system_text(text: impl Into<String>) -> Self {
        Self::System {
            content: vec![ContentPart::text(text)],
        }
    }

    /// Creates a user text message.
    #[must_use]
    pub fn user_text(text: impl Into<String>) -> Self {
        Self::User {
            content: vec![ContentPart::text(text)],
        }
    }

    /// Creates an assistant text message without tool calls.
    #[must_use]
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::Assistant {
            content: vec![ContentPart::text(text)],
            tool_calls: Vec::new(),
        }
    }

    /// Creates a tool result message.
    #[must_use]
    pub fn tool_text(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            name: name.into(),
            content: vec![ContentPart::text(text)],
        }
    }

    /// Returns the tool calls from an Assistant message, or empty slice.
    #[must_use]
    pub fn tool_calls(&self) -> &[ToolCall] {
        match self {
            Self::Assistant { tool_calls, .. } => tool_calls.as_slice(),
            _ => &[],
        }
    }

    /// Marks the last content part of this message as a cache breakpoint.
    ///
    /// If the message has no content parts, the marker is a no-op. If the
    /// last content part already has a cache control, it is replaced.
    ///
    /// This is a convenience for callers that want to place a cache
    /// breakpoint at the end of a message (the most common case).
    #[must_use]
    pub fn mark_cache_breakpoint(mut self) -> Self {
        let ctrl = CacheControl::ephemeral();
        match &mut self {
            Self::System { content } | Self::User { content } | Self::Tool { content, .. } => {
                if let Some(last) = content.last_mut() {
                    last.set_cache_control(ctrl);
                }
            }
            Self::Assistant { content, .. } => {
                if let Some(last) = content.last_mut() {
                    last.set_cache_control(ctrl);
                }
            }
        }
        self
    }
}

/// A single content part inside a chat message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
#[non_exhaustive]
pub enum ContentPart {
    /// Plain text content.
    Text {
        /// Text payload.
        text: String,
        /// Optional cache marker placed at the end of this content block.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// JSON value content.
    Json {
        /// JSON payload.
        value: Value,
        /// Optional cache marker placed at the end of this content block.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Image referenced by URL.
    ImageUrl {
        /// Public or provider-accessible image URL.
        url: String,
        /// Optional MIME type hint.
        mime_type: Option<String>,
        /// Optional cache marker placed at the end of this content block.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

impl ContentPart {
    /// Creates a text content part.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            cache_control: None,
        }
    }

    /// Creates a JSON content part.
    #[must_use]
    pub fn json(value: Value) -> Self {
        Self::Json {
            value,
            cache_control: None,
        }
    }

    /// Creates an image URL content part.
    #[must_use]
    pub fn image_url(url: impl Into<String>, mime_type: Option<String>) -> Self {
        Self::ImageUrl {
            url: url.into(),
            mime_type,
            cache_control: None,
        }
    }

    /// Returns a copy of this content part with the given cache control marker.
    #[must_use]
    pub fn with_cache_control(mut self, ctrl: CacheControl) -> Self {
        self.set_cache_control(ctrl);
        self
    }

    /// Sets the cache control marker on this content part in place.
    pub fn set_cache_control(&mut self, ctrl: CacheControl) {
        match self {
            Self::Text { cache_control, .. }
            | Self::Json { cache_control, .. }
            | Self::ImageUrl { cache_control, .. } => *cache_control = Some(ctrl),
        }
    }

    /// Returns the cache control marker on this content part, if any.
    #[must_use]
    pub fn cache_control(&self) -> Option<CacheControl> {
        match self {
            Self::Text { cache_control, .. }
            | Self::Json { cache_control, .. }
            | Self::ImageUrl { cache_control, .. } => *cache_control,
        }
    }
}

/// Output format requested from a chat provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
#[non_exhaustive]
pub enum ResponseFormat {
    /// Unconstrained text output.
    Text,
    /// Provider-native JSON object mode.
    JsonObject,
    /// Provider-native JSON schema mode.
    JsonSchema {
        /// Schema name sent to the provider.
        name: String,
        /// JSON schema document.
        schema: Value,
        /// Whether provider should strictly enforce the schema.
        strict: bool,
    },
}

/// Complete chat response from a provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Provider that produced the response.
    pub provider: ProviderId,
    /// Model that produced the response.
    pub model: ModelName,
    /// Assistant message returned by the provider.
    pub message: Message,
    /// Reason the provider stopped generating.
    pub finish_reason: FinishReason,
    /// Token accounting, when supplied by the provider.
    pub usage: Option<TokenUsage>,
    /// Raw provider response for adapters that retain it.
    pub raw: Option<Value>,
}

/// Reason generation ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishReason {
    /// Natural stop condition.
    Stop,
    /// Provider stopped because tool calls were produced.
    ToolCalls,
    /// Maximum token limit was reached.
    Length,
    /// Provider content filter interrupted generation.
    ContentFilter,
    /// Provider reported an error after partial generation.
    Error,
    /// Provider-specific finish reason.
    Unknown(String),
}

/// Token usage reported by a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of input tokens.
    pub input_tokens: u64,
    /// Number of output tokens.
    pub output_tokens: u64,
    /// Total token count.
    pub total_tokens: u64,
    /// Number of input tokens written to cache by this call.
    ///
    /// Populated by Anthropic when the request created a new cache entry.
    /// `None` for providers that do not report cache writes or when no
    /// caching occurred.
    pub cache_creation_input_tokens: Option<u64>,
    /// Number of input tokens served from cache by this call.
    ///
    /// Populated by Anthropic (`cache_read_input_tokens`) and DeepSeek
    /// (`prompt_cache_hit_tokens`). `None` when the provider does not
    /// report cache reads or no cache was hit.
    pub cache_read_input_tokens: Option<u64>,
    /// Number of input tokens served from cache by this call.
    ///
    /// Populated by OpenAI's `prompt_tokens_details.cached_tokens`.
    /// `None` for providers that do not report this field.
    pub cached_input_tokens: Option<u64>,
}

impl TokenUsage {
    /// Creates token usage and computes the total.
    ///
    /// Cache fields default to `None`; use [`TokenUsage::with_cache_stats`] to
    /// populate them or [`TokenUsage::merge`] to combine two usages.
    #[must_use]
    pub const fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            cached_input_tokens: None,
        }
    }

    /// Returns a copy of this usage with the cache fields populated.
    #[must_use]
    pub const fn with_cache_stats(
        mut self,
        cache_creation_input_tokens: Option<u64>,
        cache_read_input_tokens: Option<u64>,
        cached_input_tokens: Option<u64>,
    ) -> Self {
        self.cache_creation_input_tokens = cache_creation_input_tokens;
        self.cache_read_input_tokens = cache_read_input_tokens;
        self.cached_input_tokens = cached_input_tokens;
        self
    }

    /// Returns the total number of cache-related input tokens.
    ///
    /// Sums `cache_creation_input_tokens`, `cache_read_input_tokens`, and
    /// `cached_input_tokens` when present. Returns `0` when no provider
    /// reported any cache stats.
    #[must_use]
    pub fn cache_tokens(&self) -> u64 {
        self.cache_creation_input_tokens.unwrap_or(0)
            + self.cache_read_input_tokens.unwrap_or(0)
            + self.cached_input_tokens.unwrap_or(0)
    }

    /// Merges two usages by summing all numeric fields.
    ///
    /// Numeric fields (`input_tokens`, `output_tokens`, `total_tokens`)
    /// are summed. Cache fields use the rule: `None + None = None`,
    /// `None + Some(x) = Some(x)`, `Some(x) + None = Some(x)`,
    /// `Some(x) + Some(y) = Some(x + y)`.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            input_tokens: self.input_tokens + other.input_tokens,
            output_tokens: self.output_tokens + other.output_tokens,
            total_tokens: self.input_tokens
                + other.input_tokens
                + self.output_tokens
                + other.output_tokens,
            cache_creation_input_tokens: add_opt(
                self.cache_creation_input_tokens,
                other.cache_creation_input_tokens,
            ),
            cache_read_input_tokens: add_opt(
                self.cache_read_input_tokens,
                other.cache_read_input_tokens,
            ),
            cached_input_tokens: add_opt(self.cached_input_tokens, other.cached_input_tokens),
        }
    }
}

const fn add_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some(x), Some(y)) => Some(x + y),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn token_usage_new_has_no_cache_stats() {
        let usage = TokenUsage::new(10, 5);
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
        assert_eq!(usage.cache_creation_input_tokens, None);
        assert_eq!(usage.cache_read_input_tokens, None);
        assert_eq!(usage.cached_input_tokens, None);
    }

    #[test]
    fn token_usage_with_cache_stats_populates_fields() {
        let usage = TokenUsage::new(100, 50).with_cache_stats(Some(80), Some(20), None);
        assert_eq!(usage.cache_creation_input_tokens, Some(80));
        assert_eq!(usage.cache_read_input_tokens, Some(20));
        assert_eq!(usage.cached_input_tokens, None);
    }

    #[test]
    fn token_usage_cache_tokens_sums_all_three_fields() {
        let usage = TokenUsage::new(100, 50).with_cache_stats(Some(10), Some(20), Some(30));
        assert_eq!(usage.cache_tokens(), 60);
    }

    #[test]
    fn token_usage_cache_tokens_zero_when_all_none() {
        let usage = TokenUsage::new(100, 50);
        assert_eq!(usage.cache_tokens(), 0);
    }

    #[test]
    fn token_usage_merge_sums_numeric_fields() {
        let a = TokenUsage::new(100, 50);
        let b = TokenUsage::new(200, 30);
        let merged = a.merge(b);
        assert_eq!(merged.input_tokens, 300);
        assert_eq!(merged.output_tokens, 80);
        assert_eq!(merged.total_tokens, 380);
    }

    #[test]
    fn token_usage_merge_both_none_yields_none() {
        let a = TokenUsage::new(100, 50);
        let b = TokenUsage::new(200, 30);
        let merged = a.merge(b);
        assert_eq!(merged.cache_creation_input_tokens, None);
        assert_eq!(merged.cache_read_input_tokens, None);
        assert_eq!(merged.cached_input_tokens, None);
    }

    #[test]
    fn token_usage_merge_one_some_passes_through() {
        let a = TokenUsage::new(100, 50).with_cache_stats(Some(40), Some(20), Some(10));
        let b = TokenUsage::new(200, 30);
        let merged = a.merge(b);
        assert_eq!(merged.cache_creation_input_tokens, Some(40));
        assert_eq!(merged.cache_read_input_tokens, Some(20));
        assert_eq!(merged.cached_input_tokens, Some(10));
    }

    #[test]
    fn token_usage_merge_both_some_sums() {
        let a = TokenUsage::new(100, 50).with_cache_stats(Some(40), Some(20), Some(10));
        let b = TokenUsage::new(200, 30).with_cache_stats(Some(60), Some(80), Some(90));
        let merged = a.merge(b);
        assert_eq!(merged.cache_creation_input_tokens, Some(100));
        assert_eq!(merged.cache_read_input_tokens, Some(100));
        assert_eq!(merged.cached_input_tokens, Some(100));
    }
}
