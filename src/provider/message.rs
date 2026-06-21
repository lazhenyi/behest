//! Provider-neutral chat request and response data structures.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::provider::{ModelName, ProviderId, ToolCall, ToolChoice, ToolSpec};

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
}

/// A single content part inside a chat message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ContentPart {
    /// Plain text content.
    Text {
        /// Text payload.
        text: String,
    },
    /// JSON value content.
    Json {
        /// JSON payload.
        value: Value,
    },
    /// Image referenced by URL.
    ImageUrl {
        /// Public or provider-accessible image URL.
        url: String,
        /// Optional MIME type hint.
        mime_type: Option<String>,
    },
}

impl ContentPart {
    /// Creates a text content part.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Creates a JSON content part.
    #[must_use]
    pub fn json(value: Value) -> Self {
        Self::Json { value }
    }

    /// Creates an image URL content part.
    #[must_use]
    pub fn image_url(url: impl Into<String>, mime_type: Option<String>) -> Self {
        Self::ImageUrl {
            url: url.into(),
            mime_type,
        }
    }
}

/// Output format requested from a chat provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
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
}

impl TokenUsage {
    /// Creates token usage and computes the total.
    #[must_use]
    pub const fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        }
    }
}
