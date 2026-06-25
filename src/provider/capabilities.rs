//! Capability flags advertised by provider implementations.

use serde::{Deserialize, Serialize};

/// Static capability advertisement returned by provider implementations.
///
/// Used by the runtime for feature negotiation and routing decisions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Provider can return a complete chat response.
    pub chat: bool,
    /// Provider can stream chat events.
    pub chat_stream: bool,
    /// Provider accepts tool definitions in chat requests.
    pub tool_calling: bool,
    /// Provider can execute or request multiple tool calls concurrently.
    pub parallel_tool_calls: bool,
    /// Provider can constrain output with a JSON schema.
    pub json_schema_output: bool,
    /// Provider can consume image content.
    pub vision: bool,
    /// Provider can produce embeddings.
    pub embeddings: bool,
    /// Maximum supported input tokens, when known.
    pub max_input_tokens: Option<u32>,
    /// Maximum supported output tokens, when known.
    pub max_output_tokens: Option<u32>,
}

impl ProviderCapabilities {
    /// Returns a capability set for a basic non-streaming chat provider.
    #[must_use]
    pub const fn chat() -> Self {
        Self {
            chat: true,
            ..Self::empty()
        }
    }

    /// Returns a capability set for an embedding-only provider.
    #[must_use]
    pub const fn embeddings() -> Self {
        Self {
            embeddings: true,
            ..Self::empty()
        }
    }

    /// Returns an empty capability set.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            chat: false,
            chat_stream: false,
            tool_calling: false,
            parallel_tool_calls: false,
            json_schema_output: false,
            vision: false,
            embeddings: false,
            max_input_tokens: None,
            max_output_tokens: None,
        }
    }
}
