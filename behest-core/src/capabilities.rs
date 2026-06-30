//! Capability flags advertised by provider implementations.
//!
//! Used by the runtime for feature negotiation and routing decisions.

use serde::{Deserialize, Serialize};

use crate::cache::CacheTtl;

/// Static capability advertisement returned by provider implementations.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Provider supports prompt caching.
    ///
    /// `false` for providers with no caching (e.g. local models without a
    /// cache layer). `true` for providers that either accept explicit
    /// cache markers (Anthropic) or perform automatic prefix caching
    /// (OpenAI, DeepSeek).
    pub prompt_caching: bool,
    /// Maximum number of cache breakpoints the provider accepts.
    ///
    /// `Some(4)` for Anthropic (which permits up to four `cache_control`
    /// markers per request). `None` for providers that perform automatic
    /// caching without an explicit breakpoint concept (OpenAI, DeepSeek).
    pub max_cache_breakpoints: Option<u8>,
    /// Cache TTLs the provider supports.
    ///
    /// Empty for providers with automatic caching. Non-empty for Anthropic
    /// (`[FiveMinutes, OneHour]`).
    pub cache_ttl_options: Vec<CacheTtl>,
}

impl ProviderCapabilities {
    /// Returns a capability set for a basic non-streaming chat provider.
    #[must_use]
    pub fn chat() -> Self {
        Self {
            chat: true,
            ..Self::empty()
        }
    }

    /// Returns a capability set for an embedding-only provider.
    #[must_use]
    pub fn embeddings() -> Self {
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
            prompt_caching: false,
            max_cache_breakpoints: None,
            cache_ttl_options: Vec::new(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_capabilities_disables_all_flags() {
        let caps = ProviderCapabilities::default();
        assert!(!caps.chat);
        assert!(!caps.chat_stream);
        assert!(!caps.tool_calling);
        assert!(!caps.prompt_caching);
        assert_eq!(caps.max_cache_breakpoints, None);
        assert!(caps.cache_ttl_options.is_empty());
    }

    #[test]
    fn chat_helper_enables_chat_only() {
        let caps = ProviderCapabilities::chat();
        assert!(caps.chat);
        assert!(!caps.embeddings);
        assert!(!caps.prompt_caching);
    }

    #[test]
    fn empty_helper_is_all_false() {
        let caps = ProviderCapabilities::empty();
        assert!(!caps.chat);
        assert!(!caps.embeddings);
        assert!(!caps.prompt_caching);
        assert!(caps.cache_ttl_options.is_empty());
    }
}
