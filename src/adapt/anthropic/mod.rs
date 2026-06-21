//! Anthropic Claude provider adapter.
//!
//! Supports chat completions (with streaming), tool calling, and vision.
//! Embedding is not supported by the Anthropic API.
//!
//! # Examples
//!
//! ```no_run
//! use agents::adapt::anthropic::AnthropicChatAdapter;
//! use agents::provider::ProviderHttpConfig;
//! use agents::provider::ProviderId;
//! use secrecy::SecretString;
//!
//! let config = ProviderHttpConfig::new(
//!     ProviderId::new("anthropic"),
//!     "https://api.anthropic.com/v1",
//! )
//! .with_api_key(SecretString::new("sk-ant-...".into()));
//!
//! let chat = AnthropicChatAdapter::new(config);
//! ```

pub mod chat;
pub mod convert;
pub mod types;

pub use chat::AnthropicChatAdapter;

/// Default Anthropic API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

/// Anthropic API version header value.
pub const API_VERSION: &str = "2023-06-01";
