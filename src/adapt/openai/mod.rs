//! OpenAI-compatible provider adapter.
//!
//! Supports chat completions (with streaming), tool calling, embeddings,
//! and structured output. Works with OpenAI, Azure OpenAI, and any
//! OpenAI-compatible API endpoint.
//!
//! # Examples
//!
//! ```no_run
//! use agents::adapt::openai::{OpenAiChatAdapter, OpenAiEmbeddingAdapter};
//! use agents::provider::ProviderHttpConfig;
//! use agents::provider::ProviderId;
//! use secrecy::SecretString;
//!
//! let config = ProviderHttpConfig::new(
//!     ProviderId::new("openai"),
//!     "https://api.openai.com/v1",
//! )
//! .with_api_key(SecretString::new("sk-...".into()));
//!
//! let chat = OpenAiChatAdapter::new(config.clone());
//! let embeddings = OpenAiEmbeddingAdapter::new(config);
//! ```

pub mod chat;
pub mod convert;
pub mod embed;
pub mod types;

pub use chat::OpenAiChatAdapter;
pub use embed::OpenAiEmbeddingAdapter;

/// Default OpenAI API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
