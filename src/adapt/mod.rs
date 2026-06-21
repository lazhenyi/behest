//! Concrete provider adapters for model vendors.
//!
//! Each adapter is gated behind a Cargo feature flag and implements
//! [`ChatProvider`](crate::provider::ChatProvider) and/or
//! [`EmbeddingProvider`](crate::provider::EmbeddingProvider).
//!
//! # Available adapters
//!
//! | Feature | Adapter | Chat | Stream | Embedding | Tools |
//! |---------|---------|------|--------|-----------|-------|
//! | `openai` | [`openai::OpenAiChatAdapter`] | ✅ | ✅ | ✅ | ✅ |
//! | `anthropic` | [`anthropic::AnthropicChatAdapter`] | ✅ | ✅ | ❌ | ✅ |

pub(crate) mod http;
pub(crate) mod sse;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;
