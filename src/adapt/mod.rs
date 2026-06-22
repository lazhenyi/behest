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

#[cfg(any(feature = "openai", feature = "anthropic"))]
pub(crate) mod http;
#[cfg(any(feature = "openai", feature = "anthropic"))]
pub(crate) mod sse;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;
