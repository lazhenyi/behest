//! Provider-neutral contracts for chat, tools, streaming, and embeddings.
//!
//! Implement [`ChatProvider`] or [`EmbeddingProvider`] to adapt a model vendor,
//! then add the implementation to [`ProviderRegistry`] for runtime dispatch.

pub mod capabilities;
pub mod config;
pub mod embedding;
pub mod events;
pub mod id;
pub mod message;
pub mod registry;
pub mod tool;
pub mod traits;

pub use capabilities::ProviderCapabilities;
pub use config::ProviderHttpConfig;
pub use embedding::{Embedding, EmbeddingInput, EmbeddingRequest, EmbeddingResponse};
pub use events::ChatStreamEvent;
pub use id::{ModelName, ProviderId};
pub use message::{
    ChatRequest, ChatResponse, ContentPart, FinishReason, Message, ResponseFormat, TokenUsage,
};
pub use registry::ProviderRegistry;
pub use tool::{ToolCall, ToolChoice, ToolSpec};
pub use traits::{ChatProvider, ChatStream, EmbeddingProvider, ProviderResult};
