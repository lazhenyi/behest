//! Convenient imports for application and integration code.

pub use crate::error::{ProviderError, Result};
pub use crate::provider::{
    ChatProvider, ChatRequest, ChatResponse, ChatStream, ChatStreamEvent, ContentPart, Embedding,
    EmbeddingInput, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, FinishReason, Message,
    ModelName, ProviderCapabilities, ProviderHttpConfig, ProviderId, ProviderRegistry,
    ProviderResult, ResponseFormat, TokenUsage, ToolCall, ToolChoice, ToolSpec,
};
