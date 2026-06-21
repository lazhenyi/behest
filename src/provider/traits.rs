//! Provider traits implemented by model adapters.

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::Stream;

use crate::error::ProviderError;
use crate::provider::{
    ChatRequest, ChatResponse, ChatStreamEvent, EmbeddingRequest, EmbeddingResponse,
    ProviderCapabilities, ProviderId,
};

/// Result type returned by provider implementations.
pub type ProviderResult<T> = std::result::Result<T, ProviderError>;

/// Boxed stream returned by streaming chat providers.
pub type ChatStream = Pin<Box<dyn Stream<Item = ProviderResult<ChatStreamEvent>> + Send + 'static>>;

/// Provider capable of serving chat completion requests.
#[async_trait]
pub trait ChatProvider: Send + Sync {
    /// Returns the provider identifier.
    fn id(&self) -> ProviderId;

    /// Returns feature flags for this provider.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Completes a chat request.
    async fn complete(&self, request: ChatRequest) -> ProviderResult<ChatResponse>;

    /// Streams a chat request.
    async fn stream(&self, request: ChatRequest) -> ProviderResult<ChatStream> {
        let _ = request;

        Err(ProviderError::Unsupported {
            provider: self.id(),
            feature: "chat_stream".to_owned(),
        })
    }
}

/// Provider capable of serving embedding requests.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Returns the provider identifier.
    fn id(&self) -> ProviderId;

    /// Returns feature flags for this provider.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Embeds request inputs.
    async fn embed(&self, request: EmbeddingRequest) -> ProviderResult<EmbeddingResponse>;
}
