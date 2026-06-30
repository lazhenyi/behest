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
///
/// Wraps [`ProviderError`], covering network failures, authentication errors,
/// content filter rejections, and unsupported feature requests.
pub type ProviderResult<T> = std::result::Result<T, ProviderError>;

/// Boxed, pinned, asynchronous stream of chat events.
///
/// Each item is a [`ProviderResult`] wrapping a [`ChatStreamEvent`]. The stream
/// ends with a `Finished` event or a terminal error.
pub type ChatStream = Pin<Box<dyn Stream<Item = ProviderResult<ChatStreamEvent>> + Send + 'static>>;

/// Provider capable of serving chat completion requests.
#[async_trait]
pub trait ChatProvider: Send + Sync {
    /// Returns the provider identifier.
    fn id(&self) -> ProviderId;

    /// Returns feature flags for this provider.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Completes a chat request.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] on network failure, invalid model configuration,
    /// authentication rejection, or content filter intervention.
    async fn complete(&self, request: ChatRequest) -> ProviderResult<ChatResponse>;

    /// Streams a chat request.
    ///
    /// The default implementation returns [`ProviderError::Unsupported`].
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] when the stream cannot be established —
    /// network errors, invalid model configuration, or authentication failures.
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
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] on network failure, invalid model configuration,
    /// or authentication rejection.
    async fn embed(&self, request: EmbeddingRequest) -> ProviderResult<EmbeddingResponse>;
}
