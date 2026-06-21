//! Runtime registry for provider implementations.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::ProviderError;
use crate::provider::{
    ChatProvider, ChatRequest, ChatResponse, EmbeddingProvider, EmbeddingRequest,
    EmbeddingResponse, ProviderId, ProviderResult,
};

/// In-memory registry for chat and embedding providers.
#[derive(Clone, Default)]
pub struct ProviderRegistry {
    chat: HashMap<ProviderId, Arc<dyn ChatProvider>>,
    embeddings: HashMap<ProviderId, Arc<dyn EmbeddingProvider>>,
}

impl ProviderRegistry {
    /// Creates an empty provider registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a chat provider and returns the replaced provider, if any.
    pub fn register_chat<P>(&mut self, provider: P) -> Option<Arc<dyn ChatProvider>>
    where
        P: ChatProvider + 'static,
    {
        let id = provider.id();
        self.chat.insert(id, Arc::new(provider))
    }

    /// Registers an already shared chat provider.
    pub fn register_chat_arc(
        &mut self,
        provider: Arc<dyn ChatProvider>,
    ) -> Option<Arc<dyn ChatProvider>> {
        self.chat.insert(provider.id(), provider)
    }

    /// Registers an embedding provider and returns the replaced provider, if any.
    pub fn register_embedding<P>(&mut self, provider: P) -> Option<Arc<dyn EmbeddingProvider>>
    where
        P: EmbeddingProvider + 'static,
    {
        let id = provider.id();
        self.embeddings.insert(id, Arc::new(provider))
    }

    /// Registers an already shared embedding provider.
    pub fn register_embedding_arc(
        &mut self,
        provider: Arc<dyn EmbeddingProvider>,
    ) -> Option<Arc<dyn EmbeddingProvider>> {
        self.embeddings.insert(provider.id(), provider)
    }

    /// Returns a registered chat provider by id.
    #[must_use]
    pub fn chat(&self, id: &ProviderId) -> Option<Arc<dyn ChatProvider>> {
        self.chat.get(id).map(Arc::clone)
    }

    /// Returns a registered embedding provider by id.
    #[must_use]
    pub fn embedding(&self, id: &ProviderId) -> Option<Arc<dyn EmbeddingProvider>> {
        self.embeddings.get(id).map(Arc::clone)
    }

    /// Returns registered chat provider identifiers.
    pub fn chat_ids(&self) -> impl Iterator<Item = &ProviderId> {
        self.chat.keys()
    }

    /// Returns registered embedding provider identifiers.
    pub fn embedding_ids(&self) -> impl Iterator<Item = &ProviderId> {
        self.embeddings.keys()
    }

    /// Routes a chat request to a registered provider.
    pub async fn complete(
        &self,
        provider_id: &ProviderId,
        request: ChatRequest,
    ) -> ProviderResult<ChatResponse> {
        let provider = self
            .chat(provider_id)
            .ok_or_else(|| unsupported(provider_id, "chat"))?;

        provider.complete(request).await
    }

    /// Routes an embedding request to a registered provider.
    pub async fn embed(
        &self,
        provider_id: &ProviderId,
        request: EmbeddingRequest,
    ) -> ProviderResult<EmbeddingResponse> {
        let provider = self
            .embedding(provider_id)
            .ok_or_else(|| unsupported(provider_id, "embedding"))?;

        provider.embed(request).await
    }
}

fn unsupported(provider_id: &ProviderId, feature: &str) -> ProviderError {
    ProviderError::Unsupported {
        provider: provider_id.clone(),
        feature: feature.to_owned(),
    }
}
