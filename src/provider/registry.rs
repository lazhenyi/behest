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
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::Unsupported`] when no chat provider is registered
    /// for the given identifier, or any error from the underlying provider.
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
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::Unsupported`] when no embedding provider is
    /// registered for the given identifier, or any error from the underlying provider.
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::provider::{
        ChatRequest, ChatResponse, ChatStream, EmbeddingRequest, EmbeddingResponse, FinishReason,
        Message, ModelName, ProviderCapabilities,
    };

    struct MockChatProvider {
        id: ProviderId,
    }

    #[async_trait::async_trait]
    impl ChatProvider for MockChatProvider {
        fn id(&self) -> ProviderId {
            self.id.clone()
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::chat()
        }
        async fn complete(&self, _request: ChatRequest) -> ProviderResult<ChatResponse> {
            Ok(ChatResponse {
                provider: self.id.clone(),
                model: ModelName::new("mock-model"),
                message: Message::assistant_text("mock response"),
                finish_reason: FinishReason::Stop,
                usage: None,
                raw: None,
            })
        }
        async fn stream(&self, _request: ChatRequest) -> ProviderResult<ChatStream> {
            Err(ProviderError::Unsupported {
                provider: self.id.clone(),
                feature: "chat_stream".to_owned(),
            })
        }
    }

    struct MockEmbeddingProvider {
        id: ProviderId,
    }

    #[async_trait::async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        fn id(&self) -> ProviderId {
            self.id.clone()
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::embeddings()
        }
        async fn embed(&self, _request: EmbeddingRequest) -> ProviderResult<EmbeddingResponse> {
            Ok(EmbeddingResponse {
                provider: self.id.clone(),
                model: ModelName::new("mock-embed"),
                embeddings: vec![],
                usage: None,
                raw: None,
            })
        }
    }

    #[test]
    fn registry_should_be_empty_when_new() {
        let registry = ProviderRegistry::new();
        assert!(registry.chat_ids().count() == 0);
        assert!(registry.embedding_ids().count() == 0);
    }

    #[test]
    fn registry_should_register_and_retrieve_chat_provider() {
        let mut registry = ProviderRegistry::new();
        let id = ProviderId::new("mock");
        registry.register_chat(MockChatProvider { id: id.clone() });

        assert!(registry.chat(&id).is_some());
        assert!(registry.chat(&ProviderId::new("other")).is_none());
    }

    #[test]
    fn registry_should_register_and_retrieve_embedding_provider() {
        let mut registry = ProviderRegistry::new();
        let id = ProviderId::new("mock-embed");
        registry.register_embedding(MockEmbeddingProvider { id: id.clone() });

        assert!(registry.embedding(&id).is_some());
        assert!(registry.embedding(&ProviderId::new("other")).is_none());
    }

    #[test]
    fn registry_should_replace_existing_chat_provider() {
        let mut registry = ProviderRegistry::new();
        let id = ProviderId::new("mock");
        registry.register_chat(MockChatProvider { id: id.clone() });
        let replaced = registry.register_chat(MockChatProvider { id: id.clone() });

        assert!(replaced.is_some());
        assert_eq!(registry.chat_ids().count(), 1);
    }

    #[tokio::test]
    async fn registry_complete_should_route_to_registered_provider() {
        let mut registry = ProviderRegistry::new();
        let id = ProviderId::new("mock");
        registry.register_chat(MockChatProvider { id: id.clone() });

        let request = ChatRequest::new(ModelName::new("test"));
        let result = registry.complete(&id, request).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().model.as_str(), "mock-model");
    }

    #[tokio::test]
    async fn registry_complete_should_return_unsupported_for_unknown_provider() {
        let registry = ProviderRegistry::new();
        let id = ProviderId::new("nonexistent");
        let request = ChatRequest::new(ModelName::new("test"));

        let result = registry.complete(&id, request).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ProviderError::Unsupported { .. }
        ));
    }

    #[tokio::test]
    async fn registry_embed_should_route_to_registered_provider() {
        let mut registry = ProviderRegistry::new();
        let id = ProviderId::new("mock-embed");
        registry.register_embedding(MockEmbeddingProvider { id: id.clone() });

        let request = EmbeddingRequest::from_text(ModelName::new("test"), "hello");
        let result = registry.embed(&id, request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn registry_embed_should_return_unsupported_for_unknown_provider() {
        let registry = ProviderRegistry::new();
        let id = ProviderId::new("nonexistent");
        let request = EmbeddingRequest::from_text(ModelName::new("test"), "hello");

        let result = registry.embed(&id, request).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ProviderError::Unsupported { .. }
        ));
    }
}
