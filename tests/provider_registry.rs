//! Integration tests for provider registration and dispatch.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use agents::prelude::*;
use async_trait::async_trait;

struct EchoChatProvider {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ChatProvider for EchoChatProvider {
    fn id(&self) -> ProviderId {
        ProviderId::new("echo")
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::chat()
    }

    async fn complete(&self, request: ChatRequest) -> ProviderResult<ChatResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);

        Ok(ChatResponse {
            provider: self.id(),
            model: request.model,
            message: Message::assistant_text("ok"),
            finish_reason: FinishReason::Stop,
            usage: Some(TokenUsage::new(1, 1)),
            raw: None,
        })
    }
}

#[tokio::test]
async fn complete_routes_request_to_registered_chat_provider() {
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = EchoChatProvider {
        calls: Arc::clone(&calls),
    };
    let mut registry = ProviderRegistry::new();

    let replaced = registry.register_chat(provider);
    assert!(replaced.is_none());

    let request = ChatRequest::new(ModelName::new("test-model")).with_user_text("hello");
    let result = registry.complete(&ProviderId::new("echo"), request).await;
    let response = match result {
        Ok(response) => response,
        Err(error) => panic!("expected successful provider response, got {error}"),
    };

    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn complete_returns_unsupported_when_provider_is_missing() {
    let registry = ProviderRegistry::new();
    let request = ChatRequest::new(ModelName::new("test-model")).with_user_text("hello");

    let result = registry.complete(&ProviderId::new("missing"), request).await;
    let error = match result {
        Ok(response) => panic!("expected missing provider error, got {response:?}"),
        Err(error) => error,
    };

    assert!(matches!(error, ProviderError::Unsupported { .. }));
    assert!(!error.is_retryable());
}
