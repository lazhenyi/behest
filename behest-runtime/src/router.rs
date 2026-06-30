//! Model router for runtime.
//!
//! Wraps `ProviderRegistry` with capability checking, retry logic,
//! fallback strategies, and usage aggregation.

use std::sync::Arc;
use std::time::Duration;

use tracing::{debug, warn};

use behest_provider::{
    ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, ProviderCapabilities,
    ProviderId, ProviderRegistry,
};

use super::error::{RuntimeError, RuntimeResult};
use super::policy::RuntimePolicy;

/// Routes model requests (chat and embedding) across providers with
/// capability checking, exponential-backoff retry, and fallback chains.
pub struct ModelRouter {
    registry: Arc<ProviderRegistry>,
    policy: RuntimePolicy,
}

impl ModelRouter {
    /// Creates a new model router.
    #[must_use]
    pub fn new(registry: Arc<ProviderRegistry>, policy: RuntimePolicy) -> Self {
        Self { registry, policy }
    }

    /// Returns the provider registry.
    #[must_use]
    pub fn registry(&self) -> &ProviderRegistry {
        &self.registry
    }

    /// Returns the runtime policy.
    #[must_use]
    pub fn policy(&self) -> &RuntimePolicy {
        &self.policy
    }

    /// Routes a chat request to a provider with capability checking and retry.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` if provider not found, lacks capabilities, or all retries fail.
    #[allow(clippy::too_many_lines)]
    pub async fn route_chat(
        &self,
        provider_id: &ProviderId,
        request: ChatRequest,
        required_capabilities: Option<&ProviderCapabilities>,
    ) -> RuntimeResult<ChatResponse> {
        let provider = self
            .registry
            .chat(provider_id)
            .ok_or_else(|| RuntimeError::ProviderNotFound(provider_id.to_string()))?;

        if let Some(required) = required_capabilities {
            let caps = provider.capabilities();
            if !Self::supports_capabilities(&caps, required) {
                return Err(RuntimeError::ProviderNotFound(format!(
                    "provider {provider_id} lacks required capabilities",
                )));
            }
        }

        let mut last_error = None;
        let max_attempts = if self.policy.retry_on_provider_error {
            self.policy.max_retries + 1
        } else {
            1
        };

        for attempt in 1..=max_attempts {
            match provider.complete(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if !e.is_retryable() || attempt == max_attempts {
                        return Err(RuntimeError::from(e));
                    }

                    #[allow(clippy::cast_possible_truncation)]
                    let backoff = Duration::from_millis(100 * 2u64.pow(attempt as u32 - 1));
                    warn!(
                        attempt,
                        max_attempts,
                        ?backoff,
                        error = %e,
                        "provider call failed, retrying"
                    );
                    tokio::time::sleep(backoff).await;
                    last_error = Some(e);
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| behest_core::error::ProviderError::Timeout {
                provider: provider_id.clone(),
            })
            .into())
    }

    /// Routes a chat request across multiple providers with fallback ordering.
    ///
    /// Each provider is tried in order via [`Self::route_chat`] (which itself
    /// applies per-provider retry logic). The first successful response is
    /// returned. When all providers fail, the last error is propagated.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] if every provider in the chain fails.
    pub async fn route_chat_with_fallback(
        &self,
        provider_ids: &[ProviderId],
        request: ChatRequest,
        required_capabilities: Option<&ProviderCapabilities>,
    ) -> RuntimeResult<ChatResponse> {
        let mut last_error = None;

        for provider_id in provider_ids {
            match self
                .route_chat(provider_id, request.clone(), required_capabilities)
                .await
            {
                Ok(response) => return Ok(response),
                Err(e) => {
                    debug!(provider = %provider_id, error = %e, "provider failed, trying fallback");
                    last_error = Some(e);
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| RuntimeError::ProviderNotFound("no providers available".to_owned())))
    }

    /// Routes an embedding request with retry logic.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` if provider not found or all retries fail.
    pub async fn route_embedding(
        &self,
        provider_id: &ProviderId,
        request: EmbeddingRequest,
    ) -> RuntimeResult<EmbeddingResponse> {
        let provider = self
            .registry
            .embedding(provider_id)
            .ok_or_else(|| RuntimeError::ProviderNotFound(provider_id.to_string()))?;

        let mut last_error = None;
        let max_attempts = if self.policy.retry_on_provider_error {
            self.policy.max_retries + 1
        } else {
            1
        };

        for attempt in 1..=max_attempts {
            match provider.embed(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if !e.is_retryable() || attempt == max_attempts {
                        return Err(RuntimeError::from(e));
                    }

                    #[allow(clippy::cast_possible_truncation)]
                    let backoff = Duration::from_millis(100 * 2u64.pow(attempt as u32 - 1));
                    warn!(
                        attempt,
                        max_attempts,
                        ?backoff,
                        error = %e,
                        "embedding provider failed, retrying"
                    );
                    tokio::time::sleep(backoff).await;
                    last_error = Some(e);
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| behest_core::error::ProviderError::Timeout {
                provider: provider_id.clone(),
            })
            .into())
    }

    /// Checks if provider capabilities support all required capabilities.
    fn supports_capabilities(
        available: &ProviderCapabilities,
        required: &ProviderCapabilities,
    ) -> bool {
        (!required.chat || available.chat)
            && (!required.chat_stream || available.chat_stream)
            && (!required.tool_calling || available.tool_calling)
            && (!required.parallel_tool_calls || available.parallel_tool_calls)
            && (!required.json_schema_output || available.json_schema_output)
            && (!required.vision || available.vision)
            && (!required.embeddings || available.embeddings)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use behest_core::error::ProviderError;
    use behest_provider::{ChatProvider, FinishReason, Message, ModelName, ProviderResult};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockChatProvider {
        id: ProviderId,
        fail_count: Arc<AtomicUsize>,
        caps: ProviderCapabilities,
    }

    impl MockChatProvider {
        fn new(id: &str, fail_times: usize) -> Self {
            Self {
                id: ProviderId::new(id),
                fail_count: Arc::new(AtomicUsize::new(fail_times)),
                caps: ProviderCapabilities::chat(),
            }
        }

        fn with_capabilities(id: &str, caps: ProviderCapabilities) -> Self {
            Self {
                id: ProviderId::new(id),
                fail_count: Arc::new(AtomicUsize::new(0)),
                caps,
            }
        }
    }

    #[async_trait]
    impl ChatProvider for MockChatProvider {
        fn id(&self) -> ProviderId {
            self.id.clone()
        }

        fn capabilities(&self) -> ProviderCapabilities {
            self.caps.clone()
        }

        async fn complete(&self, _request: ChatRequest) -> ProviderResult<ChatResponse> {
            let remaining = self.fail_count.fetch_sub(1, Ordering::SeqCst);
            if remaining > 0 {
                return Err(ProviderError::Timeout {
                    provider: self.id.clone(),
                });
            }

            Ok(ChatResponse {
                provider: self.id.clone(),
                model: ModelName::new("test"),
                message: Message::assistant_text("ok"),
                finish_reason: FinishReason::Stop,
                usage: None,
                raw: None,
            })
        }
    }

    #[tokio::test]
    async fn route_chat_should_succeed_on_first_try() {
        let mut registry = ProviderRegistry::new();
        registry.register_chat(MockChatProvider::new("test", 0));

        let router = ModelRouter::new(Arc::new(registry), RuntimePolicy::new());
        let request = ChatRequest::new(ModelName::new("test"));

        let result = router
            .route_chat(&ProviderId::new("test"), request, None)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn route_chat_should_retry_on_retryable_error() {
        let mut registry = ProviderRegistry::new();
        registry.register_chat(MockChatProvider::new("test", 2));

        let policy = RuntimePolicy::new().with_max_retries(3);
        let router = ModelRouter::new(Arc::new(registry), policy);
        let request = ChatRequest::new(ModelName::new("test"));

        let result = router
            .route_chat(&ProviderId::new("test"), request, None)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn route_chat_should_fail_after_max_retries() {
        let mut registry = ProviderRegistry::new();
        registry.register_chat(MockChatProvider::new("test", 10));

        let policy = RuntimePolicy::new().with_max_retries(2);
        let router = ModelRouter::new(Arc::new(registry), policy);
        let request = ChatRequest::new(ModelName::new("test"));

        let result = router
            .route_chat(&ProviderId::new("test"), request, None)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn route_chat_should_check_capabilities() {
        let mut registry = ProviderRegistry::new();
        registry.register_chat(MockChatProvider::with_capabilities(
            "test",
            ProviderCapabilities::chat(),
        ));

        let router = ModelRouter::new(Arc::new(registry), RuntimePolicy::new());
        let request = ChatRequest::new(ModelName::new("test"));

        let required = ProviderCapabilities {
            chat_stream: true,
            ..ProviderCapabilities::chat()
        };

        let result = router
            .route_chat(&ProviderId::new("test"), request, Some(&required))
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RuntimeError::ProviderNotFound(_)
        ));
    }

    #[tokio::test]
    async fn route_chat_with_fallback_should_try_alternatives() {
        let mut registry = ProviderRegistry::new();
        registry.register_chat(MockChatProvider::new("primary", 10));
        registry.register_chat(MockChatProvider::new("fallback", 0));

        let policy = RuntimePolicy::new().with_max_retries(0);
        let router = ModelRouter::new(Arc::new(registry), policy);
        let request = ChatRequest::new(ModelName::new("test"));

        let providers = vec![ProviderId::new("primary"), ProviderId::new("fallback")];
        let result = router
            .route_chat_with_fallback(&providers, request, None)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn route_chat_should_return_error_for_unknown_provider() {
        let registry = ProviderRegistry::new();
        let router = ModelRouter::new(Arc::new(registry), RuntimePolicy::new());
        let request = ChatRequest::new(ModelName::new("test"));

        let result = router
            .route_chat(&ProviderId::new("unknown"), request, None)
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RuntimeError::ProviderNotFound(_)
        ));
    }
}
