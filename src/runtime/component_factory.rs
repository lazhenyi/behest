//! Adapters that wrap existing runtime traits as [`Component`]s.
//!
//! `behest` has many long-lived trait abstractions
//! (`ChatProvider`, `SessionStore`, etc.) that predate the
//! [`Component`] trait. To make them composable with the
//! [`ComponentRegistry`](super::registry::ComponentRegistry) without
//! forcing every existing implementation to provide a `Component::init`
//! method, this module provides ready-made wrapper factories that
//! take an already-constructed `Arc<T>` and expose it as a registered
//! component.
//!
//! These wrappers are the canonical M3 deliverable: the existing trait
//! surface stays unchanged, while the runtime becomes composable on
//! top.
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use behest::runtime::component_factory::ChatProviderComponent;
//! use behest::runtime::registry::ComponentRegistry;
//!
//! # async fn build(registry: &ComponentRegistry) {
//! // Caller-supplied provider, pre-constructed:
//! let provider: Arc<dyn behest::provider::ChatProvider> = todo!();
//! let factory = ChatProviderComponent::new("primary", provider);
//! // ... register with ComponentRegistry via register_factory().
//! # }
//! ```

#![allow(clippy::pedantic)]

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::future::BoxFuture;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::provider::{ChatProvider, EmbeddingProvider};
use crate::runtime::component::{AnyComponent, AnyComponentError, Component, ComponentContext};
use crate::runtime::extension::ExtensionError;
use crate::runtime::registry::{ComponentDescriptor, ComponentFactory, RegistryError};

/// Configuration for a chat provider component wrapper. Currently
/// unused; the wrapper takes a pre-constructed provider. Exists to
/// satisfy the [`Component::Config`] bound so the wrapper can be
/// registered in a [`ComponentRegistry`](super::registry::ComponentRegistry).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct EmptyConfig;

/// Error type for wrapper components: lifecycle errors only, since
/// the wrapped provider was constructed externally.
#[derive(Debug, thiserror::Error)]
#[error("provider wrapper error: {0}")]
pub struct WrapperError(pub String);

impl WrapperError {
    /// Construct a wrapper error from a displayable value.
    #[must_use]
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

/// A [`Component`] that wraps an existing [`ChatProvider`].
pub struct ChatProviderComponent {
    name: String,
    provider: Arc<dyn ChatProvider>,
}

impl ChatProviderComponent {
    /// Construct a wrapper around a pre-built chat provider.
    #[must_use]
    pub fn new(name: impl Into<String>, provider: Arc<dyn ChatProvider>) -> Self {
        Self {
            name: name.into(),
            provider,
        }
    }

    /// Borrow the inner provider.
    #[must_use]
    pub fn provider(&self) -> &Arc<dyn ChatProvider> {
        &self.provider
    }
}

impl fmt::Debug for ChatProviderComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatProviderComponent")
            .field("name", &self.name)
            .field("provider_id", &self.provider.id())
            .finish()
    }
}

#[async_trait]
impl Component for ChatProviderComponent {
    const NAME: &'static str = "ChatProviderComponent";
    type Config = EmptyConfig;
    type Error = WrapperError;

    async fn init(_cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
        Err(WrapperError::new(
            "wrapper components must be constructed via new()",
        ))
    }

    async fn start(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn stop(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn health(&self) -> crate::health::HealthStatus {
        crate::health::HealthStatus::healthy()
    }
}

/// A [`Component`] that wraps an existing [`EmbeddingProvider`].
pub struct EmbeddingProviderComponent {
    name: String,
    provider: Arc<dyn EmbeddingProvider>,
}

impl EmbeddingProviderComponent {
    /// Construct a wrapper around a pre-built embedding provider.
    #[must_use]
    pub fn new(name: impl Into<String>, provider: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            name: name.into(),
            provider,
        }
    }

    /// Borrow the inner provider.
    #[must_use]
    pub fn provider(&self) -> &Arc<dyn EmbeddingProvider> {
        &self.provider
    }
}

impl fmt::Debug for EmbeddingProviderComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmbeddingProviderComponent")
            .field("name", &self.name)
            .field("provider_id", &self.provider.id())
            .finish()
    }
}

#[async_trait]
impl Component for EmbeddingProviderComponent {
    const NAME: &'static str = "EmbeddingProviderComponent";
    type Config = EmptyConfig;
    type Error = WrapperError;

    async fn init(_cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
        Err(WrapperError::new(
            "wrapper components must be constructed via new()",
        ))
    }

    async fn start(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn stop(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// [`AnyComponent`] adapter for [`ChatProviderComponent`].
pub struct ChatProviderAny {
    inner: Arc<ChatProviderComponent>,
}

impl ChatProviderAny {
    /// Wrap a [`ChatProviderComponent`] as a type-erased [`AnyComponent`].
    #[must_use]
    pub fn new(inner: Arc<ChatProviderComponent>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl AnyComponent for ChatProviderAny {
    fn name(&self) -> &'static str {
        ChatProviderComponent::NAME
    }

    fn as_any_arc(&self) -> Arc<dyn Any + Send + Sync> {
        self.inner.clone()
    }

    fn start(&self) -> BoxFuture<'_, Result<(), AnyComponentError>> {
        let name = self.inner.name.clone();
        let inner = self.inner.clone();
        Box::pin(async move {
            inner
                .start()
                .await
                .map_err(|e| AnyComponentError::Component {
                    name,
                    message: e.to_string(),
                })
        })
    }

    fn stop(&self) -> BoxFuture<'_, Result<(), AnyComponentError>> {
        let name = self.inner.name.clone();
        let inner = self.inner.clone();
        Box::pin(async move {
            inner
                .stop()
                .await
                .map_err(|e| AnyComponentError::Component {
                    name,
                    message: e.to_string(),
                })
        })
    }

    fn health(&self) -> BoxFuture<'_, crate::health::HealthStatus> {
        let inner = self.inner.clone();
        Box::pin(async move { inner.health().await })
    }
}

/// [`AnyComponent`] adapter for [`EmbeddingProviderComponent`].
pub struct EmbeddingProviderAny {
    inner: Arc<EmbeddingProviderComponent>,
}

impl EmbeddingProviderAny {
    /// Wrap an [`EmbeddingProviderComponent`] as a type-erased [`AnyComponent`].
    #[must_use]
    pub fn new(inner: Arc<EmbeddingProviderComponent>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl AnyComponent for EmbeddingProviderAny {
    fn name(&self) -> &'static str {
        EmbeddingProviderComponent::NAME
    }

    fn as_any_arc(&self) -> Arc<dyn Any + Send + Sync> {
        self.inner.clone()
    }

    fn start(&self) -> BoxFuture<'_, Result<(), AnyComponentError>> {
        let name = self.inner.name.clone();
        let inner = self.inner.clone();
        Box::pin(async move {
            inner
                .start()
                .await
                .map_err(|e| AnyComponentError::Component {
                    name,
                    message: e.to_string(),
                })
        })
    }

    fn stop(&self) -> BoxFuture<'_, Result<(), AnyComponentError>> {
        let name = self.inner.name.clone();
        let inner = self.inner.clone();
        Box::pin(async move {
            inner
                .stop()
                .await
                .map_err(|e| AnyComponentError::Component {
                    name,
                    message: e.to_string(),
                })
        })
    }

    fn health(&self) -> BoxFuture<'_, crate::health::HealthStatus> {
        let inner = self.inner.clone();
        Box::pin(async move { inner.health().await })
    }
}

/// Factory for [`ChatProviderComponent`] that takes a pre-built
/// `Arc<dyn ChatProvider>` and registers it under the given name.
pub struct ChatProviderFactory {
    descriptor: ComponentDescriptor,
    provider: Arc<dyn ChatProvider>,
}

impl ChatProviderFactory {
    /// Construct a factory. The `name` is the user-assigned instance
    /// name in the registry.
    #[must_use]
    pub fn new(name: impl Into<String>, provider: Arc<dyn ChatProvider>) -> Self {
        let name = name.into();
        let descriptor = ComponentDescriptor {
            name: name.clone(),
            depends_on: Vec::new(),
            config: serde_json::json!({}),
        };
        Self {
            descriptor,
            provider,
        }
    }
}

#[async_trait]
impl ComponentFactory for ChatProviderFactory {
    fn name(&self) -> &str {
        &self.descriptor.name
    }

    fn kind(&self) -> &'static str {
        ChatProviderComponent::NAME
    }

    fn depends_on(&self) -> Vec<String> {
        self.descriptor.depends_on.clone()
    }

    async fn build(
        self: Box<Self>,
        _config: serde_json::Value,
        _ctx: &ComponentContext,
    ) -> Result<Box<dyn AnyComponent>, RegistryError> {
        let inner = Arc::new(ChatProviderComponent::new(
            self.descriptor.name.clone(),
            self.provider,
        ));
        Ok(Box::new(ChatProviderAny::new(inner)))
    }
}

/// Factory for [`EmbeddingProviderComponent`].
pub struct EmbeddingProviderFactory {
    descriptor: ComponentDescriptor,
    provider: Arc<dyn EmbeddingProvider>,
}

impl EmbeddingProviderFactory {
    /// Construct a factory.
    #[must_use]
    pub fn new(name: impl Into<String>, provider: Arc<dyn EmbeddingProvider>) -> Self {
        let name = name.into();
        let descriptor = ComponentDescriptor {
            name: name.clone(),
            depends_on: Vec::new(),
            config: serde_json::json!({}),
        };
        Self {
            descriptor,
            provider,
        }
    }
}

#[async_trait]
impl ComponentFactory for EmbeddingProviderFactory {
    fn name(&self) -> &str {
        &self.descriptor.name
    }

    fn kind(&self) -> &'static str {
        EmbeddingProviderComponent::NAME
    }

    fn depends_on(&self) -> Vec<String> {
        self.descriptor.depends_on.clone()
    }

    async fn build(
        self: Box<Self>,
        _config: serde_json::Value,
        _ctx: &ComponentContext,
    ) -> Result<Box<dyn AnyComponent>, RegistryError> {
        let inner = Arc::new(EmbeddingProviderComponent::new(
            self.descriptor.name.clone(),
            self.provider,
        ));
        Ok(Box::new(EmbeddingProviderAny::new(inner)))
    }
}

/// Convenience: re-export a name-keyed error so callers don't have to
/// import [`ExtensionError`] separately.
pub type FactoryError = ExtensionError;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ProviderError;
    use crate::health::HealthStatus;
    use crate::provider::{
        ChatRequest, ChatResponse, ChatStream, EmbeddingRequest, EmbeddingResponse, FinishReason,
        Message, ProviderCapabilities, ProviderId, ProviderResult, TokenUsage,
    };
    use crate::runtime::lifecycle::ShutdownToken;
    use crate::runtime::registry::ComponentRegistry;
    use async_trait::async_trait;

    struct StubChat;
    #[async_trait]
    impl ChatProvider for StubChat {
        fn id(&self) -> ProviderId {
            ProviderId::new("stub")
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }
        async fn complete(&self, _r: ChatRequest) -> ProviderResult<ChatResponse> {
            Ok(ChatResponse {
                provider: self.id(),
                model: crate::provider::ModelName::new("stub-model"),
                message: Message::user_text(""),
                finish_reason: FinishReason::Stop,
                usage: Some(TokenUsage::new(0, 0)),
                raw: None,
            })
        }
        async fn stream(&self, _r: ChatRequest) -> ProviderResult<ChatStream> {
            Err(ProviderError::Unsupported {
                provider: self.id(),
                feature: "stream".into(),
            })
        }
    }

    struct StubEmbedding;
    #[async_trait]
    impl EmbeddingProvider for StubEmbedding {
        fn id(&self) -> ProviderId {
            ProviderId::new("stub-emb")
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }
        async fn embed(&self, _r: EmbeddingRequest) -> ProviderResult<EmbeddingResponse> {
            Ok(EmbeddingResponse {
                provider: self.id(),
                model: crate::provider::ModelName::new("stub-emb-model"),
                embeddings: Vec::new(),
                usage: None,
                raw: None,
            })
        }
    }

    #[tokio::test]
    async fn chat_provider_wrapper_registers_with_registry() {
        let registry = ComponentRegistry::new();
        let factory = ChatProviderFactory::new("primary", Arc::new(StubChat));
        registry
            .register_factory(
                ComponentDescriptor {
                    name: "primary".into(),
                    depends_on: Vec::new(),
                    config: serde_json::json!({}),
                },
                Box::new(factory),
            )
            .unwrap_or_else(|e| panic!("{e}"));
        registry.init_all().await.unwrap_or_else(|e| panic!("{e}"));
        registry.start_all().await.unwrap_or_else(|e| panic!("{e}"));
        let c = registry
            .get::<ChatProviderComponent>("primary")
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(c.name, "primary");
        assert_eq!(c.provider.id().as_str(), "stub");
    }

    #[tokio::test]
    async fn embedding_provider_wrapper_registers_with_registry() {
        let registry = ComponentRegistry::new();
        let factory = EmbeddingProviderFactory::new("primary", Arc::new(StubEmbedding));
        registry
            .register_factory(
                ComponentDescriptor {
                    name: "primary".into(),
                    depends_on: Vec::new(),
                    config: serde_json::json!({}),
                },
                Box::new(factory),
            )
            .unwrap_or_else(|e| panic!("{e}"));
        registry.init_all().await.unwrap_or_else(|e| panic!("{e}"));
        let c = registry
            .get::<EmbeddingProviderComponent>("primary")
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(c.name, "primary");
    }

    #[tokio::test]
    async fn chat_provider_init_returns_wrapper_error() {
        let shutdown = ShutdownToken::new();
        let ctx = ComponentContext::new(shutdown);
        let cfg = EmptyConfig;
        let result = ChatProviderComponent::init(&cfg, &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn default_lifecycle_is_noop_for_provider_wrappers() {
        let c = ChatProviderComponent::new("x", Arc::new(StubChat));
        let _ = c.start().await;
        let _ = c.stop().await;
        let h = c.health().await;
        assert_eq!(h, HealthStatus::healthy());
    }
}
