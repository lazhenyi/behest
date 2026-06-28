#![allow(dead_code)]

//! Self-contained [`Component`] implementations that construct providers,
//! stores, and adapters from JSON configuration.
//!
//! Each wrapper type implements [`Component`] so it can be registered
//! either directly via `TypedFactory`
//! or through a [`FactoryRegistry`]
//! using the convenience `register_*` functions or
//! [`default_factory_registry`].

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};

use super::AnyComponent;
use super::memory::MemoryRunStore;
#[cfg(feature = "anthropic")]
use crate::adapt::anthropic::chat::AnthropicChatAdapter;
#[cfg(feature = "openai")]
use crate::adapt::openai::chat::OpenAiChatAdapter;
#[cfg(feature = "openai")]
use crate::adapt::openai::embed::OpenAiEmbeddingAdapter;
use crate::error::ProviderError;
#[cfg(any(feature = "openai", feature = "anthropic"))]
use crate::provider::ChatProvider;
#[cfg(feature = "openai")]
use crate::provider::EmbeddingProvider;
use crate::provider::config::{DEFAULT_CONNECT_TIMEOUT, DEFAULT_TIMEOUT};
use crate::provider::{ProviderHttpConfig, ProviderId};
use crate::runtime::component::{Component, ComponentContext};
use crate::runtime::context::ContextPipeline;
use crate::runtime::factory_registry::{FactoryError, FactoryRegistry};
use crate::runtime::registry::TypedAnyComponent;
use crate::store::memory::{
    MemoryArtifactStore, MemoryEmbeddingStore, MemoryExecutionStore, MemorySessionStore,
};

// ---------------------------------------------------------------------------
// JSON-serializable config for HTTP-backed providers
// ---------------------------------------------------------------------------

/// JSON-deserializable configuration for HTTP-backed providers like OpenAI
/// and Anthropic. Uses plain `String` for the API key (instead of
/// [`SecretString`]) so that it can be deserialised from JSON/YAML/TOML.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ProviderHttpComponentConfig {
    /// Logical provider identifier.
    pub id: String,
    /// Provider API base URL.
    pub base_url: String,
    /// Optional API key or bearer token.
    pub api_key: Option<String>,
    /// Optional organization or tenant identifier.
    pub organization: Option<String>,
    /// End-to-end request timeout in seconds (default: 60).
    pub timeout_secs: Option<u64>,
    /// TCP connection timeout in seconds (default: 10).
    pub connect_timeout_secs: Option<u64>,
    /// Maximum retry attempts (default: 2).
    pub max_retries: Option<usize>,
}

impl ProviderHttpComponentConfig {
    /// Converts this JSON config into a [`ProviderHttpConfig`], wrapping
    /// the plain-text API key in a [`SecretString`].
    #[must_use]
    pub fn into_provider_http_config(self) -> ProviderHttpConfig {
        let mut cfg = ProviderHttpConfig::new(ProviderId::new(&self.id), self.base_url);
        if let Some(key) = self.api_key {
            cfg = cfg.with_api_key(SecretString::new(key.into()));
        }
        if let Some(org) = self.organization {
            cfg = cfg.with_organization(org);
        }
        let timeout = self
            .timeout_secs
            .map_or(DEFAULT_TIMEOUT, std::time::Duration::from_secs);
        let connect_timeout = self
            .connect_timeout_secs
            .map_or(DEFAULT_CONNECT_TIMEOUT, std::time::Duration::from_secs);
        cfg = cfg.with_timeouts(timeout, connect_timeout);
        cfg = cfg.with_max_retries(self.max_retries.unwrap_or(2));
        cfg
    }
}

// ---------------------------------------------------------------------------
// Unified error type for config-constructed components
// ---------------------------------------------------------------------------

/// Error type for components constructed from configuration.
#[derive(Debug, thiserror::Error)]
pub enum ComponentError {
    /// Provider construction or lifecycle failure.
    #[error("provider error: {0}")]
    Provider(String),
    /// Store construction failure.
    #[error("store error: {0}")]
    Store(String),
    /// Context pipeline construction failure.
    #[error("context error: {0}")]
    Context(String),
}

impl From<ProviderError> for ComponentError {
    fn from(e: ProviderError) -> Self {
        Self::Provider(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Provider components
// ---------------------------------------------------------------------------

/// Component wrapper for [`OpenAiChatAdapter`].
#[cfg(feature = "openai")]
pub struct OpenAiChatComponent {
    inner: Arc<dyn ChatProvider>,
}

#[cfg(feature = "openai")]
#[async_trait]
impl Component for OpenAiChatComponent {
    const NAME: &'static str = "provider.openai.chat";
    type Config = ProviderHttpComponentConfig;
    type Error = ComponentError;

    async fn init(cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
        let http = cfg.clone().into_provider_http_config();
        let adapter = OpenAiChatAdapter::new(http)?;
        Ok(Self {
            inner: Arc::new(adapter),
        })
    }
}

/// Component wrapper for [`AnthropicChatAdapter`].
#[cfg(feature = "anthropic")]
pub struct AnthropicChatComponent {
    inner: Arc<dyn ChatProvider>,
}

#[cfg(feature = "anthropic")]
#[async_trait]
impl Component for AnthropicChatComponent {
    const NAME: &'static str = "provider.anthropic.chat";
    type Config = ProviderHttpComponentConfig;
    type Error = ComponentError;

    async fn init(cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
        let http = cfg.clone().into_provider_http_config();
        let adapter = AnthropicChatAdapter::new(http)?;
        Ok(Self {
            inner: Arc::new(adapter),
        })
    }
}

/// Component wrapper for [`OpenAiEmbeddingAdapter`].
#[cfg(feature = "openai")]
pub struct OpenAiEmbeddingComponent {
    inner: Arc<dyn EmbeddingProvider>,
}

#[cfg(feature = "openai")]
#[async_trait]
impl Component for OpenAiEmbeddingComponent {
    const NAME: &'static str = "provider.openai.embedding";
    type Config = ProviderHttpComponentConfig;
    type Error = ComponentError;

    async fn init(cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
        let http = cfg.clone().into_provider_http_config();
        let adapter = OpenAiEmbeddingAdapter::new(http)?;
        Ok(Self {
            inner: Arc::new(adapter),
        })
    }
}

// ---------------------------------------------------------------------------
// Memory store components
// ---------------------------------------------------------------------------

/// Component wrapper for [`MemorySessionStore`].
pub struct MemorySessionStoreComponent {
    inner: Arc<MemorySessionStore>,
}

#[async_trait]
impl Component for MemorySessionStoreComponent {
    const NAME: &'static str = "store.session.memory";
    type Config = serde_json::Value;
    type Error = ComponentError;

    async fn init(_cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: Arc::new(MemorySessionStore::new()),
        })
    }
}

/// Component wrapper for [`MemoryExecutionStore`].
pub struct MemoryExecutionStoreComponent {
    inner: Arc<MemoryExecutionStore>,
}

#[async_trait]
impl Component for MemoryExecutionStoreComponent {
    const NAME: &'static str = "store.execution.memory";
    type Config = serde_json::Value;
    type Error = ComponentError;

    async fn init(_cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: Arc::new(MemoryExecutionStore::new()),
        })
    }
}

/// Component wrapper for [`MemoryRunStore`].
pub struct MemoryRunStoreComponent {
    inner: Arc<MemoryRunStore>,
}

#[async_trait]
impl Component for MemoryRunStoreComponent {
    const NAME: &'static str = "store.run.memory";
    type Config = serde_json::Value;
    type Error = ComponentError;

    async fn init(_cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: Arc::new(MemoryRunStore::new()),
        })
    }
}

/// Component wrapper for [`MemoryEmbeddingStore`].
pub struct MemoryEmbeddingStoreComponent {
    inner: Arc<MemoryEmbeddingStore>,
}

#[async_trait]
impl Component for MemoryEmbeddingStoreComponent {
    const NAME: &'static str = "store.embedding.memory";
    type Config = serde_json::Value;
    type Error = ComponentError;

    async fn init(_cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: Arc::new(MemoryEmbeddingStore::new()),
        })
    }
}

/// Component wrapper for [`MemoryArtifactStore`].
pub struct MemoryArtifactStoreComponent {
    inner: Arc<MemoryArtifactStore>,
}

#[async_trait]
impl Component for MemoryArtifactStoreComponent {
    const NAME: &'static str = "store.artifact.memory";
    type Config = serde_json::Value;
    type Error = ComponentError;

    async fn init(_cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: Arc::new(MemoryArtifactStore::new()),
        })
    }
}

// ---------------------------------------------------------------------------
// Context pipeline component
// ---------------------------------------------------------------------------

/// JSON configuration for [`ContextPipelineComponent`].
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ContextPipelineConfig {
    /// Maximum number of history messages (default: 50).
    pub max_history_messages: Option<usize>,
    /// Maximum token budget for history (default: 64000).
    pub max_history_tokens: Option<usize>,
    /// Enable compaction filter (default: true).
    pub enable_compaction_filter: Option<bool>,
}

/// Component wrapper for [`ContextPipeline`].
pub struct ContextPipelineComponent {
    inner: Arc<ContextPipeline>,
}

#[async_trait]
impl Component for ContextPipelineComponent {
    const NAME: &'static str = "context.pipeline";
    type Config = ContextPipelineConfig;
    type Error = ComponentError;

    async fn init(cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
        let mut pipeline = ContextPipeline::new();
        if let Some(v) = cfg.max_history_messages {
            pipeline = pipeline.with_max_history(v);
        }
        if let Some(v) = cfg.max_history_tokens {
            pipeline = pipeline.with_max_history_tokens(v);
        }
        if let Some(v) = cfg.enable_compaction_filter {
            pipeline = pipeline.with_compaction_filter(v);
        }
        Ok(Self {
            inner: Arc::new(pipeline),
        })
    }
}

// ---------------------------------------------------------------------------
// Convenience registration helpers
// ---------------------------------------------------------------------------

/// Registers all provider factory invokers into a [`FactoryRegistry`].
///
/// Registered kinds:
/// - `"provider.openai.chat"`
/// - `"provider.anthropic.chat"`
/// - `"provider.openai.embedding"`
#[must_use]
pub fn register_providers(registry: FactoryRegistry) -> FactoryRegistry {
    let registry = {
        #[cfg(feature = "openai")]
        {
            registry
                .register("provider.openai.chat", |cfg, _ctx| {
                    let v: ProviderHttpComponentConfig =
                        serde_json::from_value(cfg).map_err(|e| FactoryError::InvalidConfig {
                            kind: "provider.openai.chat".into(),
                            source: e,
                        })?;
                    let http = v.into_provider_http_config();
                    let adapter = OpenAiChatAdapter::new(http).map_err(|e| {
                        FactoryError::FactoryFailed("provider.openai.chat".into(), e.to_string())
                    })?;
                    let comp = OpenAiChatComponent {
                        inner: Arc::new(adapter),
                    };
                    Ok(Box::new(TypedAnyComponent::new(comp)) as Box<dyn AnyComponent>)
                })
                .register("provider.openai.embedding", |cfg, _ctx| {
                    let v: ProviderHttpComponentConfig =
                        serde_json::from_value(cfg).map_err(|e| FactoryError::InvalidConfig {
                            kind: "provider.openai.embedding".into(),
                            source: e,
                        })?;
                    let http = v.into_provider_http_config();
                    let adapter = OpenAiEmbeddingAdapter::new(http).map_err(|e| {
                        FactoryError::FactoryFailed(
                            "provider.openai.embedding".into(),
                            e.to_string(),
                        )
                    })?;
                    let comp = OpenAiEmbeddingComponent {
                        inner: Arc::new(adapter),
                    };
                    Ok(Box::new(TypedAnyComponent::new(comp)) as Box<dyn AnyComponent>)
                })
        }
        #[cfg(not(feature = "openai"))]
        {
            registry
        }
    };

    #[cfg(feature = "anthropic")]
    {
        registry.register("provider.anthropic.chat", |cfg, _ctx| {
            let v: ProviderHttpComponentConfig =
                serde_json::from_value(cfg).map_err(|e| FactoryError::InvalidConfig {
                    kind: "provider.anthropic.chat".into(),
                    source: e,
                })?;
            let http = v.into_provider_http_config();
            let adapter = AnthropicChatAdapter::new(http).map_err(|e| {
                FactoryError::FactoryFailed("provider.anthropic.chat".into(), e.to_string())
            })?;
            let comp = AnthropicChatComponent {
                inner: Arc::new(adapter),
            };
            Ok(Box::new(TypedAnyComponent::new(comp)) as Box<dyn AnyComponent>)
        })
    }
    #[cfg(not(feature = "anthropic"))]
    {
        registry
    }
}

/// Registers all memory-store factory invokers into a [`FactoryRegistry`].
///
/// Registered kinds:
/// - `"store.session.memory"`
/// - `"store.execution.memory"`
/// - `"store.run.memory"`
/// - `"store.embedding.memory"`
/// - `"store.artifact.memory"`
#[must_use]
pub fn register_memory_stores(registry: FactoryRegistry) -> FactoryRegistry {
    registry
        .register("store.session.memory", |cfg, ctx| {
            let _ = (cfg, &ctx);
            let comp = MemorySessionStoreComponent {
                inner: Arc::new(MemorySessionStore::new()),
            };
            Ok(Box::new(TypedAnyComponent::new(comp)) as Box<dyn AnyComponent>)
        })
        .register("store.execution.memory", |cfg, ctx| {
            let _ = (cfg, &ctx);
            let comp = MemoryExecutionStoreComponent {
                inner: Arc::new(MemoryExecutionStore::new()),
            };
            Ok(Box::new(TypedAnyComponent::new(comp)) as Box<dyn AnyComponent>)
        })
        .register("store.run.memory", |cfg, ctx| {
            let _ = (cfg, &ctx);
            let comp = MemoryRunStoreComponent {
                inner: Arc::new(MemoryRunStore::new()),
            };
            Ok(Box::new(TypedAnyComponent::new(comp)) as Box<dyn AnyComponent>)
        })
        .register("store.embedding.memory", |cfg, ctx| {
            let _ = (cfg, &ctx);
            let comp = MemoryEmbeddingStoreComponent {
                inner: Arc::new(MemoryEmbeddingStore::new()),
            };
            Ok(Box::new(TypedAnyComponent::new(comp)) as Box<dyn AnyComponent>)
        })
        .register("store.artifact.memory", |cfg, ctx| {
            let _ = (cfg, &ctx);
            let comp = MemoryArtifactStoreComponent {
                inner: Arc::new(MemoryArtifactStore::new()),
            };
            Ok(Box::new(TypedAnyComponent::new(comp)) as Box<dyn AnyComponent>)
        })
}

/// Registers the context-pipeline factory invoker into a [`FactoryRegistry`].
///
/// Registered kind:
/// - `"context.pipeline"`
#[must_use]
pub fn register_context_pipeline(registry: FactoryRegistry) -> FactoryRegistry {
    registry.register("context.pipeline", |cfg, _ctx| {
        let v: ContextPipelineConfig =
            serde_json::from_value(cfg).map_err(|e| FactoryError::InvalidConfig {
                kind: "context.pipeline".into(),
                source: e,
            })?;
        let mut pipeline = ContextPipeline::new();
        if let Some(v) = v.max_history_messages {
            pipeline = pipeline.with_max_history(v);
        }
        if let Some(v) = v.max_history_tokens {
            pipeline = pipeline.with_max_history_tokens(v);
        }
        if let Some(v) = v.enable_compaction_filter {
            pipeline = pipeline.with_compaction_filter(v);
        }
        let comp = ContextPipelineComponent {
            inner: Arc::new(pipeline),
        };
        Ok(Box::new(TypedAnyComponent::new(comp)) as Box<dyn AnyComponent>)
    })
}

/// Returns a [`FactoryRegistry`] pre-populated with all built-in component
/// factory invokers:
///
/// - Provider adapters (OpenAI chat, Anthropic chat, OpenAI embedding)
/// - Memory stores (session, execution, run, embedding, artifact)
/// - Context pipeline
#[must_use]
pub fn default_factory_registry() -> FactoryRegistry {
    let reg = FactoryRegistry::new();
    let reg = register_providers(reg);
    let reg = register_memory_stores(reg);
    register_context_pipeline(reg)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;
    use crate::runtime::factory_registry::FactoryError;
    use crate::runtime::lifecycle::ShutdownToken;

    fn ctx() -> ComponentContext {
        ComponentContext::new(ShutdownToken::new())
    }

    #[test]
    fn default_registry_contains_all_kinds() {
        let reg = default_factory_registry();
        let kinds: Vec<&str> = reg.kinds().collect();
        assert!(kinds.contains(&"store.session.memory"));
        assert!(kinds.contains(&"store.execution.memory"));
        assert!(kinds.contains(&"store.run.memory"));
        assert!(kinds.contains(&"store.embedding.memory"));
        assert!(kinds.contains(&"store.artifact.memory"));
        assert!(kinds.contains(&"context.pipeline"));

        let expected_providers: usize = 0
            + if cfg!(feature = "openai") { 2 } else { 0 }
            + if cfg!(feature = "anthropic") { 1 } else { 0 };
        assert_eq!(kinds.len(), 6 + expected_providers);

        #[cfg(feature = "openai")]
        {
            assert!(kinds.contains(&"provider.openai.chat"));
            assert!(kinds.contains(&"provider.openai.embedding"));
        }
        #[cfg(feature = "anthropic")]
        {
            assert!(kinds.contains(&"provider.anthropic.chat"));
        }
    }

    #[test]
    fn default_registry_rejects_unknown_kind() {
        let reg = default_factory_registry();
        let result = reg.invoke("nope", serde_json::json!({}), &ctx());
        assert!(matches!(result, Err(FactoryError::UnknownKind(_))));
    }

    #[test]
    fn memory_session_store_invocation() {
        let reg = default_factory_registry();
        let comp = reg
            .invoke("store.session.memory", serde_json::json!({}), &ctx())
            .expect("invoke should succeed");
        assert_eq!(comp.name(), "store.session.memory");
    }

    #[test]
    fn memory_execution_store_invocation() {
        let reg = default_factory_registry();
        let comp = reg
            .invoke("store.execution.memory", serde_json::json!({}), &ctx())
            .expect("invoke should succeed");
        assert_eq!(comp.name(), "store.execution.memory");
    }

    #[test]
    fn memory_run_store_invocation() {
        let reg = default_factory_registry();
        let comp = reg
            .invoke("store.run.memory", serde_json::json!({}), &ctx())
            .expect("invoke should succeed");
        assert_eq!(comp.name(), "store.run.memory");
    }

    #[test]
    fn memory_embedding_store_invocation() {
        let reg = default_factory_registry();
        let comp = reg
            .invoke("store.embedding.memory", serde_json::json!({}), &ctx())
            .expect("invoke should succeed");
        assert_eq!(comp.name(), "store.embedding.memory");
    }

    #[test]
    fn memory_artifact_store_invocation() {
        let reg = default_factory_registry();
        let comp = reg
            .invoke("store.artifact.memory", serde_json::json!({}), &ctx())
            .expect("invoke should succeed");
        assert_eq!(comp.name(), "store.artifact.memory");
    }

    #[tokio::test]
    async fn context_pipeline_invocation() {
        let reg = default_factory_registry();
        let comp = reg
            .invoke(
                "context.pipeline",
                serde_json::json!({
                    "max_history_messages": 100,
                }),
                &ctx(),
            )
            .expect("invoke should succeed");
        assert_eq!(comp.name(), "context.pipeline");
    }

    #[cfg(feature = "openai")]
    #[test]
    fn provider_openai_chat_invocation_succeeds_without_api_key() {
        // Adapter constructors are lazy — they don't validate credentials
        // at construction time, only at request time.
        let reg = default_factory_registry();
        let result = reg.invoke(
            "provider.openai.chat",
            serde_json::json!({
                "id": "test",
                "base_url": "https://api.openai.com/v1",
            }),
            &ctx(),
        );
        assert!(result.is_ok());
    }

    #[cfg(feature = "openai")]
    #[test]
    fn provider_openai_chat_invocation_fails_with_bad_config() {
        let reg = default_factory_registry();
        let result = reg.invoke(
            "provider.openai.chat",
            serde_json::json!({ "bad_field": true }),
            &ctx(),
        );
        assert!(matches!(result, Err(FactoryError::InvalidConfig { .. })));
    }

    #[test]
    fn register_returns_chained_registry() {
        let reg = FactoryRegistry::new();
        let reg = register_memory_stores(reg);
        assert!(reg.contains("store.session.memory"));
        assert!(reg.contains("store.execution.memory"));
    }

    #[test]
    fn register_providers_returns_chained_registry() {
        let reg = FactoryRegistry::new();
        let reg = register_providers(reg);
        #[cfg(feature = "openai")]
        {
            assert!(reg.contains("provider.openai.chat"));
            assert!(reg.contains("provider.openai.embedding"));
        }
        #[cfg(feature = "anthropic")]
        {
            assert!(reg.contains("provider.anthropic.chat"));
        }
    }

    #[test]
    fn register_context_pipeline_returns_chained_registry() {
        let reg = FactoryRegistry::new();
        let reg = register_context_pipeline(reg);
        assert!(reg.contains("context.pipeline"));
    }

    #[test]
    fn provider_http_component_config_roundtrip() {
        let json = serde_json::json!({
            "id": "my-openai",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test123",
            "organization": "org-abc",
            "timeout_secs": 30,
            "connect_timeout_secs": 5,
            "max_retries": 3,
        });
        let cfg: ProviderHttpComponentConfig =
            serde_json::from_value(json).expect("deserialize should succeed");
        assert_eq!(cfg.id, "my-openai");
        assert_eq!(cfg.base_url, "https://api.openai.com/v1");
        assert_eq!(cfg.api_key.as_deref(), Some("sk-test123"));
        assert_eq!(cfg.organization.as_deref(), Some("org-abc"));
        assert_eq!(cfg.timeout_secs, Some(30));
        assert_eq!(cfg.connect_timeout_secs, Some(5));
        assert_eq!(cfg.max_retries, Some(3));

        let http = cfg.into_provider_http_config();
        assert_eq!(http.id.as_str(), "my-openai");
        assert_eq!(http.base_url, "https://api.openai.com/v1");
    }

    #[test]
    fn provider_http_component_config_defaults() {
        let json = serde_json::json!({
            "id": "minimal",
            "base_url": "https://example.com",
        });
        let cfg: ProviderHttpComponentConfig =
            serde_json::from_value(json).expect("deserialize should succeed");
        assert_eq!(cfg.id, "minimal");
        assert!(cfg.api_key.is_none());
        assert!(cfg.organization.is_none());

        let http = cfg.into_provider_http_config();
        assert_eq!(http.max_retries, 2);
    }

    #[test]
    fn context_pipeline_config_roundtrip() {
        let json = serde_json::json!({
            "max_history_messages": 100,
            "max_history_tokens": 128_000,
            "enable_compaction_filter": false,
        });
        let cfg: ContextPipelineConfig =
            serde_json::from_value(json).expect("deserialize should succeed");
        assert_eq!(cfg.max_history_messages, Some(100));
        assert_eq!(cfg.max_history_tokens, Some(128_000));
        assert_eq!(cfg.enable_compaction_filter, Some(false));
    }
}
