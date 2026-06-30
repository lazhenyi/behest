//! Centralized agent configuration.
//!
//! `AgentConfig` provides a single entry point for configuring all
//! components of an agent runtime. It supports three loading strategies:
//!
//! 1. **Manual builder** — highest priority, set by library callers
//! 2. **File** — TOML, JSON, or YAML (auto-detected by extension)
//! 3. **Environment variables** — with configurable prefix
//!
//! # Example
//!
//! ```rust,ignore
//! use behest::config::{AgentConfig, AgentConfigBuilder};
//!
//! let config = AgentConfig::builder()
//!     .with_file("config.toml")?
//!     .with_env("AGENTS")?
//!     .build()?;
//!
//! let runtime = config.into_runtime().await?;
//! ```

#![allow(clippy::too_many_lines)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::ProviderId;
use crate::runtime::lifecycle::ShutdownToken;
use crate::runtime::managed::ManagedRuntime;
use crate::runtime::registry::ComponentRegistry;
use crate::runtime::{AgentRuntime, ContextPipeline, RuntimePolicy, RuntimeStore};

pub mod component;
pub mod loader;
pub mod provider;
pub mod runtime;
pub mod store;

#[cfg(feature = "rag")]
pub mod rag;

#[cfg(feature = "queue")]
pub mod queue;

pub use component::{ComponentConfig, ComponentFile};
pub use loader::ConfigLoader;
pub use provider::{ProviderConfig, ProviderType};
pub use runtime::{RuntimeConfig, RuntimePolicyConfig};
pub use store::{StoreBackend, StoreConfig};

#[cfg(feature = "rag")]
pub use rag::RagConfig;

#[cfg(feature = "queue")]
pub use queue::{QueueBackend, QueueConfig};

/// Top-level agent configuration combining runtime, providers, stores,
/// and optional RAG and queue sub-configurations.
///
/// Serialisable and deserialisable via `serde`, enabling file-based
/// and environment-variable-based loading.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Runtime execution configuration.
    #[serde(default)]
    pub runtime: RuntimeConfig,

    /// Per-provider connection configuration, keyed by provider ID.
    #[serde(default)]
    pub providers: HashMap<ProviderId, ProviderConfig>,

    /// Storage backend configuration.
    #[serde(default)]
    pub stores: StoreConfig,

    /// RAG (Retrieval-Augmented Generation) configuration.
    #[cfg(feature = "rag")]
    #[serde(default)]
    pub rag: Option<RagConfig>,

    /// External event publishing configuration.
    #[cfg(feature = "queue")]
    #[serde(default)]
    pub queue: Option<QueueConfig>,

    /// Declarative component declarations for the composable runtime
    /// model. Populated by `[[component]]` config sections and consumed
    /// by `ComponentRegistry`.
    #[serde(default)]
    pub components: Vec<ComponentConfig>,
}

impl AgentConfig {
    /// Creates a new config builder pre-populated with defaults.
    #[must_use]
    pub fn builder() -> AgentConfigBuilder {
        AgentConfigBuilder::default()
    }

    /// Converts the config into a fully assembled `AgentRuntime`.
    ///
    /// This is a convenience method that builds all components
    /// (providers, stores, context pipeline, tool runtime, policy)
    /// from the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when any component cannot be constructed.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let runtime = AgentConfig::default().into_runtime().await?;
    /// ```
    pub async fn into_runtime(self) -> Result<AgentRuntime> {
        self.into_builder().build_runtime().await
    }

    /// Converts into a builder for further customization.
    #[must_use]
    pub fn into_builder(self) -> AgentConfigBuilder {
        AgentConfigBuilder::from_config(self)
    }

    /// Validates the configuration, returning an error for missing required fields.
    ///
    /// # Errors
    ///
    /// Returns an error when required fields are missing or invalid.
    pub fn validate(&self) -> Result<()> {
        let stores = &self.stores;

        let needs_redis = stores.session_backend == StoreBackend::Redis
            || stores.execution_backend == StoreBackend::Redis
            || stores.run_backend == StoreBackend::Redis
            || stores.artifact_backend == StoreBackend::Redis;
        if needs_redis && stores.redis_url.is_none() {
            return Err(Error::Config(
                "redis_url is required when any store backend is Redis".to_owned(),
            ));
        }

        if stores.embedding_backend == StoreBackend::Redis && stores.redis_url.is_none() {
            return Err(Error::Config(
                "redis_url is required when embedding backend is Redis".to_owned(),
            ));
        }

        #[cfg(feature = "rag")]
        if let Some(ref rag) = self.rag
            && !self.providers.contains_key(&rag.provider_id)
        {
            return Err(Error::Config(format!(
                "RAG provider '{}' is not configured in [providers]",
                rag.provider_id
            )));
        }

        #[cfg(feature = "queue")]
        if let Some(ref queue) = self.queue {
            match queue.backend {
                QueueBackend::Nats => {
                    if queue.nats_url.is_none() {
                        return Err(Error::Config(
                            "nats_url is required for NATS queue backend".to_owned(),
                        ));
                    }
                }
                QueueBackend::RedisStreams => {
                    if queue.redis_url.is_none() {
                        return Err(Error::Config(
                            "redis_url is required for Redis Streams queue backend".to_owned(),
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

/// Builder for `AgentConfig` with layered configuration support.
///
/// Layers (lowest to highest priority):
/// 1. Defaults
/// 2. File sources
/// 3. Environment variables
/// 4. Manual builder setters (highest priority)
#[derive(Debug, Clone, Default)]
pub struct AgentConfigBuilder {
    config: AgentConfig,
    file_sources: Vec<String>,
    env_prefixes: Vec<String>,
}

impl AgentConfigBuilder {
    fn from_config(config: AgentConfig) -> Self {
        Self {
            config,
            file_sources: Vec::new(),
            env_prefixes: Vec::new(),
        }
    }

    /// Adds a configuration file to the loading chain.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or parsed.
    pub fn with_file(mut self, path: impl Into<String>) -> Result<Self> {
        let path = path.into();
        let file_value: serde_json::Value = loader::load_file(&path)?;

        let mut base_value = serde_json::to_value(&self.config)
            .map_err(|e| Error::Config(format!("failed to serialize base config: {e}")))?;

        loader::merge_json(&mut base_value, file_value);
        loader::substitute_json(&mut base_value);

        self.config = serde_json::from_value(base_value)
            .map_err(|e| Error::Config(format!("failed to deserialize merged config: {e}")))?;

        self.file_sources.push(path);
        Ok(self)
    }

    /// Adds environment variable loading to the chain.
    ///
    /// # Errors
    ///
    /// Returns an error when the environment-based config cannot be parsed.
    pub fn with_env(mut self, prefix: impl Into<String>) -> Result<Self> {
        let prefix = prefix.into();
        let env_value: serde_json::Value = loader::load_env(&prefix)?;

        let mut base_value = serde_json::to_value(&self.config)
            .map_err(|e| Error::Config(format!("failed to serialize base config: {e}")))?;

        loader::merge_json(&mut base_value, env_value);
        loader::substitute_json(&mut base_value);

        self.config = serde_json::from_value(base_value)
            .map_err(|e| Error::Config(format!("failed to deserialize merged config: {e}")))?;

        self.env_prefixes.push(prefix);
        Ok(self)
    }

    /// Sets the runtime configuration (highest priority).
    #[must_use]
    pub fn with_runtime(mut self, runtime: RuntimeConfig) -> Self {
        self.config.runtime = runtime;
        self
    }

    /// Registers a provider configuration (highest priority).
    #[must_use]
    pub fn with_provider(mut self, id: impl Into<ProviderId>, config: ProviderConfig) -> Self {
        self.config.providers.insert(id.into(), config);
        self
    }

    /// Sets the store configuration (highest priority).
    #[must_use]
    pub fn with_stores(mut self, stores: StoreConfig) -> Self {
        self.config.stores = stores;
        self
    }

    /// Append a component declaration to the configuration.
    #[must_use]
    pub fn with_component(mut self, component: ComponentConfig) -> Self {
        self.config.components.push(component);
        self
    }

    /// Load `[[component]]` sections from a TOML string and append
    /// them to the configuration.
    ///
    /// # Errors
    /// Returns the underlying TOML parse error if the input is invalid.
    pub fn with_component_toml(mut self, toml_text: &str) -> Result<Self> {
        let file = ComponentFile::from_toml(toml_text)
            .map_err(|e| Error::Config(format!("component toml parse error: {e}")))?;
        for c in file.component {
            self.config.components.push(c);
        }
        Ok(self)
    }

    /// Sets the RAG configuration (highest priority).
    #[cfg(feature = "rag")]
    #[must_use]
    pub fn with_rag(mut self, rag: RagConfig) -> Self {
        self.config.rag = Some(rag);
        self
    }

    /// Sets the queue configuration (highest priority).
    #[cfg(feature = "queue")]
    #[must_use]
    pub fn with_queue(mut self, queue: QueueConfig) -> Self {
        self.config.queue = Some(queue);
        self
    }

    /// Builds the final `AgentConfig` after validation and placeholder substitution.
    ///
    /// # Errors
    ///
    /// Returns an error when serialization, placeholder substitution,
    /// or validation fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = AgentConfig::builder()
    ///     .with_runtime(runtime_config)
    ///     .build()?;
    /// ```
    pub fn build(mut self) -> Result<AgentConfig> {
        let mut value = serde_json::to_value(&self.config).map_err(|e| {
            Error::Config(format!(
                "failed to serialize config for placeholder substitution: {e}"
            ))
        })?;
        loader::substitute_json(&mut value);
        self.config = serde_json::from_value(value).map_err(|e| {
            Error::Config(format!(
                "failed to deserialize config after placeholder substitution: {e}"
            ))
        })?;

        self.config.validate()?;
        Ok(self.config)
    }

    /// Builds a fully assembled `AgentRuntime` from the configuration.
    ///
    /// This registers providers, constructs stores, sets up the context
    /// pipeline (with optional RAG adapter), and creates the tool runtime.
    ///
    /// # Errors
    ///
    /// Returns an error when validation fails or any component cannot be constructed.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let runtime = AgentConfig::builder()
    ///     .with_runtime(runtime_config)
    ///     .build_runtime().await?;
    /// ```
    pub async fn build_runtime(self) -> Result<AgentRuntime> {
        let config = self.build()?;

        let policy: RuntimePolicy = config.runtime.policy.into();

        #[allow(unused_mut)]
        let mut registry = crate::provider::ProviderRegistry::new();

        // Register providers based on their configured type.
        for (id, provider_config) in &config.providers {
            let http_config = provider_config.to_http_config(id.clone());

            match &provider_config.provider_type {
                #[cfg(feature = "openai")]
                Some(ProviderType::OpenAi) => {
                    let chat_adapter = crate::adapt::openai::OpenAiChatAdapter::new(
                        http_config.clone(),
                    )
                    .map_err(|e| {
                        Error::Config(format!(
                            "failed to create OpenAI chat adapter for '{id}': {e}"
                        ))
                    })?;
                    registry.register_chat(chat_adapter);

                    let embed_adapter = crate::adapt::openai::OpenAiEmbeddingAdapter::new(
                        http_config,
                    )
                    .map_err(|e| {
                        Error::Config(format!(
                            "failed to create OpenAI embedding adapter for '{id}': {e}"
                        ))
                    })?;
                    registry.register_embedding(embed_adapter);

                    tracing::info!(%id, "registered OpenAI chat + embedding provider");
                }

                #[cfg(feature = "anthropic")]
                Some(ProviderType::Anthropic) => {
                    let chat_adapter =
                        crate::adapt::anthropic::AnthropicChatAdapter::new(http_config.clone())
                            .map_err(|e| {
                                Error::Config(format!(
                                    "failed to create Anthropic chat adapter for '{id}': {e}"
                                ))
                            })?;
                    registry.register_chat(chat_adapter);
                    tracing::info!(%id, "registered Anthropic chat provider");
                }

                None => {
                    tracing::info!(%id, base_url = %http_config.base_url, "provider configured without adapter type; manual registration required");
                }

                // Catch-all for variants not covered by enabled features.
                #[allow(unreachable_patterns)]
                _ => {
                    tracing::debug!(%id, "provider adapter type not available with current feature flags");
                }
            }
        }

        // Build stores from StoreConfig.
        let sessions = build_session_store(&config.stores).await?;
        let executions = build_execution_store(&config.stores).await?;
        let runs = crate::runtime::memory::MemoryRunStore::new();

        // Build Extensions from components, then construct runtime from it.
        let exts = crate::runtime::extensions::Extensions::new();

        exts.session_stores
            .register_or_replace("default", std::sync::Arc::from(sessions));
        exts.execution_stores
            .register_or_replace("default", std::sync::Arc::from(executions));
        exts.run_stores
            .register_or_replace("default", std::sync::Arc::new(runs));

        // Copy providers from the provider registry into runtime extensions.
        for provider_id in registry.chat_ids() {
            if let Some(provider) = registry.chat(&provider_id) {
                let _ = exts
                    .chat_providers
                    .register_or_replace(provider_id.as_str(), provider);
            }
        }
        for provider_id in registry.embedding_ids() {
            if let Some(provider) = registry.embedding(&provider_id) {
                let _ = exts
                    .embedding_providers
                    .register_or_replace(provider_id.as_str(), provider);
            }
        }

        let _store = std::sync::Arc::new(RuntimeStore::from_extensions(&exts));

        #[allow(unused)]
        let mut context =
            ContextPipeline::new().with_max_history(config.runtime.max_history_messages);

        #[cfg(feature = "rag")]
        if let Some(ref rag_config) = config.rag {
            let provider = registry.embedding(&rag_config.provider_id).ok_or_else(|| {
                Error::Config(format!(
                    "RAG embedding provider '{}' not found in registry; register an embedding-capable provider first",
                    rag_config.provider_id
                ))
            })?;

            let embedding_store: std::sync::Arc<dyn crate::store::EmbeddingStore> =
                build_embedding_store(&config.stores).await?;

            let adapter = crate::rag::RagContextAdapter::new(
                provider,
                embedding_store,
                rag_config.model.clone(),
            )
            .with_limit(rag_config.limit)
            .with_template(rag_config.template.clone())
            .with_metadata_field(rag_config.metadata_field.clone());

            context.register_arc(std::sync::Arc::new(adapter));
        }

        #[allow(unused_mut)]
        let mut runtime = AgentRuntime::new(std::sync::Arc::new(exts), policy);

        #[cfg(feature = "queue")]
        if let Some(ref queue_config) = config.queue {
            let publisher = build_event_publisher(queue_config).await?;
            runtime = runtime.with_event_publisher(std::sync::Arc::from(publisher));
        }

        Ok(runtime)
    }

    /// Builds a fully assembled [`ManagedRuntime`] from the configuration.
    ///
    /// This is the composable counterpart to [`build_runtime`](Self::build_runtime).
    /// It wraps the runtime in a [`ManagedRuntime`] together with a
    /// [`ComponentRegistry`] and [`ShutdownToken`], enabling coordinated
    /// lifecycle management and hot-reload.
    ///
    /// When `[[component]]` sections are present, they are instantiated
    /// via the [`FactoryRegistry`](crate::runtime::FactoryRegistry) and
    /// registered into the [`ComponentRegistry`].
    ///
    /// # Errors
    ///
    /// Returns an error when validation fails or any component cannot be
    /// constructed.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let managed = AgentConfig::builder()
    ///     .with_file("config.toml")?
    ///     .build_managed().await?;
    /// managed.serve().await?;
    /// ```
    pub async fn build_managed(self) -> Result<ManagedRuntime> {
        let config = self.build()?;
        let policy: RuntimePolicy = config.runtime.policy.into();
        let shutdown = ShutdownToken::new();
        let registry = ComponentRegistry::with_shutdown(shutdown.clone());

        #[allow(unused_mut)]
        let mut provider_registry = crate::provider::ProviderRegistry::new();

        // Register providers from [providers] section.
        for (id, provider_config) in &config.providers {
            let http_config = provider_config.to_http_config(id.clone());
            match &provider_config.provider_type {
                #[cfg(feature = "openai")]
                Some(ProviderType::OpenAi) => {
                    let chat_adapter = crate::adapt::openai::OpenAiChatAdapter::new(
                        http_config.clone(),
                    )
                    .map_err(|e| {
                        Error::Config(format!(
                            "failed to create OpenAI chat adapter for '{id}': {e}"
                        ))
                    })?;
                    provider_registry.register_chat(chat_adapter);

                    let embed_adapter = crate::adapt::openai::OpenAiEmbeddingAdapter::new(
                        http_config,
                    )
                    .map_err(|e| {
                        Error::Config(format!(
                            "failed to create OpenAI embedding adapter for '{id}': {e}"
                        ))
                    })?;
                    provider_registry.register_embedding(embed_adapter);
                    tracing::info!(%id, "registered OpenAI chat + embedding provider");
                }

                #[cfg(feature = "anthropic")]
                Some(ProviderType::Anthropic) => {
                    let chat_adapter =
                        crate::adapt::anthropic::AnthropicChatAdapter::new(http_config.clone())
                            .map_err(|e| {
                                Error::Config(format!(
                                    "failed to create Anthropic chat adapter for '{id}': {e}"
                                ))
                            })?;
                    provider_registry.register_chat(chat_adapter);
                    tracing::info!(%id, "registered Anthropic chat provider");
                }

                None => {
                    tracing::info!(%id, base_url = %http_config.base_url, "provider configured without adapter type; manual registration required");
                }

                #[allow(unreachable_patterns)]
                _ => {
                    tracing::debug!(%id, "provider adapter type not available with current feature flags");
                }
            }
        }

        // Build stores from StoreConfig.
        let sessions = build_session_store(&config.stores).await?;
        let executions = build_execution_store(&config.stores).await?;
        let runs = crate::runtime::memory::MemoryRunStore::new();

        // Build Extensions and populate from providers/stores.
        let exts = crate::runtime::extensions::Extensions::new();
        exts.session_stores
            .register_or_replace("default", std::sync::Arc::from(sessions));
        exts.execution_stores
            .register_or_replace("default", std::sync::Arc::from(executions));
        exts.run_stores
            .register_or_replace("default", std::sync::Arc::new(runs));
        for provider_id in provider_registry.chat_ids() {
            if let Some(provider) = provider_registry.chat(&provider_id) {
                let _ = exts
                    .chat_providers
                    .register_or_replace(provider_id.as_str(), provider);
            }
        }
        for provider_id in provider_registry.embedding_ids() {
            if let Some(provider) = provider_registry.embedding(&provider_id) {
                let _ = exts
                    .embedding_providers
                    .register_or_replace(provider_id.as_str(), provider);
            }
        }

        let _store = std::sync::Arc::new(RuntimeStore::from_extensions(&exts));

        #[allow(unused)]
        let mut context =
            ContextPipeline::new().with_max_history(config.runtime.max_history_messages);

        #[cfg(feature = "rag")]
        if let Some(ref rag_config) = config.rag {
            let provider = provider_registry
                .embedding(&rag_config.provider_id)
                .ok_or_else(|| {
                    Error::Config(format!(
                        "RAG embedding provider '{}' not found in registry",
                        rag_config.provider_id
                    ))
                })?;
            let embedding_store: std::sync::Arc<dyn crate::store::EmbeddingStore> =
                build_embedding_store(&config.stores).await?;
            let adapter = crate::rag::RagContextAdapter::new(
                provider,
                embedding_store,
                rag_config.model.clone(),
            )
            .with_limit(rag_config.limit)
            .with_template(rag_config.template.clone())
            .with_metadata_field(rag_config.metadata_field.clone());
            context.register_arc(std::sync::Arc::new(adapter));
        }

        // Register [[component]] entries from config into ComponentRegistry
        // via FactoryRegistry.
        if !config.components.is_empty() {
            let factory = crate::runtime::default_factory_registry();
            let ctx = crate::runtime::ComponentContext::new(shutdown.child());
            for comp_cfg in &config.components {
                if factory.contains(&comp_cfg.kind) {
                    let any = factory
                        .invoke(&comp_cfg.kind, comp_cfg.config.clone(), &ctx)
                        .map_err(|e| {
                            Error::Config(format!("factory for '{}' failed: {e}", comp_cfg.kind))
                        })?;
                    let descriptor = crate::runtime::ComponentDescriptor {
                        name: comp_cfg.name.clone(),
                        depends_on: comp_cfg.depends_on.clone(),
                        config: comp_cfg.config.clone(),
                    };
                    let oneshot =
                        OneShotFactory::new(comp_cfg.name.clone(), comp_cfg.kind.clone(), any);
                    registry
                        .register_factory(descriptor, Box::new(oneshot))
                        .map_err(|e| {
                            Error::Config(format!("component registration failed: {e}"))
                        })?;
                } else {
                    tracing::warn!(
                        kind = %comp_cfg.kind,
                        name = %comp_cfg.name,
                        "no factory registered for component kind; skipping"
                    );
                }
            }
            // Init and start components registered via [[component]].
            registry
                .init_all()
                .await
                .map_err(|e| Error::Config(format!("component init failed: {e}")))?;
            registry
                .start_all()
                .await
                .map_err(|e| Error::Config(format!("component start failed: {e}")))?;
        }

        #[allow(unused_mut)]
        let mut runtime = AgentRuntime::new(std::sync::Arc::new(exts), policy);

        #[cfg(feature = "queue")]
        if let Some(ref queue_config) = config.queue {
            let publisher = build_event_publisher(queue_config).await?;
            runtime = runtime.with_event_publisher(std::sync::Arc::from(publisher));
        }

        Ok(ManagedRuntime::new(runtime, registry, shutdown))
    }
}

/// Adapter that wraps a pre-built [`AnyComponent`] into a
/// [`ComponentFactory`], allowing the [`ComponentRegistry`] to manage
/// its lifecycle uniformly.
struct OneShotFactory {
    name: String,
    kind: &'static str,
    instance: Box<dyn crate::runtime::AnyComponent>,
}

impl OneShotFactory {
    fn new(name: String, kind: String, instance: Box<dyn crate::runtime::AnyComponent>) -> Self {
        // Leak the kind string once to satisfy the `'static` bound on
        // [`ComponentFactory::kind`]. Acceptable for the small, fixed
        // set of component kinds registered at startup.
        let kind_static: &'static str = Box::leak(kind.into_boxed_str());
        Self {
            name,
            kind: kind_static,
            instance,
        }
    }
}

#[async_trait::async_trait]
impl crate::runtime::ComponentFactory for OneShotFactory {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> &'static str {
        self.kind
    }

    fn depends_on(&self) -> Vec<String> {
        Vec::new()
    }

    async fn build(
        self: Box<Self>,
        _config: serde_json::Value,
        _ctx: &crate::runtime::ComponentContext,
    ) -> std::result::Result<Box<dyn crate::runtime::AnyComponent>, crate::runtime::RegistryError>
    {
        Ok(self.instance)
    }
}

#[allow(clippy::too_many_lines, clippy::unused_async)]
async fn build_session_store(config: &StoreConfig) -> Result<Box<dyn crate::store::SessionStore>> {
    match config.session_backend {
        StoreBackend::Memory => Ok(Box::new(crate::store::memory::MemorySessionStore::new())),
        StoreBackend::Redis => {
            #[cfg(feature = "redis")]
            {
                let url = config.redis_url.as_deref().ok_or_else(|| {
                    Error::Config("redis_url is required for Redis session store".to_owned())
                })?;
                let store = crate::store::redis::RedisSessionStore::new(url).map_err(|e| {
                    Error::Config(format!("failed to create Redis session store: {e}"))
                })?;
                Ok(Box::new(store))
            }
            #[cfg(not(feature = "redis"))]
            Err(Error::Config(
                "Redis session store requires the 'redis' feature".to_owned(),
            ))
        }
        StoreBackend::Sql => {
            #[cfg(feature = "sqlx-postgres")]
            {
                let url = config.sql_url.as_deref().ok_or_else(|| {
                    Error::Config("sql_url is required for SQL session store".to_owned())
                })?;
                let pool = sqlx::PgPool::connect(url)
                    .await
                    .map_err(|e| Error::Config(format!("failed to connect to PostgreSQL: {e}")))?;
                Ok(Box::new(crate::store::sql::SqlSessionStore::new(pool)))
            }
            #[cfg(all(feature = "sqlx-mysql", not(feature = "sqlx-postgres")))]
            {
                let url = config.sql_url.as_deref().ok_or_else(|| {
                    Error::Config("sql_url is required for SQL session store".to_owned())
                })?;
                let pool = sqlx::MySqlPool::connect(url)
                    .await
                    .map_err(|e| Error::Config(format!("failed to connect to MySQL: {e}")))?;
                Ok(Box::new(crate::store::sql::SqlSessionStore::new(pool)))
            }
            #[cfg(all(
                feature = "sqlx-sqlite",
                not(feature = "sqlx-postgres"),
                not(feature = "sqlx-mysql")
            ))]
            {
                let url = config.sql_url.as_deref().ok_or_else(|| {
                    Error::Config("sql_url is required for SQL session store".to_owned())
                })?;
                let pool = sqlx::SqlitePool::connect(url)
                    .await
                    .map_err(|e| Error::Config(format!("failed to connect to SQLite: {e}")))?;
                Ok(Box::new(crate::store::sql::SqlSessionStore::new(pool)))
            }
            #[cfg(not(any(
                feature = "sqlx-postgres",
                feature = "sqlx-mysql",
                feature = "sqlx-sqlite"
            )))]
            Err(Error::Config(
                "SQL session store requires a sqlx-* feature".to_owned(),
            ))
        }
        StoreBackend::Mongo => {
            #[cfg(feature = "mongodb")]
            {
                let url = config.mongo_url.as_deref().ok_or_else(|| {
                    Error::Config("mongo_url is required for MongoDB session store".to_owned())
                })?;
                let store = crate::store::mongodb::MongodbSessionStore::new(url, "behest")
                    .await
                    .map_err(|e| {
                        Error::Config(format!("failed to create MongoDB session store: {e}"))
                    })?;
                Ok(Box::new(store))
            }
            #[cfg(not(feature = "mongodb"))]
            Err(Error::Config(
                "MongoDB session store requires the 'mongodb' feature".to_owned(),
            ))
        }
        StoreBackend::Surreal => Err(Error::Config(
            "SurrealDB session store is not supported by this build".to_owned(),
        )),
    }
}

#[allow(clippy::unused_async)]
async fn build_execution_store(
    config: &StoreConfig,
) -> Result<Box<dyn crate::store::ExecutionStore>> {
    match config.execution_backend {
        StoreBackend::Memory => Ok(Box::new(crate::store::memory::MemoryExecutionStore::new())),
        StoreBackend::Redis => Err(Error::Config(
            "Redis execution store is not supported".to_owned(),
        )),
        StoreBackend::Sql => {
            #[cfg(feature = "sqlx-postgres")]
            {
                let url = config.sql_url.as_deref().ok_or_else(|| {
                    Error::Config("sql_url is required for SQL execution store".to_owned())
                })?;
                let pool = sqlx::PgPool::connect(url)
                    .await
                    .map_err(|e| Error::Config(format!("failed to connect to PostgreSQL: {e}")))?;
                Ok(Box::new(crate::store::sql::SqlExecutionStore::new(pool)))
            }
            #[cfg(all(feature = "sqlx-mysql", not(feature = "sqlx-postgres")))]
            {
                let url = config.sql_url.as_deref().ok_or_else(|| {
                    Error::Config("sql_url is required for SQL execution store".to_owned())
                })?;
                let pool = sqlx::MySqlPool::connect(url)
                    .await
                    .map_err(|e| Error::Config(format!("failed to connect to MySQL: {e}")))?;
                Ok(Box::new(crate::store::sql::SqlExecutionStore::new(pool)))
            }
            #[cfg(all(
                feature = "sqlx-sqlite",
                not(feature = "sqlx-postgres"),
                not(feature = "sqlx-mysql")
            ))]
            {
                let url = config.sql_url.as_deref().ok_or_else(|| {
                    Error::Config("sql_url is required for SQL execution store".to_owned())
                })?;
                let pool = sqlx::SqlitePool::connect(url)
                    .await
                    .map_err(|e| Error::Config(format!("failed to connect to SQLite: {e}")))?;
                Ok(Box::new(crate::store::sql::SqlExecutionStore::new(pool)))
            }
            #[cfg(not(any(
                feature = "sqlx-postgres",
                feature = "sqlx-mysql",
                feature = "sqlx-sqlite"
            )))]
            Err(Error::Config(
                "SQL execution store requires a sqlx-* feature".to_owned(),
            ))
        }
        StoreBackend::Mongo => Err(Error::Config(
            "MongoDB execution store is not supported".to_owned(),
        )),
        StoreBackend::Surreal => Err(Error::Config(
            "SurrealDB execution store is not supported".to_owned(),
        )),
    }
}

#[cfg(feature = "rag")]
async fn build_embedding_store(
    config: &StoreConfig,
) -> Result<std::sync::Arc<dyn crate::store::EmbeddingStore>> {
    match config.embedding_backend {
        StoreBackend::Memory => Ok(std::sync::Arc::new(
            crate::store::memory::MemoryEmbeddingStore::new(),
        )),
        StoreBackend::Sql => {
            #[cfg(feature = "sqlx-postgres")]
            {
                let url = config.sql_url.as_deref().ok_or_else(|| {
                    Error::Config("sql_url is required for SQL embedding store".to_owned())
                })?;
                let pool = sqlx::PgPool::connect(url)
                    .await
                    .map_err(|e| Error::Config(format!("failed to connect to PostgreSQL: {e}")))?;
                Ok(std::sync::Arc::new(
                    crate::store::sql::SqlEmbeddingStore::new(pool),
                ))
            }
            #[cfg(not(feature = "sqlx-postgres"))]
            Err(Error::Config(
                "SQL embedding store requires the 'sqlx-postgres' feature (pgvector)".to_owned(),
            ))
        }
        StoreBackend::Redis | StoreBackend::Mongo | StoreBackend::Surreal => {
            Err(Error::Config(format!(
                "embedding store backend '{:?}' is not supported via auto-config",
                config.embedding_backend
            )))
        }
    }
}

#[cfg(feature = "queue")]
#[allow(clippy::unused_async)]
async fn build_event_publisher(
    config: &queue::QueueConfig,
) -> Result<Box<dyn crate::queue::EventPublisher>> {
    match config.backend {
        queue::QueueBackend::Nats => {
            #[cfg(feature = "nats")]
            {
                let url = config.nats_url.as_deref().ok_or_else(|| {
                    Error::Config("NATS URL is required for NATS queue backend".to_owned())
                })?;
                let publisher =
                    crate::queue::NatsEventPublisher::connect(url, &config.nats_subject)
                        .await
                        .map_err(|e| {
                            Error::Config(format!("failed to connect NATS publisher: {e}"))
                        })?;
                Ok(Box::new(publisher))
            }
            #[cfg(not(feature = "nats"))]
            Err(Error::Config(
                "NATS queue backend selected but 'nats' feature is not enabled".to_owned(),
            ))
        }
        queue::QueueBackend::RedisStreams => {
            #[cfg(feature = "redis")]
            {
                let url = config.redis_url.as_deref().ok_or_else(|| {
                    Error::Config(
                        "Redis URL is required for Redis Streams queue backend".to_owned(),
                    )
                })?;
                let publisher =
                    crate::queue::RedisStreamsPublisher::connect(url, &config.redis_stream_key)
                        .await
                        .map_err(|e| {
                            Error::Config(format!("failed to connect Redis Streams publisher: {e}"))
                        })?;
                Ok(Box::new(publisher))
            }
            #[cfg(not(feature = "redis"))]
            Err(Error::Config(
                "Redis Streams queue backend selected but 'redis' feature is not enabled"
                    .to_owned(),
            ))
        }
    }
}

#[allow(dead_code, clippy::too_many_lines)]
fn merge_configs(base: AgentConfig, overlay: AgentConfig) -> AgentConfig {
    let mut merged = base;

    // For runtime:
    if overlay.runtime.max_history_messages != 50 {
        merged.runtime.max_history_messages = overlay.runtime.max_history_messages;
    }

    if overlay.runtime.event_channel_capacity != 256 {
        merged.runtime.event_channel_capacity = overlay.runtime.event_channel_capacity;
    }

    // policy:
    if overlay.runtime.policy.max_iterations != 10 {
        merged.runtime.policy.max_iterations = overlay.runtime.policy.max_iterations;
    }
    if overlay.runtime.policy.max_tokens.is_some() {
        merged.runtime.policy.max_tokens = overlay.runtime.policy.max_tokens;
    }
    if overlay.runtime.policy.max_tool_concurrency != 4 {
        merged.runtime.policy.max_tool_concurrency = overlay.runtime.policy.max_tool_concurrency;
    }
    if overlay.runtime.policy.tool_timeout_secs != 30 {
        merged.runtime.policy.tool_timeout_secs = overlay.runtime.policy.tool_timeout_secs;
    }
    if overlay.runtime.policy.provider_timeout_secs != 60 {
        merged.runtime.policy.provider_timeout_secs = overlay.runtime.policy.provider_timeout_secs;
    }
    if !overlay.runtime.policy.continue_on_tool_failure {
        merged.runtime.policy.continue_on_tool_failure = false;
    }
    if !overlay.runtime.policy.retry_on_provider_error {
        merged.runtime.policy.retry_on_provider_error = false;
    }
    if overlay.runtime.policy.max_retries != 2 {
        merged.runtime.policy.max_retries = overlay.runtime.policy.max_retries;
    }

    // For stores:
    if overlay.stores.session_backend != StoreBackend::Memory {
        merged.stores.session_backend = overlay.stores.session_backend;
    }
    if overlay.stores.execution_backend != StoreBackend::Memory {
        merged.stores.execution_backend = overlay.stores.execution_backend;
    }
    if overlay.stores.run_backend != StoreBackend::Memory {
        merged.stores.run_backend = overlay.stores.run_backend;
    }
    if overlay.stores.embedding_backend != StoreBackend::Memory {
        merged.stores.embedding_backend = overlay.stores.embedding_backend;
    }
    if overlay.stores.artifact_backend != StoreBackend::Memory {
        merged.stores.artifact_backend = overlay.stores.artifact_backend;
    }
    if overlay.stores.redis_url.is_some() {
        merged.stores.redis_url = overlay.stores.redis_url;
    }
    if overlay.stores.sql_url.is_some() {
        merged.stores.sql_url = overlay.stores.sql_url;
    }
    if overlay.stores.mongo_url.is_some() {
        merged.stores.mongo_url = overlay.stores.mongo_url;
    }
    if overlay.stores.surreal_url.is_some() {
        merged.stores.surreal_url = overlay.stores.surreal_url;
    }
    if overlay.stores.qdrant_url.is_some() {
        merged.stores.qdrant_url = overlay.stores.qdrant_url;
    }
    if !overlay.stores.qdrant_collection.is_empty() {
        merged.stores.qdrant_collection = overlay.stores.qdrant_collection;
    }
    if overlay.stores.qdrant_dimensions != 1536 {
        merged.stores.qdrant_dimensions = overlay.stores.qdrant_dimensions;
    }

    // providers:
    for (id, cfg) in overlay.providers {
        let entry = merged.providers.entry(id).or_insert_with(|| cfg.clone());
        if cfg.base_url != entry.base_url
            && !cfg.base_url.is_empty()
            && cfg.base_url != "https://api.openai.com/v1"
        {
            entry.base_url = cfg.base_url;
        }
        if cfg.api_key.is_some() {
            entry.api_key = cfg.api_key;
        }
        if cfg.provider_type.is_some() {
            entry.provider_type = cfg.provider_type;
        }
        if cfg.model.is_some() {
            entry.model = cfg.model;
        }
        if !cfg.models.is_empty() {
            entry.models = cfg.models;
        }
        if cfg.compaction_model.is_some() {
            entry.compaction_model = cfg.compaction_model;
        }
        if cfg.organization.is_some() {
            entry.organization = cfg.organization;
        }
        if cfg.timeout_secs != 60 {
            entry.timeout_secs = cfg.timeout_secs;
        }
    }

    #[cfg(feature = "rag")]
    if overlay.rag.is_some() {
        merged.rag = overlay.rag;
    }

    #[cfg(feature = "queue")]
    if overlay.queue.is_some() {
        merged.queue = overlay.queue;
    }

    merged
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    #[cfg(feature = "rag")]
    use crate::provider::ModelName;
    use secrecy::ExposeSecret;

    #[test]
    fn default_config_should_have_reasonable_defaults() {
        let config = AgentConfig::default();
        assert_eq!(config.runtime.policy.max_iterations, 10);
        assert_eq!(config.runtime.policy.max_tool_concurrency, 4);
        assert_eq!(config.runtime.policy.tool_timeout_secs, 30);
        assert_eq!(config.runtime.policy.provider_timeout_secs, 60);
        assert_eq!(config.runtime.max_history_messages, 50);
        assert_eq!(config.runtime.event_channel_capacity, 256);
        assert_eq!(config.stores.session_backend, StoreBackend::Memory);
        assert!(config.providers.is_empty());
    }

    #[test]
    fn builder_should_set_runtime_config() {
        let runtime = RuntimeConfig {
            max_history_messages: 100,
            ..Default::default()
        };
        let config = AgentConfig::builder()
            .with_runtime(runtime)
            .build()
            .unwrap();
        assert_eq!(config.runtime.max_history_messages, 100);
    }

    #[test]
    fn builder_should_set_provider_config() {
        let provider = ProviderConfig::new("https://api.example.com");
        let config = AgentConfig::builder()
            .with_provider("example", provider)
            .build()
            .unwrap();
        assert_eq!(config.providers.len(), 1);
        let p = config
            .providers
            .get(&crate::provider::ProviderId::new("example"))
            .unwrap();
        assert_eq!(p.base_url, "https://api.example.com");
    }

    #[test]
    fn provider_config_should_resolve_env_var() {
        let provider = ProviderConfig {
            api_key: Some("env:HOME".to_owned()),
            ..ProviderConfig::new("https://api.example.com")
        };
        let resolved = provider.resolve_api_key();
        assert!(resolved.is_some());
    }

    #[test]
    fn provider_config_should_resolve_plain_key() {
        let provider = ProviderConfig {
            api_key: Some("sk-abc123".to_owned()),
            ..ProviderConfig::new("https://api.example.com")
        };
        let resolved = provider.resolve_api_key().unwrap();
        assert_eq!(resolved.expose_secret(), "sk-abc123");
    }

    #[test]
    fn provider_config_should_return_none_for_no_key() {
        let provider = ProviderConfig::new("https://api.example.com");
        assert!(provider.resolve_api_key().is_none());
    }

    #[test]
    fn build_runtime_with_defaults_should_not_panic() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let config = AgentConfig::default();
            let runtime = config.into_runtime().await.unwrap();
            assert_eq!(runtime.policy().max_iterations, 10);
        });
    }

    #[test]
    fn build_managed_with_defaults_should_succeed() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let managed = AgentConfig::builder()
                .build_managed()
                .await
                .expect("build_managed should succeed");
            assert_eq!(managed.runtime().policy().max_iterations, 10);
            assert!(managed.registry().is_empty());
            assert!(managed.is_healthy().await);
        });
    }

    #[test]
    fn build_managed_with_component_should_register() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let config = AgentConfig::builder()
                .with_component(
                    ComponentConfig::new("mem-session", "store.session.memory")
                        .with_config(serde_json::json!({})),
                )
                .build()
                .expect("build should succeed");
            let managed = config
                .into_builder()
                .build_managed()
                .await
                .expect("build_managed should succeed");
            assert_eq!(managed.registry().len(), 1);
        });
    }

    #[test]
    #[cfg(feature = "rag")]
    fn build_runtime_with_rag_should_fail_with_unregistered_provider() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let config = AgentConfig::builder()
                .with_rag(RagConfig {
                    provider_id: crate::provider::ProviderId::new("nonexistent"),
                    model: ModelName::new("text-embedding-3"),
                    limit: 3,
                    template: String::new(),
                    metadata_field: String::from("text"),
                })
                .build();
            assert!(config.is_err());
        });
    }

    #[test]
    fn config_should_serialize_and_deserialize_json() {
        let config = AgentConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let _: AgentConfig = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn store_config_defaults_to_memory() {
        let config = StoreConfig::default();
        assert_eq!(config.session_backend, StoreBackend::Memory);
        assert_eq!(config.embedding_backend, StoreBackend::Memory);
        assert!(config.redis_url.is_none());
    }

    #[test]
    fn validate_should_reject_redis_store_without_url() {
        let config = AgentConfig {
            stores: StoreConfig {
                session_backend: StoreBackend::Redis,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    #[cfg(feature = "queue")]
    fn validate_should_reject_nats_queue_without_url() {
        let config = AgentConfig {
            queue: Some(QueueConfig {
                backend: QueueBackend::Nats,
                nats_url: None,
                nats_subject: String::new(),
                redis_url: None,
                redis_stream_key: String::new(),
            }),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_should_pass_for_memory_only() {
        let config = AgentConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_loader_should_load_file() {
        use std::io::Write as _;

        let toml_content = "[runtime]\nmax_history_messages = 30\nevent_channel_capacity = 128\n";

        let mut tmp = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
        write!(tmp, "{toml_content}").unwrap();

        let config = AgentConfig::builder()
            .with_file(tmp.path().display().to_string())
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(config.runtime.max_history_messages, 30);
        assert_eq!(config.runtime.event_channel_capacity, 128);
        assert_eq!(config.runtime.policy.max_iterations, 10);
        assert_eq!(config.stores.session_backend, StoreBackend::Memory);
    }

    #[test]
    fn builder_merge_configs_should_override_defaults() {
        let overrides = AgentConfig {
            runtime: RuntimeConfig {
                max_history_messages: 42,
                ..Default::default()
            },
            ..Default::default()
        };
        let merged = merge_configs(AgentConfig::default(), overrides);
        assert_eq!(merged.runtime.max_history_messages, 42);
        assert_eq!(merged.runtime.policy.max_iterations, 10);
        assert_eq!(merged.stores.session_backend, StoreBackend::Memory);
    }

    #[test]
    fn test_env_placeholder_substitution() {
        let provider = ProviderConfig {
            base_url: String::from("${HOME}"),
            ..ProviderConfig::new("https://api.example.com")
        };
        let config = AgentConfig::builder()
            .with_provider("example", provider)
            .build()
            .unwrap();
        let expected = std::env::var("HOME").unwrap();
        assert_eq!(
            config
                .providers
                .get(&crate::provider::ProviderId::new("example"))
                .unwrap()
                .base_url,
            expected
        );
    }

    #[test]
    fn test_env_placeholder_substitution_with_default() {
        // Var is NOT set, should fall back to default
        let provider = ProviderConfig {
            base_url: String::from("${NONEXISTENT_VAR:-http://default-host}"),
            ..ProviderConfig::new("https://api.example.com")
        };
        let config = AgentConfig::builder()
            .with_provider("example", provider)
            .build()
            .unwrap();
        assert_eq!(
            config
                .providers
                .get(&crate::provider::ProviderId::new("example"))
                .unwrap()
                .base_url,
            "http://default-host"
        );
    }

    #[test]
    fn test_deep_merge_preserves_non_configured_base_values() {
        use std::io::Write as _;

        // 1. Pre-configure base with some manual values
        let base_runtime = RuntimeConfig {
            max_history_messages: 99,
            policy: RuntimePolicyConfig {
                max_iterations: 88,
                ..Default::default()
            },
            ..Default::default()
        };

        // 2. Write an overlay file that only overrides max_history_messages
        let toml_content = "[runtime]\nmax_history_messages = 123\n";
        let mut tmp = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
        write!(tmp, "{toml_content}").unwrap();

        // 3. Build and check if manual override max_iterations (88) was preserved
        let config = AgentConfig::builder()
            .with_runtime(base_runtime)
            .with_file(tmp.path().display().to_string())
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(config.runtime.max_history_messages, 123); // overridden by file
        assert_eq!(config.runtime.policy.max_iterations, 88); // PRESERVED manual override!
    }
}
