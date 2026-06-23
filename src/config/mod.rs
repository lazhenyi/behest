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
//! use agents::config::{AgentConfig, AgentConfigBuilder};
//!
//! let config = AgentConfig::builder()
//!     .with_file("config.toml")?
//!     .with_env("AGENTS")?
//!     .build()?;
//!
//! let runtime = config.into_runtime().await?;
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::ProviderId;
use crate::runtime::{AgentRuntime, ContextPipeline, RuntimePolicy, RuntimeStore, ToolRuntime};

pub mod loader;
pub mod provider;
pub mod runtime;
pub mod store;

#[cfg(feature = "rag")]
pub mod rag;

#[cfg(feature = "queue")]
pub mod queue;

pub use loader::ConfigLoader;
pub use provider::{ProviderConfig, ProviderType};
pub use runtime::{RuntimeConfig, RuntimePolicyConfig};
pub use store::{StoreBackend, StoreConfig};

#[cfg(feature = "rag")]
pub use rag::RagConfig;

#[cfg(feature = "queue")]
pub use queue::{QueueBackend, QueueConfig};

/// Top-level agent configuration.
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
        if let Some(ref rag) = self.rag {
            if !self.providers.contains_key(&rag.provider_id) {
                return Err(Error::Config(format!(
                    "RAG provider '{}' is not configured in [providers]",
                    rag.provider_id
                )));
            }
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
        let file_config: AgentConfig = loader::load_file(&path)?;
        self.config = merge_configs(self.config, file_config);
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
        let env_config: AgentConfig = loader::load_env(&prefix)?;
        self.config = merge_configs(self.config, env_config);
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

    /// Builds the final `AgentConfig` after validation.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration fails validation.
    pub fn build(self) -> Result<AgentConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Builds a fully assembled `AgentRuntime` from the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when validation fails or any component cannot be constructed.
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
        let store = std::sync::Arc::new(RuntimeStore::new(sessions, executions, Box::new(runs)));

        #[allow(unused_mut)]
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

        let tool_registry = crate::tool::ToolRegistry::new();
        let tool_runtime = ToolRuntime::new(tool_registry, policy.clone());

        #[allow(unused_mut)]
        let mut runtime = AgentRuntime::new(registry, context, tool_runtime, store, policy);

        #[cfg(feature = "queue")]
        if let Some(ref queue_config) = config.queue {
            let publisher = build_event_publisher(queue_config).await?;
            runtime = runtime.with_event_publisher(std::sync::Arc::from(publisher));
        }

        Ok(runtime)
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
                let store = crate::store::mongodb::MongodbSessionStore::new(url, "agents")
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
        StoreBackend::Surreal => {
            #[cfg(feature = "surrealdb")]
            {
                let url = config.surreal_url.as_deref().ok_or_else(|| {
                    Error::Config("surreal_url is required for SurrealDB session store".to_owned())
                })?;
                let db = surrealdb::engine::any::connect(url)
                    .await
                    .map_err(|e| Error::Config(format!("failed to connect to SurrealDB: {e}")))?;
                db.use_ns("agents").use_db("agents").await.map_err(|e| {
                    Error::Config(format!(
                        "failed to select SurrealDB namespace/database: {e}"
                    ))
                })?;
                Ok(Box::new(
                    crate::store::surrealdb::SurrealdbSessionStore::new(db),
                ))
            }
            #[cfg(not(feature = "surrealdb"))]
            Err(Error::Config(
                "SurrealDB session store requires the 'surrealdb' feature".to_owned(),
            ))
        }
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

fn merge_configs(base: AgentConfig, overlay: AgentConfig) -> AgentConfig {
    let mut merged = base;

    merged.runtime = overlay.runtime;
    merged.stores = overlay.stores;

    for (id, cfg) in overlay.providers {
        merged.providers.insert(id, cfg);
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
#[allow(clippy::unwrap_used)]
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
}
