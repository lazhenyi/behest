//! Storage backend configuration.

use serde::{Deserialize, Serialize};

/// Storage backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoreBackend {
    /// In-memory (always available, no connection needed).
    #[default]
    Memory,
    /// Redis-backed storage.
    Redis,
    /// SQL database (PostgreSQL, MySQL, SQLite).
    Sql,
    /// MongoDB.
    Mongo,
    /// SurrealDB.
    Surreal,
}

/// Storage configuration section selecting backends for each store kind
/// and providing connection URLs.
///
/// All backends default to [`StoreBackend::Memory`], which requires no
/// connection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreConfig {
    /// Session store backend. Default: `Memory`.
    #[serde(default)]
    pub session_backend: StoreBackend,

    /// Execution store backend. Default: `Memory`.
    #[serde(default)]
    pub execution_backend: StoreBackend,

    /// Run store backend. Default: `Memory`.
    #[serde(default)]
    pub run_backend: StoreBackend,

    /// Embedding store backend. Default: `Memory`.
    #[serde(default)]
    pub embedding_backend: StoreBackend,

    /// Artifact store backend. Default: `Memory`.
    #[serde(default)]
    pub artifact_backend: StoreBackend,

    /// Redis connection URL. Required when any backend is [`StoreBackend::Redis`].
    #[serde(default)]
    pub redis_url: Option<String>,

    /// SQL connection URL. Required when any backend is [`StoreBackend::Sql`].
    #[serde(default)]
    pub sql_url: Option<String>,

    /// MongoDB connection URL. Required when any backend is [`StoreBackend::Mongo`].
    #[serde(default)]
    pub mongo_url: Option<String>,

    /// SurrealDB connection URL. Required when any backend is [`StoreBackend::Surreal`].
    #[serde(default)]
    pub surreal_url: Option<String>,

    /// Qdrant gRPC URL for the embedding store.
    #[serde(default)]
    pub qdrant_url: Option<String>,

    /// Qdrant collection name. Default: `""`.
    #[serde(default = "default_qdrant_collection")]
    pub qdrant_collection: String,

    /// Qdrant vector dimensions. Default: 1536.
    #[serde(default = "default_qdrant_dimensions")]
    pub qdrant_dimensions: u64,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            session_backend: StoreBackend::Memory,
            execution_backend: StoreBackend::Memory,
            run_backend: StoreBackend::Memory,
            embedding_backend: StoreBackend::Memory,
            artifact_backend: StoreBackend::Memory,
            redis_url: None,
            sql_url: None,
            mongo_url: None,
            surreal_url: None,
            qdrant_url: None,
            qdrant_collection: default_qdrant_collection(),
            qdrant_dimensions: default_qdrant_dimensions(),
        }
    }
}

fn default_qdrant_collection() -> String {
    String::new()
}

const fn default_qdrant_dimensions() -> u64 {
    1536
}
