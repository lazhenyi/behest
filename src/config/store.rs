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

/// Storage configuration section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreConfig {
    /// Session store backend.
    #[serde(default)]
    pub session_backend: StoreBackend,

    /// Execution store backend.
    #[serde(default)]
    pub execution_backend: StoreBackend,

    /// Run store backend.
    #[serde(default)]
    pub run_backend: StoreBackend,

    /// Embedding store backend.
    #[serde(default)]
    pub embedding_backend: StoreBackend,

    /// Artifact store backend.
    #[serde(default)]
    pub artifact_backend: StoreBackend,

    /// Redis connection URL (required when any backend is Redis).
    #[serde(default)]
    pub redis_url: Option<String>,

    /// SQL connection URL (required when any backend is Sql).
    #[serde(default)]
    pub sql_url: Option<String>,

    /// MongoDB connection URL (required when any backend is Mongo).
    #[serde(default)]
    pub mongo_url: Option<String>,

    /// SurrealDB connection URL (required when any backend is Surreal).
    #[serde(default)]
    pub surreal_url: Option<String>,

    /// Qdrant gRPC URL for embedding store.
    #[serde(default)]
    pub qdrant_url: Option<String>,

    /// Qdrant collection name.
    #[serde(default = "default_qdrant_collection")]
    pub qdrant_collection: String,

    /// Qdrant vector dimensions.
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
