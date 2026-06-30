//! In-memory storage implementations for testing, development, and prototyping.
//!
//! These stores use `tokio::sync::RwLock<HashMap>` internally and are always
//! available without any feature flags. Data is lost when the process exits
//! and is not persisted to disk.

pub mod artifact;
pub mod composite;
pub mod embedding;
pub mod execution;
pub mod session;

pub use artifact::MemoryArtifactStore;
pub use composite::MemoryStore;
pub use embedding::MemoryEmbeddingStore;
pub use execution::MemoryExecutionStore;
pub use session::MemorySessionStore;
