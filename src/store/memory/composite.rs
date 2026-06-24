//! Composite in-memory store that coordinates cascading operations
//! across all storage domains.

use uuid::Uuid;

use crate::store::memory::{
    MemoryArtifactStore, MemoryEmbeddingStore, MemoryExecutionStore, MemorySessionStore,
};
use crate::store::{ArtifactStore, EmbeddingStore, ExecutionStore, SessionStore, StoreResult};

/// A composite store that holds all four in-memory sub-stores
/// and provides coordinated operations, including cascading deletes.
///
/// # Example
///
/// ```rust
/// use behest::store::memory::MemoryStore;
///
/// # async fn example() -> Result<(), behest::StorageError> {
/// let store = MemoryStore::new();
/// // Use store.sessions, store.executions, store.embeddings, store.artifacts
/// // individually, or use store.delete_session_cascading() for full cleanup.
/// # Ok(())
/// # }
/// ```
#[derive(Default)]
pub struct MemoryStore {
    /// Session and message storage.
    pub sessions: MemorySessionStore,
    /// Embedding storage.
    pub embeddings: MemoryEmbeddingStore,
    /// Artifact storage.
    pub artifacts: MemoryArtifactStore,
    /// Tool execution and usage storage.
    pub executions: MemoryExecutionStore,
}

impl MemoryStore {
    /// Creates a new empty composite memory store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Deletes a session and all related data across all sub-stores.
    ///
    /// This is semantically equivalent to what a real database would do
    /// with foreign-key `ON DELETE CASCADE`, but across in-memory collections.
    ///
    /// # Errors
    ///
    /// Individual sub-store errors are collected and returned as a single
    /// [`StorageError::BackendError`](crate::error::StorageError::BackendError).
    pub async fn delete_session_cascading(&self, id: &Uuid) -> StoreResult<()> {
        // Collect errors from all sub-stores
        let mut errors: Vec<String> = Vec::new();

        if let Err(e) = self.sessions.delete_session(id).await {
            errors.push(format!("sessions: {e}"));
        }
        if let Err(e) = self.embeddings.delete_by_session(id).await {
            errors.push(format!("embeddings: {e}"));
        }
        if let Err(e) = self.artifacts.delete_by_session(id).await {
            errors.push(format!("artifacts: {e}"));
        }
        if let Err(e) = self.executions.delete_by_session(id).await {
            errors.push(format!("executions: {e}"));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(crate::error::StorageError::BackendError {
                backend: "memory".to_owned(),
                message: format!("cascade delete failed: {}", errors.join("; ")),
                source: None,
            })
        }
    }
}
