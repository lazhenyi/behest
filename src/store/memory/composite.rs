//! Composite in-memory store that coordinates cascading operations
//! across all storage domains.

use uuid::Uuid;

use crate::store::memory::{
    MemoryArtifactStore, MemoryEmbeddingStore, MemoryExecutionStore, MemorySessionStore,
};
use crate::store::{ArtifactStore, EmbeddingStore, ExecutionStore, SessionStore, StoreResult};

/// A composite in-memory store aggregating all four storage domains
/// with coordinated operations, including cascading deletes.
///
/// Provides direct access to each sub-store through public fields for
/// fine-grained control, plus `delete_session_cascading()` for atomic
/// cross-domain cleanup.
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
    /// Creates a new empty composite memory store with all four sub-stores.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Deletes a session and all related data across all sub-stores.
    ///
    /// This is semantically equivalent to what a real database would do
    /// with foreign-key `ON DELETE CASCADE`, but across in-memory collections.
    /// Collects errors from all sub-stores instead of short-circuiting.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::BackendError`](crate::error::StorageError::BackendError)
    /// aggregating errors from any sub-stores that failed.
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
