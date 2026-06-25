//! In-memory artifact store backed by a `HashMap` protected by `RwLock`.

use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::store::{Artifact, ArtifactStore, StoreResult};

/// In-memory artifact store for testing, development, and prototyping.
///
/// Data is stored in a `HashMap<Uuid, Artifact>` protected by `RwLock`
/// and is lost when the process exits. Implements [`ArtifactStore`].
#[derive(Default)]
pub struct MemoryArtifactStore {
    artifacts: RwLock<HashMap<Uuid, Artifact>>,
}

impl MemoryArtifactStore {
    /// Creates an empty in-memory artifact store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ArtifactStore for MemoryArtifactStore {
    async fn put(&self, artifact: Artifact) -> StoreResult<Artifact> {
        let mut artifacts = self.artifacts.write().await;
        let id = artifact.id;
        artifacts.insert(id, artifact.clone());
        Ok(artifact)
    }

    async fn get(&self, id: &Uuid) -> StoreResult<Option<Artifact>> {
        let artifacts = self.artifacts.read().await;
        Ok(artifacts.get(id).cloned())
    }

    async fn delete(&self, id: &Uuid) -> StoreResult<()> {
        self.artifacts.write().await.remove(id);
        Ok(())
    }

    async fn list_by_session(&self, session_id: &Uuid) -> StoreResult<Vec<Artifact>> {
        let artifacts = self.artifacts.read().await;
        Ok(artifacts
            .values()
            .filter(|a| a.session_id == Some(*session_id))
            .cloned()
            .collect())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_artifact() -> Artifact {
        Artifact::new("test.txt", "text/plain", b"hello world".to_vec())
    }

    #[tokio::test]
    async fn memory_artifact_store_should_put_and_get() {
        let store = MemoryArtifactStore::new();
        let artifact = test_artifact();
        let id = artifact.id;

        store.put(artifact).await.unwrap();
        let loaded = store.get(&id).await.unwrap();

        assert!(loaded.is_some());
        let artifact = loaded.as_ref().unwrap();
        assert_eq!(artifact.name, "test.txt");
        assert_eq!(artifact.data, b"hello world");
    }

    #[tokio::test]
    async fn memory_artifact_store_should_delete() {
        let store = MemoryArtifactStore::new();
        let artifact = test_artifact();
        let id = artifact.id;

        store.put(artifact).await.unwrap();
        store.delete(&id).await.unwrap();

        assert!(store.get(&id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn memory_artifact_store_should_list_by_session() {
        let store = MemoryArtifactStore::new();
        let session_id = Uuid::now_v7();

        store
            .put(test_artifact().with_session(session_id))
            .await
            .unwrap();
        store
            .put(test_artifact().with_session(session_id))
            .await
            .unwrap();
        store.put(test_artifact()).await.unwrap();

        let results = store.list_by_session(&session_id).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn memory_artifact_store_should_return_none_for_unknown() {
        let store = MemoryArtifactStore::new();
        assert!(store.get(&Uuid::now_v7()).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn memory_artifact_store_should_preserve_binary_data() {
        let store = MemoryArtifactStore::new();
        let data: Vec<u8> = (0..=255).collect();
        let artifact = Artifact::new("binary.bin", "application/octet-stream", data.clone());
        let id = artifact.id;

        store.put(artifact).await.unwrap();
        let loaded = store.get(&id).await.unwrap().unwrap();

        assert_eq!(loaded.data, data);
        assert_eq!(loaded.content_type, "application/octet-stream");
    }

    #[tokio::test]
    async fn memory_artifact_store_should_support_metadata() {
        let store = MemoryArtifactStore::new();
        let artifact = test_artifact().with_metadata(json!({"source": "upload"}));
        let id = artifact.id;

        store.put(artifact).await.unwrap();
        let loaded = store.get(&id).await.unwrap().unwrap();

        assert_eq!(loaded.metadata, json!({"source": "upload"}));
    }
}
