//! Artifact store implementations using the `object_store` crate.
//!
//! Provides [`DiskArtifactStore`] for local filesystem storage and
//! [`S3ArtifactStore`] for Amazon S3-compatible storage.

use async_trait::async_trait;
use bytes::Bytes;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use uuid::Uuid;

use crate::error::StorageError;
use crate::store::{Artifact, ArtifactStore, StoreResult};

/// Local filesystem artifact store backed by `object_store::local::LocalFileSystem`.
///
/// Artifacts are stored as two files per entry: the raw binary data under
/// `artifacts/{id}` and the serialized metadata under `metadata/{id}.json`.
pub struct DiskArtifactStore {
    store: object_store::local::LocalFileSystem,
    #[allow(dead_code)]
    prefix: String,
}

impl DiskArtifactStore {
    /// Creates a disk artifact store rooted at the given directory.
    ///
    /// The directory is created if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::ConnectionFailed`] when the directory cannot be created
    /// or the local filesystem adapter cannot be initialized.
    pub fn new(root: impl Into<std::path::PathBuf>) -> StoreResult<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(|e| StorageError::ConnectionFailed {
            backend: "disk".to_owned(),
            message: format!("failed to create directory {}: {}", root.display(), e),
            source: Some(Box::new(e)),
        })?;

        let store = object_store::local::LocalFileSystem::new_with_prefix(&root).map_err(|e| {
            StorageError::ConnectionFailed {
                backend: "disk".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        Ok(Self {
            store,
            prefix: root.to_string_lossy().into_owned(),
        })
    }

    fn artifact_path(id: &Uuid) -> ObjectPath {
        ObjectPath::from(format!("artifacts/{id}"))
    }

    fn metadata_path(id: &Uuid) -> ObjectPath {
        ObjectPath::from(format!("metadata/{id}.json"))
    }
}

#[async_trait]
impl ArtifactStore for DiskArtifactStore {
    async fn put(&self, artifact: Artifact) -> StoreResult<Artifact> {
        let data_path = Self::artifact_path(&artifact.id);
        let meta_path = Self::metadata_path(&artifact.id);

        // Store binary data
        let payload = PutPayload::from_bytes(Bytes::from(artifact.data.clone()));
        self.store
            .put(&data_path, payload)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "disk".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        // Store metadata as JSON
        let meta_json =
            serde_json::to_vec(&artifact).map_err(|e| StorageError::SerializationFailed {
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;
        let payload = PutPayload::from_bytes(Bytes::from(meta_json));
        self.store
            .put(&meta_path, payload)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "disk".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        Ok(artifact)
    }

    async fn get(&self, id: &Uuid) -> StoreResult<Option<Artifact>> {
        let meta_path = Self::metadata_path(id);

        let result = self.store.get(&meta_path).await;
        match result {
            Ok(meta_bytes) => {
                let meta_data =
                    meta_bytes
                        .bytes()
                        .await
                        .map_err(|e| StorageError::BackendError {
                            backend: "disk".to_owned(),
                            message: e.to_string(),
                            source: Some(Box::new(e)),
                        })?;

                let mut artifact: Artifact = serde_json::from_slice(&meta_data).map_err(|e| {
                    StorageError::SerializationFailed {
                        message: e.to_string(),
                        source: Some(Box::new(e)),
                    }
                })?;

                // Load actual binary data
                let data_path = Self::artifact_path(id);
                let data_bytes =
                    self.store
                        .get(&data_path)
                        .await
                        .map_err(|e| StorageError::BackendError {
                            backend: "disk".to_owned(),
                            message: e.to_string(),
                            source: Some(Box::new(e)),
                        })?;

                artifact.data = data_bytes
                    .bytes()
                    .await
                    .map_err(|e| StorageError::BackendError {
                        backend: "disk".to_owned(),
                        message: e.to_string(),
                        source: Some(Box::new(e)),
                    })?
                    .to_vec();

                Ok(Some(artifact))
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(StorageError::BackendError {
                backend: "disk".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            }),
        }
    }

    async fn delete(&self, id: &Uuid) -> StoreResult<()> {
        let data_path = Self::artifact_path(id);
        let meta_path = Self::metadata_path(id);

        let _ = self.store.delete(&data_path).await;
        let _ = self.store.delete(&meta_path).await;
        Ok(())
    }

    async fn list_by_session(&self, session_id: &Uuid) -> StoreResult<Vec<Artifact>> {
        use futures_util::TryStreamExt;

        let prefix = ObjectPath::from("metadata/");
        let mut artifacts = Vec::new();

        let stream = self.store.list(Some(&prefix));
        let entries: Vec<_> =
            stream
                .try_collect()
                .await
                .map_err(|e| StorageError::BackendError {
                    backend: "disk".to_owned(),
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;

        for entry in entries {
            if let Ok(meta_bytes) = self.store.get(&entry.location).await
                && let Ok(data) = meta_bytes.bytes().await
                && let Ok(artifact) = serde_json::from_slice::<Artifact>(&data)
                && artifact.session_id == Some(*session_id)
            {
                artifacts.push(artifact);
            }
        }

        Ok(artifacts)
    }
}

/// Amazon S3-compatible artifact store backed by `object_store::aws::AmazonS3`.
///
/// Artifacts are stored as two objects per entry: the raw binary data
/// under `artifacts/{id}` and serialized metadata under `metadata/{id}.json`.
pub struct S3ArtifactStore {
    store: object_store::aws::AmazonS3,
}

impl S3ArtifactStore {
    /// Creates an S3 artifact store from a pre-configured `AmazonS3` instance.
    ///
    /// The caller is responsible for configuring the S3 client (region,
    /// credentials, endpoint, bucket, etc.).
    #[must_use]
    pub fn new(store: object_store::aws::AmazonS3) -> Self {
        Self { store }
    }

    fn artifact_path(id: &Uuid) -> ObjectPath {
        ObjectPath::from(format!("artifacts/{id}"))
    }

    fn metadata_path(id: &Uuid) -> ObjectPath {
        ObjectPath::from(format!("metadata/{id}.json"))
    }
}

#[async_trait]
impl ArtifactStore for S3ArtifactStore {
    async fn put(&self, artifact: Artifact) -> StoreResult<Artifact> {
        let data_path = Self::artifact_path(&artifact.id);
        let meta_path = Self::metadata_path(&artifact.id);

        let payload = PutPayload::from_bytes(Bytes::from(artifact.data.clone()));
        self.store
            .put(&data_path, payload)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "s3".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let meta_json =
            serde_json::to_vec(&artifact).map_err(|e| StorageError::SerializationFailed {
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;
        let payload = PutPayload::from_bytes(Bytes::from(meta_json));
        self.store
            .put(&meta_path, payload)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "s3".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        Ok(artifact)
    }

    async fn get(&self, id: &Uuid) -> StoreResult<Option<Artifact>> {
        let meta_path = Self::metadata_path(id);

        match self.store.get(&meta_path).await {
            Ok(meta_bytes) => {
                let meta_data =
                    meta_bytes
                        .bytes()
                        .await
                        .map_err(|e| StorageError::BackendError {
                            backend: "s3".to_owned(),
                            message: e.to_string(),
                            source: Some(Box::new(e)),
                        })?;

                let mut artifact: Artifact = serde_json::from_slice(&meta_data).map_err(|e| {
                    StorageError::SerializationFailed {
                        message: e.to_string(),
                        source: Some(Box::new(e)),
                    }
                })?;

                let data_path = Self::artifact_path(id);
                let data_bytes =
                    self.store
                        .get(&data_path)
                        .await
                        .map_err(|e| StorageError::BackendError {
                            backend: "s3".to_owned(),
                            message: e.to_string(),
                            source: Some(Box::new(e)),
                        })?;

                artifact.data = data_bytes
                    .bytes()
                    .await
                    .map_err(|e| StorageError::BackendError {
                        backend: "s3".to_owned(),
                        message: e.to_string(),
                        source: Some(Box::new(e)),
                    })?
                    .to_vec();

                Ok(Some(artifact))
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(StorageError::BackendError {
                backend: "s3".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            }),
        }
    }

    async fn delete(&self, id: &Uuid) -> StoreResult<()> {
        let data_path = Self::artifact_path(id);
        let meta_path = Self::metadata_path(id);

        let _ = self.store.delete(&data_path).await;
        let _ = self.store.delete(&meta_path).await;
        Ok(())
    }

    async fn list_by_session(&self, session_id: &Uuid) -> StoreResult<Vec<Artifact>> {
        use futures_util::TryStreamExt;

        let prefix = ObjectPath::from("metadata/");
        let mut artifacts = Vec::new();

        let stream = self.store.list(Some(&prefix));
        let entries: Vec<_> =
            stream
                .try_collect()
                .await
                .map_err(|e| StorageError::BackendError {
                    backend: "s3".to_owned(),
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;

        for entry in entries {
            if let Ok(meta_bytes) = self.store.get(&entry.location).await
                && let Ok(data) = meta_bytes.bytes().await
                && let Ok(artifact) = serde_json::from_slice::<Artifact>(&data)
                && artifact.session_id == Some(*session_id)
            {
                artifacts.push(artifact);
            }
        }

        Ok(artifacts)
    }
}
