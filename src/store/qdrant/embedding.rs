//! Qdrant vector database embedding store.
#![allow(clippy::cast_possible_truncation)]

use async_trait::async_trait;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, Distance, PointStruct, SearchPoints, VectorParamsBuilder,
};
use uuid::Uuid;

use crate::error::StorageError;
use crate::store::{EmbeddingRecord, EmbeddingStore, ScoredEmbedding, StoreResult};

/// Qdrant-backed embedding store with native vector similarity search.
pub struct QdrantEmbeddingStore {
    client: Qdrant,
    collection: String,
    dimensions: u64,
}

impl QdrantEmbeddingStore {
    /// Creates a Qdrant embedding store.
    #[must_use]
    pub fn new(client: Qdrant, collection: String, dimensions: u64) -> Self {
        Self {
            client,
            collection,
            dimensions,
        }
    }

    /// Ensures the collection exists, creating it if necessary.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::BackendError`] when the operation fails.
    pub async fn ensure_collection(&self) -> StoreResult<()> {
        let exists = self
            .client
            .collection_exists(&self.collection)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "qdrant".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        if !exists {
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.collection).vectors_config(
                        VectorParamsBuilder::new(self.dimensions, Distance::Cosine),
                    ),
                )
                .await
                .map_err(|e| StorageError::BackendError {
                    backend: "qdrant".to_owned(),
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;
        }

        Ok(())
    }
}

fn record_to_payload(
    record: &EmbeddingRecord,
) -> StoreResult<(
    Vec<f32>,
    std::collections::HashMap<String, qdrant_client::qdrant::Value>,
)> {
    use qdrant_client::qdrant::Value;

    let mut payload = std::collections::HashMap::new();
    payload.insert("id".to_owned(), Value::from(record.id.to_string()));
    payload.insert("model".to_owned(), Value::from(record.model.as_str()));
    if let Some(sid) = record.session_id {
        payload.insert("session_id".to_owned(), Value::from(sid.to_string()));
    }
    payload.insert(
        "metadata".to_owned(),
        Value::from(crate::store::util::to_json_string(
            &record.metadata,
            "embedding.metadata",
        )?),
    );
    payload.insert(
        "created_at".to_owned(),
        Value::from(record.created_at.to_rfc3339()),
    );

    Ok((record.vector.clone(), payload))
}

fn scored_to_record(point: &qdrant_client::qdrant::ScoredPoint) -> Option<ScoredEmbedding> {
    let payload = &point.payload;

    let id_str = payload.get("id")?.as_str()?;
    let id = Uuid::parse_str(id_str).ok()?;
    let model = payload.get("model")?.as_str()?.to_owned();
    let session_id = payload
        .get("session_id")
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok());
    let metadata_str = payload
        .get("metadata")
        .and_then(|v| v.as_str())
        .map_or("{}", |v| v);
    let metadata = serde_json::from_str(metadata_str).unwrap_or(serde_json::Value::Null);
    let created_at_str = payload
        .get("created_at")
        .and_then(|v| v.as_str())
        .map_or("", |v| v);
    let created_at = chrono::DateTime::parse_from_rfc3339(created_at_str)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or_default();

    // Extract vector from the point
    let vector = if let Some(vectors) = &point.vectors {
        if let Some(qdrant_client::qdrant::vectors_output::VectorsOptions::Vector(v)) =
            &vectors.vectors_options
        {
            #[allow(deprecated)]
            v.data.clone()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    Some(ScoredEmbedding {
        record: EmbeddingRecord {
            id,
            session_id,
            model,
            vector,
            metadata,
            created_at,
        },
        score: point.score,
    })
}

#[async_trait]
impl EmbeddingStore for QdrantEmbeddingStore {
    async fn upsert(&self, record: EmbeddingRecord) -> StoreResult<EmbeddingRecord> {
        use qdrant_client::qdrant::UpsertPoints;

        let (vector, payload) = record_to_payload(&record)?;
        let point_id = record.id.as_u128().to_string();

        let point = PointStruct::new(point_id, vector, payload);

        self.client
            .upsert_points(UpsertPoints {
                collection_name: self.collection.clone(),
                points: vec![point],
                ..Default::default()
            })
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "qdrant".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        Ok(record)
    }

    async fn search(&self, query: &[f32], limit: usize) -> StoreResult<Vec<ScoredEmbedding>> {
        let results = self
            .client
            .search_points(SearchPoints {
                collection_name: self.collection.clone(),
                vector: query.to_vec(),
                limit: limit as u64,
                with_payload: Some(true.into()),
                with_vectors: Some(true.into()),
                ..Default::default()
            })
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "qdrant".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        Ok(results.result.iter().filter_map(scored_to_record).collect())
    }

    async fn delete(&self, id: &Uuid) -> StoreResult<()> {
        use qdrant_client::qdrant::DeletePoints;

        let point_id = id.as_u128().to_string();

        self.client
            .delete_points(DeletePoints {
                collection_name: self.collection.clone(),
                points: Some(vec![qdrant_client::qdrant::PointId::from(point_id)].into()),
                ..Default::default()
            })
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "qdrant".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        Ok(())
    }

    async fn delete_by_session(&self, session_id: &Uuid) -> StoreResult<u64> {
        use qdrant_client::qdrant::{DeletePoints, Filter};

        let filter = Filter::must([Condition::matches("session_id", session_id.to_string())]);

        // Qdrant doesn't return count from delete, so we search first
        let results = self
            .client
            .search_points(SearchPoints {
                collection_name: self.collection.clone(),
                vector: vec![0.0; self.dimensions as usize],
                limit: 10000,
                filter: Some(filter.clone()),
                with_payload: Some(false.into()),
                ..Default::default()
            })
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "qdrant".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let count = results.result.len() as u64;

        if count > 0 {
            self.client
                .delete_points(DeletePoints {
                    collection_name: self.collection.clone(),
                    wait: None,
                    ordering: None,
                    points: Some(
                        Filter::must([Condition::matches("session_id", session_id.to_string())])
                            .into(),
                    ),
                    ..Default::default()
                })
                .await
                .map_err(|e| StorageError::BackendError {
                    backend: "qdrant".to_owned(),
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;
        }

        Ok(count)
    }
}
