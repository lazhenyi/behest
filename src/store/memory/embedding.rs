//! In-memory embedding store with brute-force cosine similarity search
//! over all stored vectors.

use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::store::{EmbeddingRecord, EmbeddingStore, ScoredEmbedding, StoreResult};

/// In-memory embedding store for testing, development, and prototyping.
///
/// Uses brute-force O(n) cosine similarity scan for nearest-neighbor search.
/// Data is lost when the process exits. Implements [`EmbeddingStore`].
#[derive(Default)]
pub struct MemoryEmbeddingStore {
    records: RwLock<HashMap<Uuid, EmbeddingRecord>>,
}

impl MemoryEmbeddingStore {
    /// Creates an empty in-memory embedding store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Computes the cosine similarity between two vectors of equal length.
///
/// Returns a value in `[-1.0, 1.0]` where `1.0` indicates identical direction,
/// `0.0` indicates orthogonality, and `-1.0` indicates opposite direction.
/// Returns `0.0` if the vectors have different lengths or either is zero-vector.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a.sqrt()) * (norm_b.sqrt());
    if denom == 0.0 { 0.0 } else { dot / denom }
}

#[async_trait]
impl EmbeddingStore for MemoryEmbeddingStore {
    async fn upsert(&self, record: EmbeddingRecord) -> StoreResult<EmbeddingRecord> {
        let mut records = self.records.write().await;
        let id = record.id;
        records.insert(id, record.clone());
        Ok(record)
    }

    async fn search(&self, query: &[f32], limit: usize) -> StoreResult<Vec<ScoredEmbedding>> {
        let records = self.records.read().await;

        let mut scored: Vec<ScoredEmbedding> = records
            .values()
            .map(|record| ScoredEmbedding {
                score: cosine_similarity(query, &record.vector),
                record: record.clone(),
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(limit);

        Ok(scored)
    }

    async fn delete(&self, id: &Uuid) -> StoreResult<()> {
        self.records.write().await.remove(id);
        Ok(())
    }

    async fn delete_by_session(&self, session_id: &Uuid) -> StoreResult<u64> {
        let mut records = self.records.write().await;
        let before = records.len();
        records.retain(|_, record| record.session_id != Some(*session_id));
        let deleted = before - records.len();
        Ok(deleted as u64)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_record(vector: Vec<f32>) -> EmbeddingRecord {
        EmbeddingRecord::new("test-model", vector)
    }

    #[tokio::test]
    async fn memory_embedding_store_should_upsert_and_search() {
        let store = MemoryEmbeddingStore::new();

        store
            .upsert(test_record(vec![1.0, 0.0, 0.0]))
            .await
            .unwrap();
        store
            .upsert(test_record(vec![0.0, 1.0, 0.0]))
            .await
            .unwrap();
        store
            .upsert(test_record(vec![0.0, 0.0, 1.0]))
            .await
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!((results[0].score - 1.0).abs() < f32::EPSILON);
        assert!(results[0].score >= results[1].score);
    }

    #[tokio::test]
    async fn memory_embedding_store_should_upsert_existing_record() {
        let store = MemoryEmbeddingStore::new();

        let record = test_record(vec![1.0, 0.0]);
        let id = record.id;
        store.upsert(record).await.unwrap();

        let updated = EmbeddingRecord {
            id,
            session_id: None,
            model: "updated-model".to_owned(),
            vector: vec![0.0, 1.0],
            metadata: json!({"updated": true}),
            created_at: chrono::Utc::now(),
        };
        store.upsert(updated).await.unwrap();

        let results = store.search(&[0.0, 1.0], 1).await.unwrap();
        assert_eq!(results[0].record.model, "updated-model");
    }

    #[tokio::test]
    async fn memory_embedding_store_should_delete_by_id() {
        let store = MemoryEmbeddingStore::new();

        let record = test_record(vec![1.0, 0.0]);
        let id = record.id;
        store.upsert(record).await.unwrap();
        store.delete(&id).await.unwrap();

        let results = store.search(&[1.0, 0.0], 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn memory_embedding_store_should_delete_by_session() {
        let store = MemoryEmbeddingStore::new();
        let session_id = Uuid::now_v7();

        store
            .upsert(test_record(vec![1.0]).with_session(session_id))
            .await
            .unwrap();
        store
            .upsert(test_record(vec![0.0, 1.0]).with_session(session_id))
            .await
            .unwrap();
        store
            .upsert(test_record(vec![0.0, 0.0, 1.0]))
            .await
            .unwrap();

        let deleted = store.delete_by_session(&session_id).await.unwrap();
        assert_eq!(deleted, 2);

        let remaining = store.search(&[1.0], 10).await.unwrap();
        assert_eq!(remaining.len(), 1);
    }

    #[tokio::test]
    async fn memory_embedding_store_should_handle_empty_search() {
        let store = MemoryEmbeddingStore::new();
        let results = store.search(&[1.0, 0.0], 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn cosine_similarity_should_return_one_for_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn cosine_similarity_should_return_zero_for_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < f32::EPSILON);
    }

    #[test]
    fn cosine_similarity_should_handle_different_lengths() {
        assert!((cosine_similarity(&[1.0], &[1.0, 2.0])).abs() < f32::EPSILON);
    }

    #[test]
    fn cosine_similarity_should_handle_zero_vectors() {
        assert!((cosine_similarity(&[0.0, 0.0], &[1.0, 0.0])).abs() < f32::EPSILON);
    }
}
