//! PostgreSQL embedding store using pgvector for nearest-neighbor search.
#![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]

use async_trait::async_trait;
use sqlx::{Pool, Postgres};
use uuid::Uuid;

use crate::error::StorageError;
use crate::store::{EmbeddingRecord, EmbeddingStore, ScoredEmbedding, StoreResult};

/// PostgreSQL-backed embedding store using the pgvector extension for vector similarity search.
///
/// Requires the `pgvector` extension to be installed on the PostgreSQL server
/// and the `embeddings` table to exist. Implements [`EmbeddingStore`].
///
/// # Migrations
///
/// Run `003_create_embeddings.sql` from the Postgres migrations directory,
/// or use [`SqlEmbeddingStore::migrate`] to apply it programmatically.
pub struct SqlEmbeddingStore {
    pool: Pool<Postgres>,
}

impl SqlEmbeddingStore {
    /// Creates a SQL embedding store from a PostgreSQL connection pool.
    ///
    /// The pool should already be configured and connected.
    #[must_use]
    pub fn new(pool: Pool<Postgres>) -> Self {
        Self { pool }
    }

    /// Runs the embedded `pgvector` migration against the connected database.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::MigrationFailed`] when the migration fails to apply.
    pub async fn migrate(&self) -> StoreResult<()> {
        sqlx::migrate!("src/store/sql/migrations/postgres")
            .run(&self.pool)
            .await
            .map_err(|e| StorageError::MigrationFailed {
                backend: "postgres".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })
    }
}

#[async_trait]
impl EmbeddingStore for SqlEmbeddingStore {
    async fn upsert(&self, record: EmbeddingRecord) -> StoreResult<EmbeddingRecord> {
        let metadata_str =
            crate::store::util::to_json_string(&record.metadata, "embedding.metadata")?;
        let vector_str = format!(
            "[{}]",
            record
                .vector
                .iter()
                .map(f32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );

        sqlx::query(
            r"INSERT INTO embeddings (id, session_id, model, vector, metadata, created_at)
               VALUES ($1, $2, $3, $4::vector, $5, $6)
               ON CONFLICT (id) DO UPDATE SET
                   session_id = EXCLUDED.session_id,
                   model = EXCLUDED.model,
                   vector = EXCLUDED.vector,
                   metadata = EXCLUDED.metadata,
                   created_at = EXCLUDED.created_at",
        )
        .bind(record.id)
        .bind(record.session_id)
        .bind(&record.model)
        .bind(&vector_str)
        .bind(&metadata_str)
        .bind(record.created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "postgres".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        Ok(record)
    }

    async fn search(&self, query: &[f32], limit: usize) -> StoreResult<Vec<ScoredEmbedding>> {
        let vector_str = format!(
            "[{}]",
            query
                .iter()
                .map(f32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );

        let rows = sqlx::query_as::<
            _,
            (
                Uuid,
                Option<Uuid>,
                String,
                String,
                String,
                chrono::DateTime<chrono::Utc>,
                f64,
            ),
        >(
            r"SELECT e.id, e.session_id, e.model,
                      e.vector::text AS vector_text,
                      e.metadata, e.created_at,
                      1 - (e.vector <=> $1::vector) AS score
               FROM embeddings e
               ORDER BY e.vector <=> $1::vector
               LIMIT $2",
        )
        .bind(&vector_str)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "postgres".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        rows.into_iter()
            .map(
                |(id, session_id, model, vector_text, metadata, created_at, score)| {
                    let vector = parse_pg_vector(&vector_text);
                    let meta = serde_json::from_str(&metadata).unwrap_or(serde_json::Value::Null);
                    Ok(ScoredEmbedding {
                        record: EmbeddingRecord {
                            id,
                            session_id,
                            model,
                            vector,
                            metadata: meta,
                            created_at,
                        },
                        score: score as f32,
                    })
                },
            )
            .collect()
    }

    async fn delete(&self, id: &Uuid) -> StoreResult<()> {
        sqlx::query("DELETE FROM embeddings WHERE id = $1")
            .bind(*id)
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "postgres".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;
        Ok(())
    }

    async fn delete_by_session(&self, session_id: &Uuid) -> StoreResult<u64> {
        let result = sqlx::query("DELETE FROM embeddings WHERE session_id = $1")
            .bind(*session_id)
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "postgres".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;
        Ok(result.rows_affected())
    }
}

/// Parses a pgvector text representation `[1.0,2.0,3.0]` into `Vec<f32>`.
fn parse_pg_vector(text: &str) -> Vec<f32> {
    text.trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pg_vector_should_parse_standard_format() {
        let result = parse_pg_vector("[1.0,2.5,3.7]");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < f32::EPSILON);
        assert!((result[1] - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn parse_pg_vector_should_handle_empty() {
        let result = parse_pg_vector("[]");
        assert!(result.is_empty());
    }

    #[test]
    fn parse_pg_vector_should_handle_whitespace() {
        let result = parse_pg_vector("[ 1.0 , 2.0 , 3.0 ]");
        assert_eq!(result.len(), 3);
    }
}
