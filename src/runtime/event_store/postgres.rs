//! PostgreSQL-backed [`RuntimeEventStore`].
//!
//! Uses a `runtime_events` table with a composite primary key `(run_id, seq)`.
//! [`append`] uses `INSERT ... RETURNING seq` with a subquery for the next
//! sequence number. [`list_after`] uses `SELECT ... WHERE run_id = $1 AND seq > $2
//! ORDER BY seq LIMIT $3`.
//!
//! Session tracking is handled via a separate `runtime_sessions` table,
//! upserted on `RunStarted`.

use async_trait::async_trait;
use chrono::Utc;
use sqlx::PgPool;

use crate::runtime::event::AgentEvent;
use crate::runtime::run::RunId;
use crate::runtime::stream::{RuntimeEventEnvelope, RuntimeEventId};

use crate::runtime::event_store::{RuntimeEventStore, RuntimeEventStoreError};

/// PostgreSQL-backed [`RuntimeEventStore`].
///
/// # Table layout
///
/// ```sql
/// CREATE TABLE runtime_events (
///     run_id UUID NOT NULL,
///     seq BIGINT NOT NULL,
///     event_id UUID NOT NULL,
///     session_id UUID,
///     event JSONB NOT NULL,
///     emitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
///     PRIMARY KEY (run_id, seq)
/// );
///
/// CREATE TABLE runtime_sessions (
///     run_id UUID PRIMARY KEY,
///     session_id UUID NOT NULL
/// );
/// ```
///
/// The caller is responsible for creating these tables before using the store.
#[derive(Clone)]
pub struct PostgresRuntimeEventStore {
    pool: PgPool,
}

impl PostgresRuntimeEventStore {
    /// Creates a new PostgreSQL-backed event store.
    ///
    /// `pool` must be a [`sqlx::PgPool`] connected to a PostgreSQL database.
    #[must_use]
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl RuntimeEventStore for PostgresRuntimeEventStore {
    async fn append(
        &self,
        event: AgentEvent,
    ) -> Result<RuntimeEventEnvelope, RuntimeEventStoreError> {
        let run_id = event.run_id();
        let event_id = RuntimeEventId::new();
        let emitted_at = Utc::now();

        let session_id = if let AgentEvent::RunStarted(started) = &event {
            let sid = started.session_id;
            sqlx::query(
                "INSERT INTO runtime_sessions (run_id, session_id) VALUES ($1, $2) \
                 ON CONFLICT (run_id) DO UPDATE SET session_id = EXCLUDED.session_id",
            )
            .bind(run_id.as_uuid())
            .bind(sid)
            .execute(&self.pool)
            .await
            .map_err(|e| RuntimeEventStoreError::Append {
                message: format!("failed to upsert session: {e}"),
            })?;
            Some(sid)
        } else {
            sqlx::query_scalar::<_, uuid::Uuid>(
                "SELECT session_id FROM runtime_sessions WHERE run_id = $1",
            )
            .bind(run_id.as_uuid())
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| RuntimeEventStoreError::Append {
                message: format!("failed to read session: {e}"),
            })?
        };

        let event_json =
            serde_json::to_value(&event).map_err(|e| RuntimeEventStoreError::Append {
                message: format!("failed to serialize event: {e}"),
            })?;

        let seq: i64 = sqlx::query_scalar(
            "INSERT INTO runtime_events (run_id, seq, event_id, session_id, event, emitted_at) \
             VALUES ($1, COALESCE((SELECT MAX(seq) FROM runtime_events WHERE run_id = $1), 0) + 1, $2, $3, $4, $5) \
             RETURNING seq",
        )
        .bind(run_id.as_uuid())
        .bind(event_id.as_uuid())
        .bind(session_id)
        .bind(&event_json)
        .bind(emitted_at)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| RuntimeEventStoreError::Append {
            message: format!("INSERT failed: {e}"),
        })?;

        Ok(RuntimeEventEnvelope {
            event_id,
            seq: u64::try_from(seq).map_err(|e| RuntimeEventStoreError::Append {
                message: format!("seq out of range: {e}"),
            })?,
            run_id,
            session_id,
            event,
            emitted_at,
        })
    }

    async fn list_after(
        &self,
        run_id: RunId,
        after_seq: Option<u64>,
        limit: usize,
    ) -> Result<Vec<RuntimeEventEnvelope>, RuntimeEventStoreError> {
        let rows = if let Some(after) = after_seq {
            sqlx::query_as::<_, EventRow>(
                "SELECT run_id, seq, event_id, session_id, event, emitted_at \
                 FROM runtime_events \
                 WHERE run_id = $1 AND seq > $2 \
                 ORDER BY seq \
                 LIMIT $3",
            )
            .bind(run_id.as_uuid())
            .bind(
                i64::try_from(after).map_err(|e| RuntimeEventStoreError::Append {
                    message: format!("after_seq out of range: {e}"),
                })?,
            )
            .bind(
                i64::try_from(limit).map_err(|e| RuntimeEventStoreError::Append {
                    message: format!("limit out of range: {e}"),
                })?,
            )
            .fetch_all(&self.pool)
            .await
        } else {
            sqlx::query_as::<_, EventRow>(
                "SELECT run_id, seq, event_id, session_id, event, emitted_at \
                 FROM runtime_events \
                 WHERE run_id = $1 \
                 ORDER BY seq \
                 LIMIT $2",
            )
            .bind(run_id.as_uuid())
            .bind(
                i64::try_from(limit).map_err(|e| RuntimeEventStoreError::Append {
                    message: format!("limit out of range: {e}"),
                })?,
            )
            .fetch_all(&self.pool)
            .await
        }
        .map_err(|e| RuntimeEventStoreError::Append {
            message: format!("SELECT failed: {e}"),
        })?;

        let mut envelopes = Vec::with_capacity(rows.len());
        for row in rows {
            let event: AgentEvent =
                serde_json::from_value(row.event).map_err(|e| RuntimeEventStoreError::Append {
                    message: format!("failed to deserialize event: {e}"),
                })?;
            envelopes.push(RuntimeEventEnvelope {
                event_id: RuntimeEventId::from_uuid(row.event_id),
                seq: u64::try_from(row.seq).map_err(|e| RuntimeEventStoreError::Append {
                    message: format!("seq out of range: {e}"),
                })?,
                run_id: RunId::from_uuid(row.run_id),
                session_id: row.session_id,
                event,
                emitted_at: row.emitted_at,
            });
        }

        Ok(envelopes)
    }
}

#[derive(sqlx::FromRow)]
struct EventRow {
    run_id: uuid::Uuid,
    seq: i64,
    event_id: uuid::Uuid,
    session_id: Option<uuid::Uuid>,
    event: serde_json::Value,
    emitted_at: chrono::DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::provider::{ModelName, ProviderId};
    use crate::runtime::event::RunStarted;
    use uuid::Uuid;

    fn started(run_id: RunId, session_id: Uuid) -> AgentEvent {
        AgentEvent::RunStarted(RunStarted {
            run_id,
            session_id,
            provider: ProviderId::new("acme"),
            model: ModelName::new("gpt-test"),
            timestamp: Utc::now(),
        })
    }

    #[tokio::test]
    #[ignore = "requires a running PostgreSQL instance"]
    async fn append_and_list_after_postgres() {
        let pool = PgPool::connect("postgres://localhost/behest_test")
            .await
            .expect("pg pool");

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS runtime_events (
                run_id UUID NOT NULL,
                seq BIGINT NOT NULL,
                event_id UUID NOT NULL,
                session_id UUID,
                event JSONB NOT NULL,
                emitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (run_id, seq)
            )",
        )
        .execute(&pool)
        .await
        .expect("create runtime_events");

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS runtime_sessions (
                run_id UUID PRIMARY KEY,
                session_id UUID NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .expect("create runtime_sessions");

        let store = PostgresRuntimeEventStore::new(pool);
        let run = RunId::new();
        let sid = Uuid::now_v7();

        let env = store.append(started(run, sid)).await.unwrap();
        assert_eq!(env.seq, 1);
        assert_eq!(env.session_id, Some(sid));

        let page = store.list_after(run, None, 10).await.unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].seq, 1);
    }

    #[tokio::test]
    #[ignore = "requires a running PostgreSQL instance"]
    async fn list_after_unknown_run_returns_empty() {
        let pool = PgPool::connect("postgres://localhost/behest_test")
            .await
            .expect("pg pool");
        let store = PostgresRuntimeEventStore::new(pool);
        let run = RunId::new();

        let page = store.list_after(run, None, 10).await.unwrap();
        assert!(page.is_empty());
    }
}
