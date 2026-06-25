//! Redis Streams-backed [`RuntimeEventStore`].
//!
//! Each `run_id` maps to a Redis Stream key `run:{run_id}`. [`RuntimeEventStore::append`] uses
//! `XADD` with an auto-incrementing `*` id and stores the serialized envelope
//! in a single field `data`. [`RuntimeEventStore::list_after`] uses `XRANGE` with an optional
//! lower-bound id derived from `after_seq`.
//!
//! The store relies on Redis for both durability and ordering; it does not
//! maintain local counters. This makes it suitable for multi-instance
//! deployments where a shared Redis instance serves as the authoritative
//! event log.

use async_trait::async_trait;
use chrono::Utc;
use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use redis::streams::{StreamMaxlen, StreamRangeReply};
use serde_json;

use crate::runtime::event::AgentEvent;
use crate::runtime::run::RunId;
use crate::runtime::stream::{RuntimeEventEnvelope, RuntimeEventId};

use crate::runtime::event_store::{RuntimeEventStore, RuntimeEventStoreError};

/// Redis Streams event store key prefix.
const STREAM_KEY_PREFIX: &str = "run";

/// Maximum number of events retained per run stream (approximate).
///
/// `MAXLEN ~` provides approximate trimming; exact count is not required
/// for correctness because the store is the replay source, not the primary
/// state store.
const STREAM_MAXLEN: usize = 10_000;

/// Redis Streams-backed [`RuntimeEventStore`].
///
/// # Redis Stream key layout
///
/// - `run:{run_id}` — Stream of serialized [`RuntimeEventEnvelope`]s.
/// - `run:{run_id}:seq` — String counter for the per-run monotonic sequence.
/// - `run:{run_id}:session` — String holding the cached session id.
#[derive(Clone)]
pub struct RedisRuntimeEventStore {
    conn: ConnectionManager,
}

impl RedisRuntimeEventStore {
    /// Creates a new Redis-backed event store.
    ///
    /// `conn` must be a [`redis::aio::ConnectionManager`] connected to a
    /// Redis instance with Streams support (Redis >= 5.0).
    #[must_use]
    pub fn new(conn: ConnectionManager) -> Self {
        Self { conn }
    }

    fn stream_key(run_id: RunId) -> String {
        format!("{STREAM_KEY_PREFIX}:{run_id}")
    }

    fn seq_key(run_id: RunId) -> String {
        format!("{STREAM_KEY_PREFIX}:{run_id}:seq")
    }

    fn session_key(run_id: RunId) -> String {
        format!("{STREAM_KEY_PREFIX}:{run_id}:session")
    }
}

#[async_trait]
impl RuntimeEventStore for RedisRuntimeEventStore {
    async fn append(
        &self,
        event: AgentEvent,
    ) -> Result<RuntimeEventEnvelope, RuntimeEventStoreError> {
        let run_id = event.run_id();
        let mut conn = self.conn.clone();

        let session_id = if let AgentEvent::RunStarted(started) = &event {
            let sid = started.session_id;
            let _: () = conn
                .set(Self::session_key(run_id), sid.to_string())
                .await
                .map_err(|e| RuntimeEventStoreError::Append {
                    message: format!("failed to cache session: {e}"),
                })?;
            Some(sid)
        } else {
            let raw: Option<String> = conn.get(Self::session_key(run_id)).await.map_err(|e| {
                RuntimeEventStoreError::Append {
                    message: format!("failed to read session: {e}"),
                }
            })?;
            raw.and_then(|s| uuid::Uuid::parse_str(&s).ok())
        };

        let next_seq: u64 = conn.incr(Self::seq_key(run_id), 1).await.map_err(|e| {
            RuntimeEventStoreError::Append {
                message: format!("failed to increment seq: {e}"),
            }
        })?;

        let envelope = RuntimeEventEnvelope {
            event_id: RuntimeEventId::new(),
            seq: next_seq,
            run_id,
            session_id,
            event,
            emitted_at: Utc::now(),
        };

        let payload =
            serde_json::to_string(&envelope).map_err(|e| RuntimeEventStoreError::Append {
                message: format!("failed to serialize envelope: {e}"),
            })?;

        let _: String = conn
            .xadd_maxlen(
                Self::stream_key(run_id),
                StreamMaxlen::Approx(STREAM_MAXLEN),
                "*",
                &[("data", payload.as_str())],
            )
            .await
            .map_err(|e| RuntimeEventStoreError::Append {
                message: format!("XADD failed: {e}"),
            })?;

        Ok(envelope)
    }

    async fn list_after(
        &self,
        run_id: RunId,
        after_seq: Option<u64>,
        limit: usize,
    ) -> Result<Vec<RuntimeEventEnvelope>, RuntimeEventStoreError> {
        let mut conn = self.conn.clone();
        let stream_key = Self::stream_key(run_id);

        let start = after_seq.map_or_else(|| "-".to_string(), |seq| format!("{seq}"));

        let raw: StreamRangeReply = conn
            .xrange_count(stream_key, start, "+", limit)
            .await
            .map_err(|e| RuntimeEventStoreError::Append {
                message: format!("XRANGE failed: {e}"),
            })?;

        let mut envelopes = Vec::with_capacity(raw.ids.len());
        for stream_id in raw.ids {
            match stream_id.get::<String>("data") {
                Some(data) => match serde_json::from_str::<RuntimeEventEnvelope>(&data) {
                    Ok(env) => {
                        if after_seq.is_none_or(|seq| env.seq > seq) {
                            envelopes.push(env);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            run_id = %run_id,
                            error = %e,
                            "failed to deserialize runtime event envelope from Redis stream"
                        );
                    }
                },
                None => {
                    tracing::warn!(
                        run_id = %run_id,
                        "redis stream entry missing 'data' field"
                    );
                }
            }
        }

        Ok(envelopes)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::provider::{ModelName, ProviderId};
    use crate::runtime::event::RunStarted;
    use chrono::Utc;
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
    #[ignore = "requires a running Redis instance"]
    async fn append_and_list_after_redis() {
        let client = redis::Client::open("redis://127.0.0.1:6379/").expect("redis client");
        let conn = ConnectionManager::new(client)
            .await
            .expect("connection manager");
        let store = RedisRuntimeEventStore::new(conn);
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
    #[ignore = "requires a running Redis instance"]
    async fn list_after_unknown_run_returns_empty() {
        let client = redis::Client::open("redis://127.0.0.1:6379/").expect("redis client");
        let conn = ConnectionManager::new(client)
            .await
            .expect("connection manager");
        let store = RedisRuntimeEventStore::new(conn);
        let run = RunId::new();

        let page = store.list_after(run, None, 10).await.unwrap();
        assert!(page.is_empty());
    }
}
