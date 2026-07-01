//! Redis Streams event publisher.
//!
//! Publishes [`AgentEvent`] values as JSON payloads to a Redis stream
//! via the `XADD` command.

use async_trait::async_trait;
use redis::AsyncCommands;
use redis::aio::MultiplexedConnection;
use serde_json;

use super::{EventPublisher, QueueError, QueueResult};
use crate::runtime::AgentEvent;

/// Publishes `AgentEvent` values to a Redis stream via `XADD`.
///
/// Each event is serialized as JSON and appended to the configured
/// stream key. Useful for audit logging and event sourcing pipelines.
pub struct RedisStreamsPublisher {
    conn: MultiplexedConnection,
    stream_key: String,
}

impl RedisStreamsPublisher {
    /// Creates a publisher from an existing Redis multiplexed connection.
    ///
    /// # Parameters
    /// - `conn` – an established Redis multiplexed async connection.
    /// - `stream_key` – the Redis stream key to `XADD` events to.
    #[must_use]
    pub fn new(conn: MultiplexedConnection, stream_key: impl Into<String>) -> Self {
        Self {
            conn,
            stream_key: stream_key.into(),
        }
    }

    /// Connects to a Redis instance and returns a stream publisher.
    ///
    /// # Parameters
    /// - `url` – Redis connection URL (e.g. `redis://localhost:6379`).
    /// - `stream_key` – the Redis stream key to `XADD` events to.
    ///
    /// # Errors
    ///
    /// Returns [`QueueError::ConnectionFailed`] if the Redis client creation
    /// or the async connection handshake fails.
    pub async fn connect(url: &str, stream_key: impl Into<String>) -> QueueResult<Self> {
        let client = redis::Client::open(url).map_err(|e| QueueError::ConnectionFailed {
            message: format!("Redis client creation failed: {e}"),
        })?;

        let conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| QueueError::ConnectionFailed {
                message: format!("Redis async connection failed: {e}"),
            })?;

        Ok(Self {
            conn,
            stream_key: stream_key.into(),
        })
    }

    /// Returns a reference to the underlying stream key.
    #[must_use]
    pub fn stream_key(&self) -> &str {
        &self.stream_key
    }
}

#[async_trait]
impl EventPublisher for RedisStreamsPublisher {
    async fn publish(&self, event: AgentEvent) -> QueueResult<()> {
        let payload =
            serde_json::to_string(&event).map_err(|e| QueueError::SerializationFailed {
                message: format!("failed to serialize AgentEvent: {e}"),
            })?;

        let mut conn = self.conn.clone();

        let entry_id: String = conn
            .xadd(&self.stream_key, "*", &[("event", &payload)])
            .await
            .map_err(|e| QueueError::PublishFailed {
                message: format!("Redis XADD failed: {e}"),
            })?;

        tracing::debug!(
            stream_key = %self.stream_key,
            entry_id = %entry_id,
            "published AgentEvent to Redis Stream",
        );

        Ok(())
    }
}

#[async_trait]
impl crate::runtime::RuntimeEventPublisher for RedisStreamsPublisher {
    async fn publish(&self, event: AgentEvent) -> Result<(), crate::runtime::EventPublishError> {
        <Self as EventPublisher>::publish(self, event)
            .await
            .map_err(|e| crate::runtime::EventPublishError::PublishFailed {
                message: e.to_string(),
            })
    }
}
