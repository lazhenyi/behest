//! External event publishing for agent observability.
//!
//! The [`EventPublisher`] trait allows `AgentRuntime` to publish
//! every [`AgentEvent`] to an external message broker (NATS JetStream,
//! Redis Streams, etc.) for downstream consumers such as dashboards,
//! audit logs, or multi‑agent coordination.

use async_trait::async_trait;
use thiserror::Error;

use crate::runtime::AgentEvent;

/// Result type for event publishing operations.
pub type QueueResult<T> = Result<T, QueueError>;

/// Errors produced by event publishers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum QueueError {
    /// Failed to establish a connection to the broker.
    #[error("queue connection failed: {message}")]
    ConnectionFailed {
        /// Human-readable failure description.
        message: String,
    },

    /// Publishing a single event to the broker failed.
    #[error("queue publish failed: {message}")]
    PublishFailed {
        /// Human-readable failure description.
        message: String,
    },

    /// Event serialization failed before publishing.
    #[error("queue serialization failed: {message}")]
    SerializationFailed {
        /// Human-readable failure description.
        message: String,
    },
}

/// Publishes agent events to an external message broker.
///
/// Implementations must be `Send + Sync` so they can be shared across
/// tasks inside `AgentRuntime`.
#[async_trait]
pub trait EventPublisher: Send + Sync {
    /// Publish an agent event to the broker.
    ///
    /// # Errors
    ///
    /// Returns [`QueueError`] when connectivity, serialization, or
    /// broker-side failures prevent the event from being delivered.
    async fn publish(&self, event: AgentEvent) -> QueueResult<()>;
}

#[cfg(feature = "nats")]
mod nats;

#[cfg(feature = "redis")]
mod redis_streams;

#[cfg(feature = "nats")]
pub use nats::NatsEventPublisher;

#[cfg(feature = "redis")]
pub use redis_streams::RedisStreamsPublisher;

/// A no‑op publisher for testing and opt‑out scenarios.
///
/// Accepts all events silently without connecting to any broker.
/// Useful as a default implementation or for disabling event publishing
/// in configurations where observability is not required.
pub struct NoOpPublisher;

impl NoOpPublisher {
    /// Creates a no‑op publisher that discards all events.
    ///
    /// This is a zero-allocation operation.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoOpPublisher {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EventPublisher for NoOpPublisher {
    async fn publish(&self, _event: AgentEvent) -> QueueResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    use crate::runtime::RunId;
    use crate::runtime::event::RunStarted;

    #[tokio::test]
    async fn no_op_publisher_should_accept_any_event() {
        let publisher = NoOpPublisher::new();
        let event = AgentEvent::RunStarted(RunStarted {
            run_id: RunId::new(),
            session_id: Uuid::new_v4(),
            provider: crate::provider::ProviderId::new("test"),
            model: crate::provider::ModelName::new("test"),
            timestamp: chrono::Utc::now(),
        });
        let result = publisher.publish(event).await;
        assert!(result.is_ok());
    }
}
