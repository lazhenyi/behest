//! External event publishing boundary for runtime events.

use async_trait::async_trait;
use thiserror::Error;

use crate::event::AgentEvent;

/// Errors produced by runtime event publishers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EventPublishError {
    /// Publishing a single event to the broker failed.
    #[error("runtime event publish failed: {message}")]
    PublishFailed {
        /// Human-readable failure description.
        message: String,
    },
}

/// Publishes agent events to an external message broker.
#[async_trait]
pub trait EventPublisher: Send + Sync {
    /// Publishes an agent event.
    ///
    /// # Errors
    ///
    /// Returns [`EventPublishError`] when the event cannot be delivered.
    async fn publish(&self, event: AgentEvent) -> Result<(), EventPublishError>;
}
