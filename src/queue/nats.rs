//! NATS JetStream event publisher.
//!
//! Publishes [`AgentEvent`] values as JSON payloads to a configurable
//! JetStream subject.

use async_nats::jetstream;
use async_trait::async_trait;
use serde_json;

use super::{EventPublisher, QueueError, QueueResult};
use crate::runtime::AgentEvent;

/// Publishes `AgentEvent` values to NATS JetStream.
pub struct NatsEventPublisher {
    jetstream: jetstream::Context,
    subject: String,
}

impl NatsEventPublisher {
    /// Creates a publisher backed by an existing JetStream context.
    ///
    /// The `subject` is the NATS subject to publish events to.
    #[must_use]
    pub fn new(jetstream: jetstream::Context, subject: impl Into<String>) -> Self {
        Self {
            jetstream,
            subject: subject.into(),
        }
    }

    /// Connects to a NATS server and creates a JetStream publisher.
    ///
    /// # Errors
    ///
    /// Returns [`QueueError::ConnectionFailed`] if the NATS connection or
    /// JetStream context cannot be established.
    pub async fn connect(url: &str, subject: impl Into<String>) -> QueueResult<Self> {
        let client = async_nats::connect(url)
            .await
            .map_err(|e| QueueError::ConnectionFailed {
                message: format!("NATS connect failed: {e}"),
            })?;

        let jetstream = jetstream::new(client);

        Ok(Self {
            jetstream,
            subject: subject.into(),
        })
    }
}

#[async_trait]
impl EventPublisher for NatsEventPublisher {
    async fn publish(&self, event: AgentEvent) -> QueueResult<()> {
        let payload = serde_json::to_vec(&event).map_err(|e| QueueError::SerializationFailed {
            message: format!("failed to serialize AgentEvent: {e}"),
        })?;

        let _ack = self
            .jetstream
            .publish(self.subject.clone(), payload.into())
            .await
            .map_err(|e| QueueError::PublishFailed {
                message: format!("NATS publish failed: {e}"),
            })?;

        tracing::debug!(
            subject = %self.subject,
            "published AgentEvent to NATS JetStream",
        );

        Ok(())
    }
}
