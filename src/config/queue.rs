//! Message queue (event publishing) configuration.

use serde::{Deserialize, Serialize};

/// Supported queue backends for external event publishing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QueueBackend {
    /// NATS JetStream.
    Nats,
    /// Redis Streams.
    #[serde(rename = "redis_streams", alias = "redis")]
    RedisStreams,
}

/// Configuration for external event publishing via message queues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    /// Queue backend.
    pub backend: QueueBackend,

    /// NATS connection URL.
    #[serde(default)]
    pub nats_url: Option<String>,

    /// NATS subject for publishing events.
    #[serde(default = "default_nats_subject")]
    pub nats_subject: String,

    /// Redis connection URL.
    #[serde(default)]
    pub redis_url: Option<String>,

    /// Redis stream key for publishing events.
    #[serde(default = "default_redis_stream_key")]
    pub redis_stream_key: String,
}

fn default_nats_subject() -> String {
    String::from("agents.events")
}

fn default_redis_stream_key() -> String {
    String::from("agents:events")
}
