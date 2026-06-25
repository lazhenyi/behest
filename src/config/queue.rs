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
///
/// Supports NATS JetStream and Redis Streams backends. Only the fields
/// relevant to the selected `backend` need to be set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    /// Queue backend selection.
    pub backend: QueueBackend,

    /// NATS connection URL. Required when `backend` is `Nats`.
    #[serde(default)]
    pub nats_url: Option<String>,

    /// NATS subject for publishing events. Default: `"behest.events"`.
    #[serde(default = "default_nats_subject")]
    pub nats_subject: String,

    /// Redis connection URL. Required when `backend` is `RedisStreams`.
    #[serde(default)]
    pub redis_url: Option<String>,

    /// Redis stream key for publishing events. Default: `"behest:events"`.
    #[serde(default = "default_redis_stream_key")]
    pub redis_stream_key: String,
}

fn default_nats_subject() -> String {
    String::from("behest.events")
}

fn default_redis_stream_key() -> String {
    String::from("behest:events")
}
