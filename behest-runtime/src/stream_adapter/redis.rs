//! Redis Pub/Sub-backed [`RuntimeStreamAdapter`].
//!
//! Each [`RuntimeRoom`] maps to a Redis Pub/Sub channel. [`RuntimeStreamAdapter::publish`] uses
//! `PUBLISH` with a JSON-serialized envelope. [`RuntimeStreamAdapter::subscribe`] uses `SUBSCRIBE`
//! and bridges into a [`BoxRuntimeEventStream`].
//!
//! This adapter is best-effort only: Redis Pub/Sub is fire-and-forget with
//! no replay. Durability and replay are the responsibility of
//! [`RuntimeEventStore`](crate::runtime::event_store::RuntimeEventStore).

use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use redis::Msg;
use redis::aio::{ConnectionManager, PubSub};
use serde_json;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::warn;

use crate::stream::{BoxRuntimeEventStream, RuntimeEventEnvelope, RuntimeRoom, RuntimeStreamError};

use crate::stream_adapter::RuntimeStreamAdapter;

/// Redis Pub/Sub channel prefix for runtime event rooms.
const CHANNEL_PREFIX: &str = "rt";

/// Capacity for the internal MPSC bridge channel.
const BRIDGE_CAPACITY: usize = 256;

/// Redis Pub/Sub-backed [`RuntimeStreamAdapter`].
///
/// # Channel layout
///
/// - `rt:run:{run_id}` — Events for a specific run.
/// - `rt:session:{session_id}` — Events for a specific session.
/// - `rt:provider:{provider_id}` — Events for a specific provider.
#[derive(Clone)]
pub struct RedisRuntimeStreamAdapter {
    conn: ConnectionManager,
    client: redis::Client,
}

impl RedisRuntimeStreamAdapter {
    /// Creates a new Redis Pub/Sub adapter.
    ///
    /// `conn` must be a [`redis::aio::ConnectionManager`] connected to a
    /// Redis instance. `client` is used to create dedicated Pub/Sub
    /// connections for subscriptions.
    #[must_use]
    pub fn new(conn: ConnectionManager, client: redis::Client) -> Self {
        Self { conn, client }
    }

    fn channel_name(room: &RuntimeRoom) -> String {
        match room {
            RuntimeRoom::Run(run_id) => format!("{CHANNEL_PREFIX}:run:{run_id}"),
            RuntimeRoom::Session(session_id) => format!("{CHANNEL_PREFIX}:session:{session_id}"),
            RuntimeRoom::Provider(provider_id) => {
                format!("{CHANNEL_PREFIX}:provider:{provider_id}")
            }
        }
    }
}

#[async_trait]
impl RuntimeStreamAdapter for RedisRuntimeStreamAdapter {
    async fn publish(
        &self,
        room: RuntimeRoom,
        event: RuntimeEventEnvelope,
    ) -> Result<(), RuntimeStreamError> {
        let channel = Self::channel_name(&room);
        let payload = serde_json::to_string(&event).map_err(|e| RuntimeStreamError::Publish {
            message: format!("failed to serialize envelope: {e}"),
        })?;

        let mut conn = self.conn.clone();
        let _: i32 = redis::cmd("PUBLISH")
            .arg(&channel)
            .arg(&payload)
            .query_async(&mut conn)
            .await
            .map_err(|e| RuntimeStreamError::Publish {
                message: format!("PUBLISH failed: {e}"),
            })?;

        Ok(())
    }

    async fn subscribe(
        &self,
        room: RuntimeRoom,
    ) -> Result<BoxRuntimeEventStream, RuntimeStreamError> {
        let channel = Self::channel_name(&room);

        let mut pubsub: PubSub =
            self.client
                .get_async_pubsub()
                .await
                .map_err(|e| RuntimeStreamError::Subscribe {
                    message: format!("failed to create pubsub connection: {e}"),
                })?;

        pubsub
            .subscribe(&channel)
            .await
            .map_err(|e| RuntimeStreamError::Subscribe {
                message: format!("SUBSCRIBE failed: {e}"),
            })?;

        let (tx, rx) =
            mpsc::channel::<Result<RuntimeEventEnvelope, RuntimeStreamError>>(BRIDGE_CAPACITY);

        let handle = tokio::spawn(async move {
            let mut message_stream = pubsub.on_message();
            loop {
                let msg: Msg = if let Some(msg) = message_stream.next().await {
                    msg
                } else {
                    warn!(%channel, "redis pubsub stream ended");
                    break;
                };

                let payload: String = match msg.get_payload() {
                    Ok(p) => p,
                    Err(e) => {
                        warn!(%channel, error = %e, "redis pubsub payload decode error");
                        continue;
                    }
                };

                match serde_json::from_str::<RuntimeEventEnvelope>(&payload) {
                    Ok(envelope) => {
                        if tx.send(Ok(envelope)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        warn!(%channel, error = %e, "failed to deserialize runtime event envelope from redis pubsub");
                    }
                }
            }
        });

        Ok(Box::pin(RedisPubSubStream { rx, handle }))
    }
}

/// Stream bridging a Redis Pub/Sub subscription into a [`Stream`].
pub struct RedisPubSubStream {
    rx: mpsc::Receiver<Result<RuntimeEventEnvelope, RuntimeStreamError>>,
    handle: JoinHandle<()>,
}

impl Stream for RedisPubSubStream {
    type Item = Result<RuntimeEventEnvelope, RuntimeStreamError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl Drop for RedisPubSubStream {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::event::{AgentEvent, RunStarted};
    use crate::provider::{ModelName, ProviderId};
    use crate::run::RunId;
    use crate::stream::RuntimeEventId;
    use chrono::Utc;
    use futures_util::StreamExt;
    use std::time::Duration;
    use uuid::Uuid;

    fn envelope(run: RunId, seq: u64, session_id: Option<Uuid>) -> RuntimeEventEnvelope {
        let event = AgentEvent::RunStarted(RunStarted {
            run_id: run,
            session_id: session_id.unwrap_or_default(),
            provider: ProviderId::new("acme"),
            model: ModelName::new("gpt-test"),
            timestamp: Utc::now(),
        });
        RuntimeEventEnvelope {
            event_id: RuntimeEventId::new(),
            seq,
            run_id: run,
            session_id,
            event,
            emitted_at: Utc::now(),
        }
    }

    #[tokio::test]
    #[ignore = "requires a running Redis instance"]
    async fn publish_and_subscribe_redis() {
        let client = redis::Client::open("redis://127.0.0.1:6379/").expect("redis client");
        let conn = ConnectionManager::new(client.clone())
            .await
            .expect("connection manager");
        let adapter = RedisRuntimeStreamAdapter::new(conn.clone(), client);
        let run = RunId::new();
        let room = RuntimeRoom::Run(run);

        let mut stream = adapter.subscribe(room.clone()).await.unwrap();

        let env = envelope(run, 1, Some(Uuid::now_v7()));
        adapter.publish(room, env.clone()).await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        match tokio::time::timeout(Duration::from_secs(2), stream.next()).await {
            Ok(Some(Ok(received))) => {
                assert_eq!(received.run_id, run);
                assert_eq!(received.seq, 1);
            }
            Ok(Some(Err(e))) => panic!("unexpected stream error: {e}"),
            Ok(None) => panic!("stream closed unexpectedly"),
            Err(e) => panic!("timed out waiting for pubsub message: {e}"),
        }
    }
}
