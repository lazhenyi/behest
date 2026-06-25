//! NATS JetStream-backed [`RuntimeStreamAdapter`].
//!
//! Each [`RuntimeRoom`] maps to a JetStream subject. [`publish`] uses
//! `jetstream.publish()` with a JSON-serialized envelope. [`subscribe`]
//! creates an ephemeral consumer and bridges into a [`BoxRuntimeEventStream`].
//!
//! JetStream provides durable, replayable streams. Unlike the in-memory
//! or Redis Pub/Sub adapters, JetStream retains published events even after
//! consumers disconnect, enabling late-joining subscribers to catch up.

use std::pin::Pin;
use std::task::{Context, Poll};

use async_nats::jetstream::consumer::AckPolicy;
use async_nats::jetstream::stream::{Config as StreamConfig, RetentionPolicy};
use async_nats::jetstream::{self};
use async_trait::async_trait;
use futures_util::{Stream, TryStreamExt};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::warn;

use crate::runtime::stream::{
    BoxRuntimeEventStream, RuntimeEventEnvelope, RuntimeRoom, RuntimeStreamError,
};

use crate::runtime::stream_adapter::RuntimeStreamAdapter;

/// NATS JetStream subject prefix for runtime event rooms.
const SUBJECT_PREFIX: &str = "behest.rt";

/// Capacity for the internal MPSC bridge channel.
const BRIDGE_CAPACITY: usize = 256;

/// NATS JetStream-backed [`RuntimeStreamAdapter`].
///
/// # Subject layout
///
/// - `behest.rt.run.{run_id}` — Events for a specific run.
/// - `behest.rt.session.{session_id}` — Events for a specific session.
/// - `behest.rt.provider.{provider_id}` — Events for a specific provider.
///
/// Each room gets its own JetStream stream with an interest-based retention
/// policy, so events are deleted when no consumer references them.
#[derive(Clone)]
pub struct NatsJetStreamStreamAdapter {
    jetstream: jetstream::Context,
}

impl NatsJetStreamStreamAdapter {
    /// Creates a new NATS JetStream adapter.
    ///
    /// `jetstream` must be a [`async_nats::jetstream::Context`] from an
    /// established NATS connection with JetStream enabled.
    #[must_use]
    pub fn new(jetstream: jetstream::Context) -> Self {
        Self { jetstream }
    }

    fn subject(room: &RuntimeRoom) -> String {
        match room {
            RuntimeRoom::Run(run_id) => format!("{SUBJECT_PREFIX}.run.{run_id}"),
            RuntimeRoom::Session(session_id) => format!("{SUBJECT_PREFIX}.session.{session_id}"),
            RuntimeRoom::Provider(provider_id) => {
                format!("{SUBJECT_PREFIX}.provider.{provider_id}")
            }
        }
    }

    fn stream_name(room: &RuntimeRoom) -> String {
        match room {
            RuntimeRoom::Run(run_id) => format!("RT_RUN_{run_id}"),
            RuntimeRoom::Session(session_id) => format!("RT_SESSION_{session_id}"),
            RuntimeRoom::Provider(provider_id) => format!("RT_PROVIDER_{provider_id}"),
        }
    }
}

#[async_trait]
impl RuntimeStreamAdapter for NatsJetStreamStreamAdapter {
    async fn publish(
        &self,
        room: RuntimeRoom,
        envelope: RuntimeEventEnvelope,
    ) -> Result<(), RuntimeStreamError> {
        let subject = Self::subject(&room);
        let stream_name = Self::stream_name(&room);

        let payload = serde_json::to_vec(&envelope).map_err(|e| RuntimeStreamError::Publish {
            message: format!("failed to serialize envelope: {e}"),
        })?;

        let _stream = self
            .jetstream
            .get_or_create_stream(StreamConfig {
                name: stream_name,
                subjects: vec![subject.clone()],
                retention: RetentionPolicy::Interest,
                ..Default::default()
            })
            .await
            .map_err(|e| RuntimeStreamError::Publish {
                message: format!("failed to get or create stream: {e}"),
            })?;

        let ack = self
            .jetstream
            .publish(subject, payload.into())
            .await
            .map_err(|e| RuntimeStreamError::Publish {
                message: format!("publish failed: {e}"),
            })?;

        let _ = ack.await.map_err(|e| RuntimeStreamError::Publish {
            message: format!("publish ack failed: {e}"),
        })?;

        Ok(())
    }

    async fn subscribe(
        &self,
        room: RuntimeRoom,
    ) -> Result<BoxRuntimeEventStream, RuntimeStreamError> {
        let subject = Self::subject(&room);
        let stream_name = Self::stream_name(&room);

        let stream = self
            .jetstream
            .get_or_create_stream(StreamConfig {
                name: stream_name.clone(),
                subjects: vec![subject],
                retention: RetentionPolicy::Interest,
                ..Default::default()
            })
            .await
            .map_err(|e| RuntimeStreamError::Subscribe {
                message: format!("failed to get or create stream: {e}"),
            })?;

        let consumer = stream
            .get_or_create_consumer(
                &format!("{stream_name}_consumer"),
                async_nats::jetstream::consumer::pull::Config {
                    ack_policy: AckPolicy::None,
                    deliver_policy: async_nats::jetstream::consumer::DeliverPolicy::All,
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| RuntimeStreamError::Subscribe {
                message: format!("failed to create consumer: {e}"),
            })?;

        let mut messages =
            consumer
                .messages()
                .await
                .map_err(|e| RuntimeStreamError::Subscribe {
                    message: format!("failed to get messages: {e}"),
                })?;

        let (tx, rx) =
            mpsc::channel::<Result<RuntimeEventEnvelope, RuntimeStreamError>>(BRIDGE_CAPACITY);

        let handle = tokio::spawn(async move {
            loop {
                match messages.try_next().await {
                    Ok(Some(msg)) => {
                        match serde_json::from_slice::<RuntimeEventEnvelope>(&msg.payload) {
                            Ok(envelope) => {
                                if tx.send(Ok(envelope)).await.is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!(
                                    stream = %stream_name,
                                    error = %e,
                                    "failed to deserialize runtime event envelope from nats jetstream"
                                );
                            }
                        }
                    }
                    Ok(None) => {
                        break;
                    }
                    Err(e) => {
                        warn!(
                            stream = %stream_name,
                            error = %e,
                            "nats jetstream consumer error"
                        );
                        let _ = tx.send(Err(RuntimeStreamError::Closed)).await;
                        break;
                    }
                }
            }
        });

        Ok(Box::pin(NatsJetStreamStream { rx, handle }))
    }
}

/// Stream bridging a NATS JetStream subscription into a [`Stream`].
pub struct NatsJetStreamStream {
    rx: mpsc::Receiver<Result<RuntimeEventEnvelope, RuntimeStreamError>>,
    handle: JoinHandle<()>,
}

impl Stream for NatsJetStreamStream {
    type Item = Result<RuntimeEventEnvelope, RuntimeStreamError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl Drop for NatsJetStreamStream {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::provider::{ModelName, ProviderId};
    use crate::runtime::event::{AgentEvent, RunStarted};
    use crate::runtime::run::RunId;
    use crate::runtime::stream::RuntimeEventId;
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
    #[ignore = "requires a running NATS server with JetStream enabled"]
    async fn publish_and_subscribe_jetstream() {
        let client = async_nats::connect("nats://localhost:4222")
            .await
            .expect("nats client");
        let jetstream = async_nats::jetstream::new(client);
        let adapter = NatsJetStreamStreamAdapter::new(jetstream);
        let run = RunId::new();
        let room = RuntimeRoom::Run(run);

        let mut stream = adapter.subscribe(room.clone()).await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        let env = envelope(run, 1, Some(Uuid::now_v7()));
        adapter.publish(room, env.clone()).await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        match tokio::time::timeout(Duration::from_secs(5), stream.next()).await {
            Ok(Some(Ok(received))) => {
                assert_eq!(received.run_id, run);
                assert_eq!(received.seq, 1);
            }
            Ok(Some(Err(e))) => panic!("unexpected stream error: {e}"),
            Ok(None) => panic!("stream closed unexpectedly"),
            Err(elapsed) => panic!("timed out waiting for jetstream message after {elapsed}"),
        }
    }
}
