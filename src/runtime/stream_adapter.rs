//! Best-effort live fanout for runtime events.
//!
//! [`RuntimeStreamAdapter`] is inspired by the Socket.IO Adapter contract
//! (room-based fanout, per-room ordering, non-blocking publish) but is
//! **not** a Socket.IO implementation and carries no transport. It only moves
//! already-emitted envelopes to live subscribers; durability and replay live
//! in [`RuntimeEventStore`](super::event_store::RuntimeEventStore).
//!
//! Delivery is at-least-once. A slow consumer that falls behind receives a
//! [`RuntimeStreamError::Lagged`] item and may reconcile from the store.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures_util::Stream;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tracing::warn;

use super::stream::{BoxRuntimeEventStream, RuntimeEventEnvelope, RuntimeRoom, RuntimeStreamError};

/// Capacity for the in-memory broadcast channel backing each room.
const ROOM_CHANNEL_CAPACITY: usize = 256;

/// Transport-neutral live fanout for runtime event envelopes.
///
/// `publish` is best-effort: a publish with no live subscribers is not an
/// error. Reliability and replay are the responsibility of
/// [`RuntimeEventStore`](super::event_store::RuntimeEventStore).
#[async_trait]
pub trait RuntimeStreamAdapter: Send + Sync {
    /// Best-effort fanout of `event` to all live subscribers of `room`.
    async fn publish(
        &self,
        room: RuntimeRoom,
        event: RuntimeEventEnvelope,
    ) -> Result<(), RuntimeStreamError>;

    /// Subscribes to the live event stream for `room`.
    async fn subscribe(
        &self,
        room: RuntimeRoom,
    ) -> Result<BoxRuntimeEventStream, RuntimeStreamError>;
}

/// In-memory [`RuntimeStreamAdapter`] for tests and single-instance setups.
///
/// Each [`RuntimeRoom`] is backed by a [`tokio::sync::broadcast`] channel.
/// Slow consumers receive [`RuntimeStreamError::Lagged`] items rather than
/// blocking publishers.
#[derive(Debug, Default)]
pub struct MemoryRuntimeStreamAdapter {
    rooms: tokio::sync::Mutex<HashMap<RuntimeRoom, broadcast::Sender<RuntimeEventEnvelope>>>,
}

impl MemoryRuntimeStreamAdapter {
    /// Creates an empty adapter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    async fn sender_for(&self, room: &RuntimeRoom) -> broadcast::Sender<RuntimeEventEnvelope> {
        let mut rooms = self.rooms.lock().await;
        rooms
            .entry(room.clone())
            .or_insert_with(|| broadcast::channel(ROOM_CHANNEL_CAPACITY).0)
            .clone()
    }
}

#[async_trait]
impl RuntimeStreamAdapter for MemoryRuntimeStreamAdapter {
    async fn publish(
        &self,
        room: RuntimeRoom,
        event: RuntimeEventEnvelope,
    ) -> Result<(), RuntimeStreamError> {
        let sender = self.sender_for(&room).await;
        // `send` fails only when there are no active receivers, which is not
        // an error per the adapter contract.
        if let Err(broadcast::error::SendError(_envelope)) = sender.send(event) {
            tracing::trace!(
                room = %room,
                "runtime stream publish had no live subscribers"
            );
        }
        Ok(())
    }

    async fn subscribe(
        &self,
        room: RuntimeRoom,
    ) -> Result<BoxRuntimeEventStream, RuntimeStreamError> {
        let sender = self.sender_for(&room).await;
        let mut broadcast_rx = sender.subscribe();

        let (mpsc_tx, mpsc_rx) = mpsc::channel::<Result<RuntimeEventEnvelope, RuntimeStreamError>>(
            ROOM_CHANNEL_CAPACITY,
        );
        let handle = tokio::spawn(async move {
            loop {
                match broadcast_rx.recv().await {
                    Ok(envelope) => {
                        if mpsc_tx.send(Ok(envelope)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        warn!(
                            skipped,
                            "runtime stream subscriber lagged behind live fanout"
                        );
                        if mpsc_tx
                            .send(Err(RuntimeStreamError::Lagged { skipped }))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
        });

        Ok(Box::pin(BroadcastEnvelopeStream {
            rx: mpsc_rx,
            handle,
        }))
    }
}

/// Owned stream bridging a [`broadcast::Receiver`] into a [`Stream`].
///
/// Implemented by hand (instead of pulling in `tokio-stream`) on top of
/// [`mpsc::Receiver::poll_recv`]. Dropping the stream aborts the forwarder
/// task so consumers do not leak.
pub struct BroadcastEnvelopeStream {
    rx: mpsc::Receiver<Result<RuntimeEventEnvelope, RuntimeStreamError>>,
    handle: JoinHandle<()>,
}

impl Stream for BroadcastEnvelopeStream {
    type Item = Result<RuntimeEventEnvelope, RuntimeStreamError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl Drop for BroadcastEnvelopeStream {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// [`RuntimeStreamAdapter`] that always fails. Used by tests asserting a
/// failed publish does not panic the runtime bridge.
#[derive(Debug, Default, Clone, Copy)]
pub struct FailingRuntimeStreamAdapter;

impl FailingRuntimeStreamAdapter {
    /// Creates a new failing adapter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl RuntimeStreamAdapter for FailingRuntimeStreamAdapter {
    async fn publish(
        &self,
        _room: RuntimeRoom,
        _event: RuntimeEventEnvelope,
    ) -> Result<(), RuntimeStreamError> {
        Err(RuntimeStreamError::Publish {
            message: "failing runtime stream adapter always rejects publish".to_owned(),
        })
    }

    async fn subscribe(
        &self,
        _room: RuntimeRoom,
    ) -> Result<BoxRuntimeEventStream, RuntimeStreamError> {
        Err(RuntimeStreamError::Subscribe {
            message: "failing runtime stream adapter never subscribes".to_owned(),
        })
    }
}

/// Convenience alias for shared, trait-object stream adapters.
pub type DynRuntimeStreamAdapter = Arc<dyn RuntimeStreamAdapter>;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::time::Duration;

    use chrono::Utc;
    use futures_util::StreamExt;
    use uuid::Uuid;

    use super::*;
    use crate::provider::{ModelName, ProviderId};
    use crate::runtime::event::{AgentEvent, RunCompleted, RunStarted};
    use crate::runtime::run::RunId;
    use crate::runtime::stream::RuntimeEventId;

    fn envelope(run: RunId, seq: u64, session_id: Option<Uuid>) -> RuntimeEventEnvelope {
        let event = if seq == 1 {
            AgentEvent::RunStarted(RunStarted {
                run_id: run,
                session_id: session_id.unwrap_or_default(),
                provider: ProviderId::new("acme"),
                model: ModelName::new("gpt-test"),
                timestamp: Utc::now(),
            })
        } else {
            AgentEvent::RunCompleted(RunCompleted {
                run_id: run,
                finish_reason: crate::provider::FinishReason::Stop,
                iterations: usize::try_from(seq).unwrap_or(usize::MAX),
                timestamp: Utc::now(),
            })
        };
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
    async fn publish_reaches_subscriber() {
        let adapter = MemoryRuntimeStreamAdapter::new();
        let run = RunId::new();
        let room = RuntimeRoom::Run(run);

        let mut stream = adapter.subscribe(room.clone()).await.unwrap();
        adapter.publish(room, envelope(run, 1, None)).await.unwrap();

        let received = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("timed out waiting for live event")
            .expect("stream ended")
            .expect("lagged");
        assert_eq!(received.seq, 1);
    }

    #[tokio::test]
    async fn different_rooms_do_not_cross_talk() {
        let adapter = MemoryRuntimeStreamAdapter::new();
        let run_a = RunId::new();
        let run_b = RunId::new();

        let mut stream_a = adapter.subscribe(RuntimeRoom::Run(run_a)).await.unwrap();

        adapter
            .publish(RuntimeRoom::Run(run_b), envelope(run_b, 1, None))
            .await
            .unwrap();
        adapter
            .publish(RuntimeRoom::Run(run_a), envelope(run_a, 1, None))
            .await
            .unwrap();

        let received = tokio::time::timeout(Duration::from_secs(1), stream_a.next())
            .await
            .expect("timed out waiting for live event")
            .expect("stream ended")
            .expect("lagged");
        assert_eq!(received.run_id, run_a);
    }

    #[tokio::test]
    async fn publish_without_subscribers_is_not_an_error() {
        let adapter = MemoryRuntimeStreamAdapter::new();
        let run = RunId::new();
        adapter
            .publish(RuntimeRoom::Run(run), envelope(run, 1, None))
            .await
            .expect("publish with no subscribers must not error");
    }

    #[tokio::test]
    async fn failing_adapter_publish_returns_error_without_panic() {
        let adapter = FailingRuntimeStreamAdapter::new();
        let run = RunId::new();
        let err = adapter
            .publish(RuntimeRoom::Run(run), envelope(run, 1, None))
            .await
            .unwrap_err();
        assert!(matches!(err, RuntimeStreamError::Publish { .. }));
    }
}
