//! Coordination layer combining replay and live fanout.
//!
//! [`RuntimeSubscriptionHub`] stitches [`RuntimeEventStore`] (authoritative
//! replay) together with [`RuntimeStreamAdapter`] (best-effort live fanout) so
//! a reconnecting client can receive `replay` first and then drain `live`.
//!
//! [`RuntimeEventBridge`] drains [`AgentRuntime::subscribe`](super::agent::AgentRuntime::subscribe)
//! into the store+adapter pair: every event is appended first, and only on
//! success is it published to the run (and, when known, session) room.
//! Per the at-least-once contract, consumers deduplicate via
//! [`RuntimeEventEnvelope::event_id`](super::stream::RuntimeEventEnvelope::event_id)
//! or `seq`.

use std::sync::Arc;

use thiserror::Error;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tracing::warn;

use super::agent::AgentRuntime;
use super::event::AgentEvent;
use super::event_store::{RuntimeEventStore, RuntimeEventStoreError};
use super::run::RunId;
use super::stream::{BoxRuntimeEventStream, RuntimeEventEnvelope, RuntimeRoom, RuntimeStreamError};
use super::stream_adapter::RuntimeStreamAdapter;

/// Replay page plus a live stream for a single subscription.
///
/// Callers are expected to drain `replay` first, then poll `live`. Overlap is
/// possible (an event present in `replay` may also arrive via `live`); dedup
/// via [`RuntimeEventEnvelope::event_id`](super::stream::RuntimeEventEnvelope::event_id)
/// or `seq`.
pub struct RuntimeSubscription {
    /// Buffered replay events read from the store.
    pub replay: Vec<RuntimeEventEnvelope>,
    /// Live fanout stream for subsequent events.
    pub live: BoxRuntimeEventStream,
}

/// Errors raised while assembling a [`RuntimeSubscription`].
#[derive(Debug, Error)]
pub enum RuntimeSubscriptionError {
    /// Replay failed against the event store.
    #[error(transparent)]
    Replay(#[from] RuntimeEventStoreError),
    /// Live subscription failed against the stream adapter.
    #[error(transparent)]
    Live(#[from] RuntimeStreamError),
}

/// Errors surfaced by a [`RuntimeEventBridge`] handle.
#[derive(Debug, Error)]
pub enum RuntimeEventBridgeError {
    /// The bridge background task was cancelled.
    #[error("runtime event bridge task was cancelled")]
    Cancelled,
    /// The bridge background task panicked.
    #[error("runtime event bridge task panicked: {message}")]
    TaskPanic {
        /// Human-readable diagnostic.
        message: String,
    },
}

/// Combines replay and live fanout for runtime event streams.
///
/// `subscribe_run` reads a replay page from the store and opens a live
/// subscription to the run room in one call. The caller owns the ordering
/// decision (replay first, then live).
pub struct RuntimeSubscriptionHub {
    event_store: Arc<dyn RuntimeEventStore>,
    stream_adapter: Arc<dyn RuntimeStreamAdapter>,
}

impl RuntimeSubscriptionHub {
    /// Creates a hub over the given store and adapter.
    #[must_use]
    pub fn new(
        event_store: Arc<dyn RuntimeEventStore>,
        stream_adapter: Arc<dyn RuntimeStreamAdapter>,
    ) -> Self {
        Self {
            event_store,
            stream_adapter,
        }
    }

    /// Returns replay events plus a live stream for `run_id`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeSubscriptionError::Replay`] when the event store
    /// cannot fulfil the replay request, and
    /// [`RuntimeSubscriptionError::Live`] when the stream adapter refuses
    /// the live subscription.
    pub async fn subscribe_run(
        &self,
        run_id: RunId,
        after_seq: Option<u64>,
        limit: usize,
    ) -> Result<RuntimeSubscription, RuntimeSubscriptionError> {
        let replay = self
            .event_store
            .list_after(run_id, after_seq, limit)
            .await
            .map_err(RuntimeSubscriptionError::Replay)?;
        let live = self
            .stream_adapter
            .subscribe(RuntimeRoom::Run(run_id))
            .await
            .map_err(RuntimeSubscriptionError::Live)?;
        Ok(RuntimeSubscription { replay, live })
    }
}

/// Drains [`AgentRuntime::subscribe`](super::agent::AgentRuntime::subscribe)
/// into the event store and stream adapter.
///
/// The bridge appends each event to the store first; only on a successful
/// append does it publish the envelope to the run room (and, when a session id
/// is known, to the session room). Append failure therefore never produces a
/// live event with an incomplete replay source.
pub struct RuntimeEventBridge {
    runtime: Arc<AgentRuntime>,
    event_store: Arc<dyn RuntimeEventStore>,
    stream_adapter: Arc<dyn RuntimeStreamAdapter>,
}

impl RuntimeEventBridge {
    /// Creates a bridge wiring `runtime` to `event_store` and `stream_adapter`.
    #[must_use]
    pub fn new(
        runtime: Arc<AgentRuntime>,
        event_store: Arc<dyn RuntimeEventStore>,
        stream_adapter: Arc<dyn RuntimeStreamAdapter>,
    ) -> Self {
        Self {
            runtime,
            event_store,
            stream_adapter,
        }
    }

    /// Spawns the background forwarding task and returns a handle.
    ///
    /// The task runs until the runtime's event broadcast closes or the handle
    /// is dropped/aborted.
    #[must_use]
    pub fn spawn(self: Arc<Self>) -> RuntimeEventBridgeHandle {
        let rx = self.runtime.subscribe();
        let event_store = self.event_store.clone();
        let stream_adapter = self.stream_adapter.clone();
        let task = tokio::spawn(forward_events(rx, event_store, stream_adapter));
        RuntimeEventBridgeHandle { task }
    }
}

/// Forwarding loop shared by [`RuntimeEventBridge::spawn`] and tests.
///
/// Factored out so tests can drive the bridge with a synthetic
/// [`broadcast::Receiver`] without constructing a full [`AgentRuntime`].
async fn forward_events(
    mut rx: broadcast::Receiver<AgentEvent>,
    event_store: Arc<dyn RuntimeEventStore>,
    stream_adapter: Arc<dyn RuntimeStreamAdapter>,
) {
    loop {
        match rx.recv().await {
            Ok(event) => {
                let run_id = event.run_id();
                match event_store.append(event).await {
                    Ok(envelope) => {
                        if let Err(error) = stream_adapter
                            .publish(RuntimeRoom::Run(run_id), envelope.clone())
                            .await
                        {
                            warn!(%error, "runtime event bridge publish to run room failed");
                        }
                        if let Some(session_id) = envelope.session_id {
                            if let Err(error) = stream_adapter
                                .publish(RuntimeRoom::Session(session_id), envelope)
                                .await
                            {
                                warn!(%error, "runtime event bridge publish to session room failed");
                            }
                        }
                    }
                    Err(error) => {
                        warn!(
                            %error,
                            "runtime event bridge append failed; skipping live publish"
                        );
                    }
                }
            }
            Err(broadcast::error::RecvError::Closed) => break,
            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                warn!(
                    skipped,
                    "runtime event bridge lagged behind runtime broadcast"
                );
            }
        }
    }
}

/// Handle to a running [`RuntimeEventBridge`] task.
///
/// Dropping the handle aborts the task. Call [`RuntimeEventBridgeHandle::abort`]
/// to stop it explicitly, or [`RuntimeEventBridgeHandle::join`] to await
/// completion.
pub struct RuntimeEventBridgeHandle {
    task: JoinHandle<()>,
}

impl RuntimeEventBridgeHandle {
    /// Aborts the bridge task. Idempotent.
    pub fn abort(&self) {
        self.task.abort();
    }

    /// Returns `true` once the bridge task has finished.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.task.is_finished()
    }

    /// Awaits task completion.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeEventBridgeError::Cancelled`] if the task was
    /// aborted, and [`RuntimeEventBridgeError::TaskPanic`] if it exited
    /// unexpectedly (e.g. panic).
    pub async fn join(mut self) -> Result<(), RuntimeEventBridgeError> {
        match (&mut self.task).await {
            Ok(()) => Ok(()),
            Err(err) if err.is_cancelled() => Err(RuntimeEventBridgeError::Cancelled),
            Err(_) => Err(RuntimeEventBridgeError::TaskPanic {
                message: "bridge task exited unexpectedly".to_owned(),
            }),
        }
    }
}

impl Drop for RuntimeEventBridgeHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use chrono::Utc;
    use futures_util::StreamExt;
    use uuid::Uuid;

    use super::*;
    use crate::provider::{FinishReason, ModelName, ProviderId};
    use crate::runtime::event::{RunCompleted, RunStarted};
    use crate::runtime::event_store::{FailingRuntimeEventStore, MemoryRuntimeEventStore};
    use crate::runtime::stream::RuntimeEventId;
    use crate::runtime::stream_adapter::MemoryRuntimeStreamAdapter;

    fn started(run: RunId, session_id: Uuid) -> AgentEvent {
        AgentEvent::RunStarted(RunStarted {
            run_id: run,
            session_id,
            provider: ProviderId::new("acme"),
            model: ModelName::new("gpt-test"),
            timestamp: Utc::now(),
        })
    }

    fn terminal(run: RunId) -> AgentEvent {
        AgentEvent::RunCompleted(RunCompleted {
            run_id: run,
            finish_reason: FinishReason::Stop,
            iterations: 1,
            timestamp: Utc::now(),
        })
    }

    fn envelope(
        run: RunId,
        seq: u64,
        session_id: Option<Uuid>,
        event: AgentEvent,
    ) -> RuntimeEventEnvelope {
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
    async fn hub_returns_replay_and_live() {
        let store: Arc<dyn RuntimeEventStore> = Arc::new(MemoryRuntimeEventStore::new());
        let adapter: Arc<dyn RuntimeStreamAdapter> = Arc::new(MemoryRuntimeStreamAdapter::new());

        let run = RunId::new();
        let sid = Uuid::now_v7();
        store.append(started(run, sid)).await.unwrap();
        store.append(terminal(run)).await.unwrap();

        let hub = RuntimeSubscriptionHub::new(store.clone(), adapter.clone());
        let mut sub = hub.subscribe_run(run, None, 10).await.unwrap();
        assert_eq!(sub.replay.len(), 2);

        // Publish a live event after subscribing; it must arrive on `live`.
        adapter
            .publish(
                RuntimeRoom::Run(run),
                envelope(run, 3, Some(sid), terminal(run)),
            )
            .await
            .unwrap();
        let received = tokio::time::timeout(Duration::from_secs(1), sub.live.next())
            .await
            .expect("timed out waiting for live event")
            .expect("stream ended")
            .expect("lagged");
        assert_eq!(received.seq, 3);
    }

    #[tokio::test]
    async fn bridge_persists_and_publishes_events() {
        let (tx, rx) = broadcast::channel::<AgentEvent>(16);
        let store: Arc<dyn RuntimeEventStore> = Arc::new(MemoryRuntimeEventStore::new());
        let adapter: Arc<dyn RuntimeStreamAdapter> = Arc::new(MemoryRuntimeStreamAdapter::new());

        let run = RunId::new();
        let sid = Uuid::now_v7();
        let mut live = adapter.subscribe(RuntimeRoom::Run(run)).await.unwrap();

        let _handle = tokio::spawn(forward_events(rx, store.clone(), adapter.clone()));
        tx.send(started(run, sid)).unwrap();

        let received = tokio::time::timeout(Duration::from_secs(1), live.next())
            .await
            .expect("timed out waiting for live event")
            .expect("stream ended")
            .expect("lagged");
        assert_eq!(received.seq, 1);
        assert_eq!(received.session_id, Some(sid));

        let replayed = store.list_after(run, None, 10).await.unwrap();
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].seq, 1);
    }

    #[derive(Default)]
    struct CountingRuntimeStreamAdapter {
        publishes: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl RuntimeStreamAdapter for CountingRuntimeStreamAdapter {
        async fn publish(
            &self,
            _room: RuntimeRoom,
            _event: RuntimeEventEnvelope,
        ) -> Result<(), RuntimeStreamError> {
            self.publishes.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        async fn subscribe(
            &self,
            _room: RuntimeRoom,
        ) -> Result<BoxRuntimeEventStream, RuntimeStreamError> {
            Err(RuntimeStreamError::Subscribe {
                message: "counting adapter does not support subscribe".to_owned(),
            })
        }
    }

    #[tokio::test]
    async fn append_failure_does_not_publish_live_event() {
        let (tx, rx) = broadcast::channel::<AgentEvent>(16);
        let store: Arc<dyn RuntimeEventStore> = Arc::new(FailingRuntimeEventStore::new());
        let adapter = Arc::new(CountingRuntimeStreamAdapter::default());

        let run = RunId::new();
        let sid = Uuid::now_v7();
        let _handle = tokio::spawn(forward_events(rx, store.clone(), adapter.clone()));
        tx.send(started(run, sid)).unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            adapter.publishes.load(Ordering::Relaxed),
            0,
            "no live event should be published when append fails"
        );
    }
}
