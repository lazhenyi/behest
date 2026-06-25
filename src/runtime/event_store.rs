//! Reliable runtime event log.
//!
//! [`RuntimeEventStore`] is the **authoritative replay source** for runtime
//! events. Unlike [`RuntimeStreamAdapter`](super::stream_adapter::RuntimeStreamAdapter),
//! which only performs best-effort live fanout, the store guarantees that any
//! event accepted by [`RuntimeEventStore::append`] can be replayed later via
//! [`RuntimeEventStore::list_after`].
//!
//! Delivery semantics are at-least-once: a consumer reconnecting with
//! `run_id + after_seq` may receive duplicates of events it already observed
//! live; deduplicate via [`RuntimeEventEnvelope::event_id`](super::stream::RuntimeEventEnvelope::event_id)
//! or `seq`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use thiserror::Error;
use tokio::sync::Mutex;

use super::event::AgentEvent;
use super::run::RunId;
use super::stream::{RuntimeEventEnvelope, RuntimeEventId};

#[cfg(feature = "redis")]
#[path = "event_store/redis.rs"]
pub mod redis;

/// Single locked state for [`MemoryRuntimeEventStore`].
///
/// Merging `events`, `seq`, and `sessions` under one [`Mutex`] guarantees that
/// sequence assignment, session propagation, and event insertion are atomic
/// per append — no interleaved racing between the three maps.
#[derive(Debug, Default)]
struct StoreState {
    events: HashMap<RunId, Vec<RuntimeEventEnvelope>>,
    seq: HashMap<RunId, u64>,
    sessions: HashMap<RunId, Option<uuid::Uuid>>,
}

/// Errors raised by a [`RuntimeEventStore`].
#[derive(Debug, Error)]
pub enum RuntimeEventStoreError {
    /// An append could not be persisted.
    #[error("runtime event store append failed: {message}")]
    Append {
        /// Human-readable diagnostic.
        message: String,
    },
    /// The requested run has no recorded events.
    #[error("runtime event store has no events for run {run_id}")]
    NotFound {
        /// Run that was queried.
        run_id: RunId,
    },
}

/// Authoritative replay source for runtime events.
///
/// Implementations are responsible for minting [`RuntimeEventId`] and the
/// per-run `seq` counter on [`RuntimeEventStore::append`].
#[async_trait]
pub trait RuntimeEventStore: Send + Sync {
    /// Appends an event and returns the resulting envelope with identity and
    /// sequence assigned.
    ///
    /// On failure the event MUST NOT be considered persisted; callers (such as
    /// [`RuntimeEventBridge`](super::subscription::RuntimeEventBridge)) rely on
    /// this contract to avoid publishing live events whose replay source is
    /// incomplete.
    async fn append(
        &self,
        event: AgentEvent,
    ) -> Result<RuntimeEventEnvelope, RuntimeEventStoreError>;

    /// Replays events for `run_id` with `seq > after_seq`.
    ///
    /// `after_seq = None` replays from the beginning. `limit` caps the page
    /// size to avoid unbounded memory use.
    async fn list_after(
        &self,
        run_id: RunId,
        after_seq: Option<u64>,
        limit: usize,
    ) -> Result<Vec<RuntimeEventEnvelope>, RuntimeEventStoreError>;
}

/// In-memory [`RuntimeEventStore`] for tests and single-instance development.
///
/// `seq` is monotonic per `run_id`. When a [`AgentEvent::RunStarted`] is
/// appended, its `session_id` is cached and attached to subsequent events of
/// the same run. All state is guarded by a single [`Mutex`] so sequence
/// assignment, session propagation, and event insertion are atomic per
/// [`RuntimeEventStore::append`].
#[derive(Debug, Default)]
pub struct MemoryRuntimeEventStore {
    state: Mutex<StoreState>,
}

impl MemoryRuntimeEventStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl RuntimeEventStore for MemoryRuntimeEventStore {
    async fn append(
        &self,
        event: AgentEvent,
    ) -> Result<RuntimeEventEnvelope, RuntimeEventStoreError> {
        let run_id = event.run_id();
        let mut state = self.state.lock().await;

        let session_id = if let AgentEvent::RunStarted(started) = &event {
            state.sessions.insert(run_id, Some(started.session_id));
            Some(started.session_id)
        } else {
            state.sessions.get(&run_id).copied().flatten()
        };

        let next_seq = {
            let entry = state.seq.entry(run_id).or_default();
            *entry += 1;
            *entry
        };

        let envelope = RuntimeEventEnvelope {
            event_id: RuntimeEventId::new(),
            seq: next_seq,
            run_id,
            session_id,
            event,
            emitted_at: Utc::now(),
        };

        state
            .events
            .entry(run_id)
            .or_default()
            .push(envelope.clone());

        Ok(envelope)
    }

    async fn list_after(
        &self,
        run_id: RunId,
        after_seq: Option<u64>,
        limit: usize,
    ) -> Result<Vec<RuntimeEventEnvelope>, RuntimeEventStoreError> {
        let state = self.state.lock().await;
        let Some(run_events) = state.events.get(&run_id) else {
            return Ok(Vec::new());
        };

        let filtered: Vec<RuntimeEventEnvelope> = run_events
            .iter()
            .filter(|env| match after_seq {
                None => true,
                Some(seq) => env.seq > seq,
            })
            .take(limit)
            .cloned()
            .collect();

        Ok(filtered)
    }
}

/// [`RuntimeEventStore`] that always fails. Used by tests that assert a failed
/// append does not propagate to the live adapter.
#[derive(Debug, Default, Clone, Copy)]
pub struct FailingRuntimeEventStore;

impl FailingRuntimeEventStore {
    /// Creates a new failing store.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl RuntimeEventStore for FailingRuntimeEventStore {
    async fn append(
        &self,
        _event: AgentEvent,
    ) -> Result<RuntimeEventEnvelope, RuntimeEventStoreError> {
        Err(RuntimeEventStoreError::Append {
            message: "failing runtime event store always rejects appends".to_owned(),
        })
    }

    async fn list_after(
        &self,
        run_id: RunId,
        _after_seq: Option<u64>,
        _limit: usize,
    ) -> Result<Vec<RuntimeEventEnvelope>, RuntimeEventStoreError> {
        Err(RuntimeEventStoreError::NotFound { run_id })
    }
}

/// Convenience alias for shared, trait-object event stores.
pub type DynRuntimeEventStore = Arc<dyn RuntimeEventStore>;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use chrono::Utc;
    use uuid::Uuid;

    use super::*;
    use crate::provider::{ModelName, ProviderId};
    use crate::runtime::event::{RunCancelled, RunCompleted, RunFailed, RunStarted};

    fn started(run_id: RunId, session_id: Uuid) -> AgentEvent {
        AgentEvent::RunStarted(RunStarted {
            run_id,
            session_id,
            provider: ProviderId::new("acme"),
            model: ModelName::new("gpt-test"),
            timestamp: Utc::now(),
        })
    }

    fn terminal(run_id: RunId) -> AgentEvent {
        AgentEvent::RunCompleted(RunCompleted {
            run_id,
            finish_reason: crate::provider::FinishReason::Stop,
            iterations: 1,
            timestamp: Utc::now(),
        })
    }

    fn failed(run_id: RunId) -> AgentEvent {
        AgentEvent::RunFailed(RunFailed {
            run_id,
            error: "boom".to_owned(),
            timestamp: Utc::now(),
        })
    }

    fn cancelled(run_id: RunId) -> AgentEvent {
        AgentEvent::RunCancelled(RunCancelled {
            run_id,
            timestamp: Utc::now(),
        })
    }

    #[tokio::test]
    async fn append_assigns_monotonic_seq_per_run() {
        let store = MemoryRuntimeEventStore::new();
        let run = RunId::new();
        let sid = Uuid::now_v7();

        let e1 = store.append(started(run, sid)).await.unwrap();
        let e2 = store.append(terminal(run)).await.unwrap();
        let e3 = store.append(failed(run)).await.unwrap();

        assert_eq!(e1.seq, 1);
        assert_eq!(e2.seq, 2);
        assert_eq!(e3.seq, 3);
    }

    #[tokio::test]
    async fn append_propagates_session_id_from_run_started() {
        let store = MemoryRuntimeEventStore::new();
        let run = RunId::new();
        let sid = Uuid::now_v7();

        let started_env = store.append(started(run, sid)).await.unwrap();
        assert_eq!(started_env.session_id, Some(sid));

        let terminal_env = store.append(terminal(run)).await.unwrap();
        assert_eq!(terminal_env.session_id, Some(sid));
    }

    #[tokio::test]
    async fn list_after_filters_by_seq() {
        let store = MemoryRuntimeEventStore::new();
        let run = RunId::new();
        let sid = Uuid::now_v7();

        store.append(started(run, sid)).await.unwrap();
        let e2 = store.append(terminal(run)).await.unwrap();
        let e3 = store.append(failed(run)).await.unwrap();

        let page = store.list_after(run, Some(e2.seq), 10).await.unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].seq, e3.seq);
    }

    #[tokio::test]
    async fn list_after_respects_limit() {
        let store = MemoryRuntimeEventStore::new();
        let run = RunId::new();
        let sid = Uuid::now_v7();

        store.append(started(run, sid)).await.unwrap();
        store.append(terminal(run)).await.unwrap();
        store.append(failed(run)).await.unwrap();

        let page = store.list_after(run, None, 2).await.unwrap();
        assert_eq!(page.len(), 2);
    }

    #[tokio::test]
    async fn list_after_unknown_run_returns_empty() {
        let store = MemoryRuntimeEventStore::new();
        let run = RunId::new();
        let page = store.list_after(run, None, 10).await.unwrap();
        assert!(page.is_empty());
    }

    #[tokio::test]
    async fn envelope_is_terminal_recognizes_terminal_variants() {
        let store = MemoryRuntimeEventStore::new();
        let run = RunId::new();
        let sid = Uuid::now_v7();

        let non_terminal = store.append(started(run, sid)).await.unwrap();
        assert!(!non_terminal.is_terminal());

        let completed = store.append(terminal(run)).await.unwrap();
        let failed_env = store.append(failed(run)).await.unwrap();
        let cancelled_env = store.append(cancelled(run)).await.unwrap();

        assert!(completed.is_terminal());
        assert!(failed_env.is_terminal());
        assert!(cancelled_env.is_terminal());
    }
}
