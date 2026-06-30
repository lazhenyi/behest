//! In-memory implementations of runtime stores.
//!
//! Provides [`MemoryRunStore`], an ephemeral [`RunStore`] backed by
//! `tokio::sync::RwLock`-guarded [`HashMap`]s. Useful for testing and
//! single-process deployments where persistence is not required.
//!
//! # Architecture
//!
//! Each run is stored as three in-memory maps:
//! - `runs` — [`RunRecord`] keyed by [`Uuid`].
//! - `events` — [`RunEventRecord`] vectors keyed by run UUID.
//! - `projections` — Materialised [`RunState`] projections updated
//!   transactionally on every event append.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::error::{RuntimeError, RuntimeResult};
use super::run::{RunId, RunRecord, RunStatus};
use super::state::RunState;
use super::store::{RunEventRecord, RunStore};

/// In-memory [`RunStore`] implementation backed by `RwLock`-protected hash maps.
///
/// Stores run records, event streams, and materialised [`RunState`] projections
/// in process memory. All data is lost when the store is dropped.
pub struct MemoryRunStore {
    runs: RwLock<HashMap<Uuid, RunRecord>>,
    events: RwLock<HashMap<Uuid, Vec<RunEventRecord>>>,
    projections: RwLock<HashMap<Uuid, RunState>>,
    sequence: AtomicU64,
}

impl MemoryRunStore {
    /// Creates a new in-memory run store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            runs: RwLock::new(HashMap::new()),
            events: RwLock::new(HashMap::new()),
            projections: RwLock::new(HashMap::new()),
            sequence: AtomicU64::new(0),
        }
    }
}

impl Default for MemoryRunStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RunStore for MemoryRunStore {
    async fn create_run(&self, record: RunRecord) -> RuntimeResult<()> {
        let id = *record.id.as_uuid();
        let initial_state = RunState::create(&record, &[]);
        self.runs.write().await.insert(id, record);
        self.projections.write().await.insert(id, initial_state);
        Ok(())
    }

    async fn get_run(&self, run_id: RunId) -> RuntimeResult<Option<RunRecord>> {
        Ok(self.runs.read().await.get(run_id.as_uuid()).cloned())
    }

    async fn get_run_state(&self, run_id: RunId) -> RuntimeResult<Option<RunState>> {
        Ok(self.projections.read().await.get(run_id.as_uuid()).cloned())
    }

    async fn update_run_status(&self, run_id: RunId, status: RunStatus) -> RuntimeResult<()> {
        let mut runs = self.runs.write().await;
        let record = runs
            .get_mut(run_id.as_uuid())
            .ok_or(RuntimeError::RunNotFound(run_id))?;
        record.update_status(status);

        let mut projections = self.projections.write().await;
        if let Some(state) = projections.get_mut(run_id.as_uuid()) {
            state.status = status;
            state.updated_at = chrono::Utc::now();
        }
        Ok(())
    }

    async fn append_event(&self, mut record: RunEventRecord) -> RuntimeResult<()> {
        record.sequence = self.sequence.fetch_add(1, Ordering::SeqCst);
        let run_id_uuid = *record.run_id.as_uuid();

        let mut events = self.events.write().await;
        let mut projections = self.projections.write().await;

        events.entry(run_id_uuid).or_default().push(record.clone());

        if let Some(state) = projections.get_mut(&run_id_uuid) {
            state.apply(&record.event);
            state.event_count += 1;
            state.updated_at = record.timestamp;
        } else {
            let runs = self.runs.read().await;
            if let Some(record_val) = runs.get(&run_id_uuid) {
                let mut state = RunState::create(record_val, &[]);
                state.apply(&record.event);
                state.event_count = 1;
                state.updated_at = record.timestamp;
                projections.insert(run_id_uuid, state);
            }
        }
        Ok(())
    }

    async fn list_events(&self, run_id: RunId) -> RuntimeResult<Vec<RunEventRecord>> {
        Ok(self
            .events
            .read()
            .await
            .get(run_id.as_uuid())
            .cloned()
            .unwrap_or_default())
    }

    async fn list_runs(&self, session_id: Uuid) -> RuntimeResult<Vec<RunRecord>> {
        Ok(self
            .runs
            .read()
            .await
            .values()
            .filter(|r| r.session_id == session_id)
            .cloned()
            .collect())
    }

    async fn list_runs_filtered(
        &self,
        session_id: Option<Uuid>,
        status: Option<RunStatus>,
        limit: usize,
        offset: usize,
    ) -> RuntimeResult<Vec<RunRecord>> {
        let runs = self.runs.read().await;
        let mut result: Vec<RunRecord> = runs
            .values()
            .filter(|r| {
                if let Some(sid) = session_id
                    && r.session_id != sid
                {
                    return false;
                }
                if let Some(s) = &status
                    && r.status != *s
                {
                    return false;
                }
                true
            })
            .cloned()
            .collect();
        result.sort_by_key(|r| std::cmp::Reverse(r.created_at));
        Ok(result
            .into_iter()
            .skip(offset)
            .take(limit.clamp(1, 1000))
            .collect())
    }

    async fn delete_run(&self, run_id: RunId) -> RuntimeResult<()> {
        self.runs.write().await.remove(run_id.as_uuid());
        self.events.write().await.remove(run_id.as_uuid());
        self.projections.write().await.remove(run_id.as_uuid());
        Ok(())
    }

    async fn health_check(&self) -> RuntimeResult<()> {
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::event::{AgentEvent, RunStarted as RunStartedEvent};
    use behest_provider::{ModelName, ProviderId};
    use serde_json::Value;

    fn make_run(session_id: Uuid, provider: &str, model: &str) -> RunRecord {
        RunRecord::new(
            RunId::new(),
            session_id,
            ProviderId::new(provider),
            ModelName::new(model),
            Value::Null,
            None,
        )
    }

    #[tokio::test]
    async fn memory_run_store_should_create_and_get() {
        let store = MemoryRunStore::new();
        let session_id = Uuid::new_v4();
        let record = make_run(session_id, "test", "gpt-4");
        let run_id = record.id;

        store.create_run(record).await.unwrap();
        let fetched = store.get_run(run_id).await.unwrap();
        assert!(fetched.is_some());
    }

    #[tokio::test]
    async fn memory_run_store_should_update_status() {
        let store = MemoryRunStore::new();
        let session_id = Uuid::new_v4();
        let record = make_run(session_id, "test", "gpt-4");
        let run_id = record.id;

        store.create_run(record).await.unwrap();
        store
            .update_run_status(run_id, RunStatus::Completed)
            .await
            .unwrap();

        let fetched = store.get_run(run_id).await.unwrap().unwrap();
        assert_eq!(fetched.status, RunStatus::Completed);
    }

    #[tokio::test]
    async fn memory_run_store_should_append_and_list_events() {
        let store = MemoryRunStore::new();
        let session_id = Uuid::new_v4();
        let record = make_run(session_id, "test", "gpt-4");
        let run_id = record.id;
        store.create_run(record).await.unwrap();

        let event = RunEventRecord::new(
            0,
            run_id,
            AgentEvent::RunStarted(RunStartedEvent {
                run_id,
                session_id,
                provider: ProviderId::new("test"),
                model: ModelName::new("gpt-4"),
                timestamp: chrono::Utc::now(),
            }),
        );
        store.append_event(event).await.unwrap();

        let events = store.list_events(run_id).await.unwrap();
        assert_eq!(events.len(), 1);
    }

    #[tokio::test]
    async fn memory_run_store_should_list_by_session() {
        let store = MemoryRunStore::new();
        let session_id = Uuid::new_v4();

        let r1 = make_run(session_id, "a", "m1");
        let r2 = make_run(session_id, "b", "m2");
        let r3 = make_run(Uuid::new_v4(), "c", "m3");

        store.create_run(r1).await.unwrap();
        store.create_run(r2).await.unwrap();
        store.create_run(r3).await.unwrap();

        let runs = store.list_runs(session_id).await.unwrap();
        assert_eq!(runs.len(), 2);
    }

    #[tokio::test]
    async fn memory_run_store_should_delete() {
        let store = MemoryRunStore::new();
        let session_id = Uuid::new_v4();
        let record = make_run(session_id, "test", "m");
        let run_id = record.id;

        store.create_run(record).await.unwrap();
        store.delete_run(run_id).await.unwrap();

        let fetched = store.get_run(run_id).await.unwrap();
        assert!(fetched.is_none());
    }

    #[tokio::test]
    async fn memory_run_store_should_maintain_transactional_projection() {
        let store = MemoryRunStore::new();
        let session_id = Uuid::new_v4();
        let record = make_run(session_id, "test", "gpt-4");
        let run_id = record.id;
        store.create_run(record).await.unwrap();

        // 1. Check initial projection is Pending
        let state = store.get_run_state(run_id).await.unwrap().unwrap();
        assert_eq!(state.status, RunStatus::Pending);
        assert_eq!(state.event_count, 0);

        // 2. Append RunStarted event and check projection
        let event1 = RunEventRecord::new(
            0,
            run_id,
            AgentEvent::RunStarted(RunStartedEvent {
                run_id,
                session_id,
                provider: ProviderId::new("test"),
                model: ModelName::new("gpt-4"),
                timestamp: chrono::Utc::now(),
            }),
        );
        store.append_event(event1).await.unwrap();

        let state = store.get_run_state(run_id).await.unwrap().unwrap();
        assert_eq!(state.status, RunStatus::SessionLoaded);
        assert_eq!(state.event_count, 1);

        // 3. Append UsageRecorded event and check projection accumulates usage
        let event2 = RunEventRecord::new(
            0,
            run_id,
            AgentEvent::UsageRecorded(UsageRecorded {
                run_id,
                usage: behest_provider::TokenUsage::new(100, 200),
                timestamp: chrono::Utc::now(),
            }),
        );
        store.append_event(event2).await.unwrap();

        let state = store.get_run_state(run_id).await.unwrap().unwrap();
        assert_eq!(state.total_usage.input_tokens, 100);
        assert_eq!(state.total_usage.output_tokens, 200);
        assert_eq!(state.event_count, 2);
    }
}
