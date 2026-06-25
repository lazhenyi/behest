//! Runtime store facade.
//!
//! Provides a unified interface for runtime persistence operations,
//! composing session, execution, and run stores.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::provider::Message;
use crate::store::{
    ArtifactStore, EmbeddingStore, ExecutionStore, MessageRecord, MessageRole, SessionStore,
};

use super::error::{RuntimeError, RuntimeResult};
use super::event::AgentEvent;
use super::run::{RunId, RunRecord, RunStatus};
use super::state::RunState;

/// Persistent record of a run event with sequence number ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEventRecord {
    /// Monotonically increasing event sequence number within the run.
    pub sequence: u64,
    /// Run this event belongs to.
    pub run_id: RunId,
    /// The event payload.
    pub event: AgentEvent,
    /// When the event was recorded.
    pub timestamp: DateTime<Utc>,
}

impl RunEventRecord {
    /// Creates a new run event record with the current timestamp.
    #[must_use]
    pub fn new(sequence: u64, run_id: RunId, event: AgentEvent) -> Self {
        Self {
            sequence,
            run_id,
            event,
            timestamp: Utc::now(),
        }
    }
}

/// Store for run lifecycle and events.
///
/// Implementations provide persistence for run metadata and the
/// event-sourced event log. Default implementations are provided for
/// [`get_run_state`](Self::get_run_state) and
/// [`list_runs_filtered`](Self::list_runs_filtered), which backends
/// may override with native projections for efficiency.
#[async_trait]
pub trait RunStore: Send + Sync {
    /// Persists a new run record.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Storage`] on persistence failure.
    async fn create_run(&self, record: RunRecord) -> RuntimeResult<()>;

    /// Loads a run record by its identifier.
    ///
    /// Returns `None` when no run with the given ID exists.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Storage`] on persistence failure.
    async fn get_run(&self, run_id: RunId) -> RuntimeResult<Option<RunRecord>>;

    /// Gets the event-sourced state of a run by replaying its event log.
    ///
    /// Default implementation calls [`Self::get_run`] + [`Self::list_events`] and
    /// folds them into a [`RunState`]. Backends may override with a
    /// native projection for better performance.
    async fn get_run_state(&self, run_id: RunId) -> RuntimeResult<Option<RunState>> {
        let Some(record) = self.get_run(run_id).await? else {
            return Ok(None);
        };
        let events = self.list_events(run_id).await?;
        Ok(Some(RunState::create(&record, &events)))
    }

    /// Updates the status of an existing run.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Storage`] on persistence failure.
    async fn update_run_status(&self, run_id: RunId, status: RunStatus) -> RuntimeResult<()>;

    /// Appends an event to a run's event log.
    ///
    /// The event is stored as a [`RunEventRecord`] with a monotonically
    /// increasing sequence number.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Storage`] on persistence failure.
    async fn append_event(&self, record: RunEventRecord) -> RuntimeResult<()>;

    /// Returns the full event log for a run, ordered by sequence number.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::RunNotFound`] when the run does not exist,
    /// or [`RuntimeError::Storage`] on persistence failure.
    async fn list_events(&self, run_id: RunId) -> RuntimeResult<Vec<RunEventRecord>>;

    /// Lists all runs belonging to a session.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Storage`] on persistence failure.
    async fn list_runs(&self, session_id: Uuid) -> RuntimeResult<Vec<RunRecord>>;

    /// Lists runs with optional filters and pagination.
    ///
    /// Default implementation iterates all sessions; backends should override
    /// with a native query for efficiency.
    async fn list_runs_filtered(
        &self,
        session_id: Option<Uuid>,
        status: Option<RunStatus>,
        limit: usize,
        offset: usize,
    ) -> RuntimeResult<Vec<RunRecord>> {
        let _ = (session_id, status, limit, offset);
        Err(RuntimeError::Storage(crate::StorageError::BackendError {
            backend: "run".to_owned(),
            message: "list_runs_filtered not implemented".to_owned(),
            source: None,
        }))
    }

    /// Deletes a run and all its associated events.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Storage`] on persistence failure.
    async fn delete_run(&self, run_id: RunId) -> RuntimeResult<()>;

    /// Performs a health check against the underlying storage backend.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Storage`] when the backend is unhealthy.
    async fn health_check(&self) -> RuntimeResult<()>;
}

/// Runtime store facade composing session, execution, run, embedding, and artifact stores.
///
/// Provides a unified interface for all runtime persistence operations.
/// Individual sub-stores are accessed via their respective accessor methods.
pub struct RuntimeStore {
    sessions: Box<dyn SessionStore>,
    executions: Box<dyn ExecutionStore>,
    runs: Box<dyn RunStore>,
    embeddings: Option<Box<dyn EmbeddingStore>>,
    artifacts: Option<Box<dyn ArtifactStore>>,
}

impl RuntimeStore {
    /// Creates a new runtime store with session, execution, and run stores.
    ///
    /// Embedding and artifact stores are optional — attach them with
    /// [`with_embeddings`](Self::with_embeddings) and
    /// [`with_artifacts`](Self::with_artifacts).
    #[must_use]
    pub fn new(
        sessions: Box<dyn SessionStore>,
        executions: Box<dyn ExecutionStore>,
        runs: Box<dyn RunStore>,
    ) -> Self {
        Self {
            sessions,
            executions,
            runs,
            embeddings: None,
            artifacts: None,
        }
    }

    /// Attaches an optional embedding store for vector search operations.
    #[must_use]
    pub fn with_embeddings(mut self, store: Box<dyn EmbeddingStore>) -> Self {
        self.embeddings = Some(store);
        self
    }

    /// Attaches an optional artifact store for file/blob storage.
    #[must_use]
    pub fn with_artifacts(mut self, store: Box<dyn ArtifactStore>) -> Self {
        self.artifacts = Some(store);
        self
    }

    /// Returns the session store.
    #[must_use]
    pub fn sessions(&self) -> &dyn SessionStore {
        &*self.sessions
    }

    /// Returns the execution store.
    #[must_use]
    pub fn executions(&self) -> &dyn ExecutionStore {
        &*self.executions
    }

    /// Returns the run store.
    #[must_use]
    pub fn runs(&self) -> &dyn RunStore {
        &*self.runs
    }

    /// Returns the embedding store, if configured.
    #[must_use]
    pub fn embeddings(&self) -> Option<&dyn EmbeddingStore> {
        self.embeddings.as_deref()
    }

    /// Returns the artifact store, if configured.
    #[must_use]
    pub fn artifacts(&self) -> Option<&dyn ArtifactStore> {
        self.artifacts.as_deref()
    }

    /// Creates or loads a session.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::SessionNotFound`] when `session_id` is provided
    /// but does not exist, or [`RuntimeError::Storage`] on persistence failure.
    pub async fn ensure_session(&self, session_id: Option<Uuid>) -> RuntimeResult<Uuid> {
        if let Some(id) = session_id {
            self.sessions
                .get_session(&id)
                .await
                .map_err(RuntimeError::from)?
                .ok_or(RuntimeError::SessionNotFound(id))?;
            Ok(id)
        } else {
            let session =
                crate::store::Session::new("Agent Run", crate::provider::ModelName::new("default"));
            self.sessions
                .create_session(session.clone())
                .await
                .map_err(RuntimeError::from)?;
            Ok(session.id)
        }
    }

    /// Appends a message to a session.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Storage`] on persistence failure.
    pub async fn append_message(&self, session_id: Uuid, message: &Message) -> RuntimeResult<Uuid> {
        let record = message_to_record(session_id, message);
        let result = self
            .sessions
            .append_message(record)
            .await
            .map_err(RuntimeError::from)?;
        Ok(result.id)
    }

    /// Lists messages for a session.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Storage`] on persistence failure.
    pub async fn list_messages(&self, session_id: Uuid) -> RuntimeResult<Vec<Message>> {
        let records = self
            .sessions
            .list_messages(&session_id)
            .await
            .map_err(RuntimeError::from)?;
        Ok(records.into_iter().filter_map(record_to_message).collect())
    }
}

/// Converts a provider [`Message`] to a persisted [`MessageRecord`].
///
/// Maps message role variants to their corresponding store representations,
/// preserving tool call metadata for assistant and tool messages.
fn message_to_record(session_id: Uuid, message: &Message) -> MessageRecord {
    match message {
        Message::System { content } => {
            MessageRecord::new(session_id, MessageRole::System, content.clone())
        }
        Message::User { content } => {
            MessageRecord::new(session_id, MessageRole::User, content.clone())
        }
        Message::Assistant {
            content,
            tool_calls,
        } => MessageRecord::new(session_id, MessageRole::Assistant, content.clone())
            .with_tool_calls(tool_calls.clone()),
        Message::Tool {
            tool_call_id,
            name,
            content,
        } => MessageRecord::new(session_id, MessageRole::Tool, content.clone())
            .with_tool_result(tool_call_id.clone(), name.clone()),
    }
}

/// Converts a stored [`MessageRecord`] back to a provider [`Message`].
///
/// Returns `None` for unrecognized role variants. Preserves tool call IDs
/// and names for tool role messages.
#[must_use]
pub fn record_to_message(record: MessageRecord) -> Option<Message> {
    match record.role {
        MessageRole::System => Some(Message::System {
            content: record.content,
        }),
        MessageRole::User => Some(Message::User {
            content: record.content,
        }),
        MessageRole::Assistant => Some(Message::Assistant {
            content: record.content,
            tool_calls: record.tool_calls,
        }),
        MessageRole::Tool => Some(Message::Tool {
            tool_call_id: record.tool_call_id.unwrap_or_default(),
            name: record.tool_name.unwrap_or_default(),
            content: record.content,
        }),
    }
}
