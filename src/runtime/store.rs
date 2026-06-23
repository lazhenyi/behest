//! Runtime store facade.
//!
//! Provides a unified interface for runtime persistence operations,
//! composing session, execution, and run stores.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::provider::Message;
use crate::store::{ExecutionStore, MessageRecord, MessageRole, SessionStore};

use super::error::{RuntimeError, RuntimeResult};
use super::event::AgentEvent;
use super::run::{RunId, RunRecord, RunStatus};

/// Persistent record of a run event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEventRecord {
    /// Event sequence number within the run.
    pub sequence: u64,
    /// Run identifier.
    pub run_id: RunId,
    /// Event payload.
    pub event: AgentEvent,
    /// When the event was recorded.
    pub timestamp: DateTime<Utc>,
}

impl RunEventRecord {
    /// Creates a new run event record.
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
#[async_trait]
pub trait RunStore: Send + Sync {
    /// Creates a new run record.
    async fn create_run(&self, record: RunRecord) -> RuntimeResult<()>;

    /// Gets a run by ID.
    async fn get_run(&self, run_id: RunId) -> RuntimeResult<Option<RunRecord>>;

    /// Updates run status.
    async fn update_run_status(&self, run_id: RunId, status: RunStatus) -> RuntimeResult<()>;

    /// Appends an event to a run.
    async fn append_event(&self, record: RunEventRecord) -> RuntimeResult<()>;

    /// Lists events for a run.
    async fn list_events(&self, run_id: RunId) -> RuntimeResult<Vec<RunEventRecord>>;

    /// Lists runs for a session.
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

    /// Deletes a run and its events.
    async fn delete_run(&self, run_id: RunId) -> RuntimeResult<()>;

    /// Health check.
    async fn health_check(&self) -> RuntimeResult<()>;
}

/// Runtime store facade combining session, execution, and run stores.
pub struct RuntimeStore {
    sessions: Box<dyn SessionStore>,
    executions: Box<dyn ExecutionStore>,
    runs: Box<dyn RunStore>,
}

impl RuntimeStore {
    /// Creates a new runtime store.
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
        }
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

/// Converts a provider Message to a store MessageRecord.
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

/// Converts a store MessageRecord back to a provider Message.
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
