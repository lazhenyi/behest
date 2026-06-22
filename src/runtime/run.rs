//! Run lifecycle types.
//!
//! Defines the core types for tracking agent execution runs,
//! including run identifiers, status states, and request payloads.

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::provider::{ModelName, ProviderId, ToolChoice};

/// Unique identifier for an agent run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RunId(Uuid);

impl RunId {
    /// Creates a new run identifier.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Returns the underlying UUID.
    #[must_use]
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Status of an agent run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    /// Run has been created but not yet started.
    Pending,
    /// Run is loading or validating session state.
    SessionLoaded,
    /// Run is building context from adapters.
    BuildingContext,
    /// Run is calling the model provider.
    CallingModel,
    /// Run is waiting for tool execution.
    WaitingForTools,
    /// Run is persisting results.
    Persisting,
    /// Run completed successfully.
    Completed,
    /// Run failed with an error.
    Failed,
    /// Run was cancelled.
    Cancelled,
}

impl RunStatus {
    /// Returns true if the run is in a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Request to start a new agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRequest {
    /// Optional session ID. If None, a new session will be created.
    pub session_id: Option<Uuid>,
    /// Provider to use for model calls.
    pub provider: ProviderId,
    /// Model to use for generation.
    pub model: ModelName,
    /// User input message.
    pub input: String,
    /// Optional metadata for the run.
    pub metadata: Value,
    /// Tool choice strategy.
    pub tool_choice: ToolChoice,
}

impl RunRequest {
    /// Creates a new run request.
    #[must_use]
    pub fn new(provider: ProviderId, model: ModelName, input: impl Into<String>) -> Self {
        Self {
            session_id: None,
            provider,
            model,
            input: input.into(),
            metadata: Value::Null,
            tool_choice: ToolChoice::Auto,
        }
    }

    /// Sets the session ID.
    #[must_use]
    pub fn with_session_id(mut self, session_id: Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Sets the metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets the tool choice strategy.
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }
}

/// Persistent record of an agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Unique run identifier.
    pub id: RunId,
    /// Session this run belongs to.
    pub session_id: Uuid,
    /// Current status of the run.
    pub status: RunStatus,
    /// Provider used for model calls.
    pub provider: ProviderId,
    /// Model used for generation.
    pub model: ModelName,
    /// Run metadata.
    pub metadata: Value,
    /// When the run was created.
    pub created_at: DateTime<Utc>,
    /// When the run was last updated.
    pub updated_at: DateTime<Utc>,
}

impl RunRecord {
    /// Creates a new run record.
    #[must_use]
    pub fn new(
        id: RunId,
        session_id: Uuid,
        provider: ProviderId,
        model: ModelName,
        metadata: Value,
    ) -> Self {
        let now = Utc::now();
        Self {
            id,
            session_id,
            status: RunStatus::Pending,
            provider,
            model,
            metadata,
            created_at: now,
            updated_at: now,
        }
    }

    /// Updates the status and timestamp.
    pub fn update_status(&mut self, status: RunStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_id_generation() {
        let id1 = RunId::new();
        let id2 = RunId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn run_status_terminal() {
        assert!(!RunStatus::Pending.is_terminal());
        assert!(!RunStatus::CallingModel.is_terminal());
        assert!(RunStatus::Completed.is_terminal());
        assert!(RunStatus::Failed.is_terminal());
        assert!(RunStatus::Cancelled.is_terminal());
    }

    #[test]
    fn run_request_builder() {
        let provider = ProviderId::new("test");
        let model = ModelName::new("gpt-4");
        let request = RunRequest::new(provider.clone(), model.clone(), "hello")
            .with_metadata(Value::String("meta".to_string()))
            .with_tool_choice(ToolChoice::Required);

        assert_eq!(request.provider, provider);
        assert_eq!(request.model, model);
        assert_eq!(request.input, "hello");
        assert_eq!(request.metadata, Value::String("meta".to_string()));
        assert!(matches!(request.tool_choice, ToolChoice::Required));
    }
}
