//! Agent execution snapshotting and recovery.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs;
use uuid::Uuid;

use crate::provider::{FinishReason, Message, TokenUsage};
use crate::runtime::error::{RuntimeError, RuntimeResult};
use crate::runtime::run::{RunId, RunRequest, RunStatus};
use crate::runtime::turn::TurnState;

/// A serialized execution snapshot of an active agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Unique identifier for the run.
    pub run_id: RunId,
    /// Session ID associated with the run.
    pub session_id: Uuid,
    /// Current run status.
    pub status: RunStatus,
    /// Current iteration count.
    pub iteration: usize,
    /// Current turn state within the loop.
    pub current_state: TurnState,
    /// Accumulated token usage so far.
    pub total_usage: TokenUsage,
    /// Last finish reason returned by the model, if any.
    pub last_finish: Option<FinishReason>,
    /// Optional last committed assistant message.
    pub assistant_message: Option<Message>,
    /// Optional ID of the last committed assistant message in the store.
    pub assistant_msg_id: Option<Uuid>,
    /// The original run request.
    pub request: RunRequest,
    /// Number of output recovery attempts made so far for truncated responses.
    #[serde(default)]
    pub output_recovery_count: u32,
    /// Timestamp when this snapshot was captured.
    pub timestamp: DateTime<Utc>,
}

/// Trait for snapshot persistence backends.
#[async_trait]
pub trait SnapshotStore: Send + Sync + 'static {
    /// Saves a snapshot.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` if serialization or disk I/O fails.
    async fn save(&self, snapshot: &Snapshot) -> RuntimeResult<()>;

    /// Loads a snapshot by run ID.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` if disk I/O or deserialization fails.
    async fn load(&self, run_id: RunId) -> RuntimeResult<Option<Snapshot>>;

    /// Deletes a snapshot by run ID.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` if I/O fails.
    async fn delete(&self, run_id: RunId) -> RuntimeResult<()>;

    /// Lists all active snapshots.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` if directory reading fails.
    async fn list(&self) -> RuntimeResult<Vec<Snapshot>>;
}

/// Filesystem-based snapshot storage.
pub struct FileSnapshotStore {
    base_dir: PathBuf,
}

impl FileSnapshotStore {
    /// Creates a new filesystem-backed snapshot store in the given directory.
    #[must_use]
    pub fn new(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }

    fn path_for(&self, run_id: RunId) -> PathBuf {
        self.base_dir.join(format!("snapshot_{run_id}.json"))
    }
}

#[async_trait]
impl SnapshotStore for FileSnapshotStore {
    async fn save(&self, snapshot: &Snapshot) -> RuntimeResult<()> {
        if let Err(e) = fs::create_dir_all(&self.base_dir).await {
            return Err(RuntimeError::RecoveryFailed(format!(
                "failed to create snapshot dir: {e}"
            )));
        }

        let serialized = serde_json::to_string_pretty(snapshot).map_err(|e| {
            RuntimeError::RecoveryFailed(format!("failed to serialize snapshot: {e}"))
        })?;

        let temp_path = self
            .base_dir
            .join(format!("snapshot_{}.json.tmp", snapshot.run_id));
        if let Err(e) = fs::write(&temp_path, serialized).await {
            return Err(RuntimeError::RecoveryFailed(format!(
                "failed to write snapshot: {e}"
            )));
        }

        let final_path = self.path_for(snapshot.run_id);
        fs::rename(temp_path, final_path).await.map_err(|e| {
            RuntimeError::RecoveryFailed(format!("failed to finalize snapshot: {e}"))
        })?;

        Ok(())
    }

    async fn load(&self, run_id: RunId) -> RuntimeResult<Option<Snapshot>> {
        let path = self.path_for(run_id);
        if !path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&path).await.map_err(|e| {
            RuntimeError::RecoveryFailed(format!("failed to read snapshot file: {e}"))
        })?;

        let snapshot: Snapshot = serde_json::from_str(&content).map_err(|e| {
            RuntimeError::RecoveryFailed(format!("failed to deserialize snapshot file: {e}"))
        })?;

        Ok(Some(snapshot))
    }

    async fn delete(&self, run_id: RunId) -> RuntimeResult<()> {
        let path = self.path_for(run_id);
        if path.exists() {
            fs::remove_file(path).await.map_err(|e| {
                RuntimeError::RecoveryFailed(format!("failed to delete snapshot file: {e}"))
            })?;
        }
        Ok(())
    }

    async fn list(&self) -> RuntimeResult<Vec<Snapshot>> {
        if !self.base_dir.exists() {
            return Ok(Vec::new());
        }

        let mut snapshots = Vec::new();
        let mut entries = fs::read_dir(&self.base_dir).await.map_err(|e| {
            RuntimeError::RecoveryFailed(format!("failed to read snapshot dir: {e}"))
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            RuntimeError::RecoveryFailed(format!("failed to read snapshot directory entry: {e}"))
        })? {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "json" {
                        if let Ok(content) = fs::read_to_string(&path).await {
                            if let Ok(snapshot) = serde_json::from_str::<Snapshot>(&content) {
                                snapshots.push(snapshot);
                            }
                        }
                    }
                }
            }
        }

        Ok(snapshots)
    }
}
