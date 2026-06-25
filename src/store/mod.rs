//! Persistence layer for sessions, embeddings, artifacts, and executions.
//!
//! This module defines four storage trait abstractions:
//!
//! - [`SessionStore`]: CRUD for conversation sessions and message history
//! - [`EmbeddingStore`]: Vector persistence and nearest-neighbor search
//! - [`ArtifactStore`]: Binary blob storage for files and attachments
//! - [`ExecutionStore`]: Tool execution records, token usage tracking, and session stats
//!
//! Each trait has an in-memory implementation (always available) and
//! feature-gated backend implementations for SQL databases, MongoDB,
//! SurrealDB, Redis, Qdrant, and object stores.
//!
//! # Example
//!
//! ```rust
//! use behest::store::{SessionStore, Session, MessageRecord, MessageRole};
//! use behest::store::memory::MemorySessionStore;
//! use behest::provider::{ModelName, ContentPart};
//! use uuid::Uuid;
//!
//! # async fn example() -> Result<(), behest::StorageError> {
//! let store = MemorySessionStore::new();
//!
//! let session = Session::new("My Chat", ModelName::new("gpt-4"));
//! let session = store.create_session(session).await?;
//!
//! let message = MessageRecord::new(
//!     session.id,
//!     MessageRole::User,
//!     vec![ContentPart::text("Hello!")],
//! );
//! store.append_message(message).await?;
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::error::StorageError;
use crate::provider::{ContentPart, ModelName, TokenUsage, ToolCall};

pub mod memory;
pub(crate) mod util;

#[cfg(any(
    feature = "sqlx-postgres",
    feature = "sqlx-mysql",
    feature = "sqlx-sqlite"
))]
pub mod sql;

#[cfg(feature = "mongodb")]
pub mod mongodb;

#[cfg(feature = "surrealdb")]
pub mod surrealdb;

#[cfg(feature = "redis")]
pub mod redis;

#[cfg(feature = "qdrant")]
pub mod qdrant;

#[cfg(feature = "object_store")]
pub mod object;

/// Convenience result alias for storage operations using [`StorageError`].
pub type StoreResult<T> = std::result::Result<T, StorageError>;

// ---------------------------------------------------------------------------
// Pagination and filter types
// ---------------------------------------------------------------------------

/// Pagination parameters for paginated list operations.
///
/// Uses offset-based pagination with configurable limit and offset.
/// Defaults to returning the first 100 items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pagination {
    /// Maximum number of items to return.
    pub limit: u32,
    /// Number of items to skip (offset-based pagination).
    pub offset: u32,
}

impl Default for Pagination {
    fn default() -> Self {
        Self {
            limit: 100,
            offset: 0,
        }
    }
}

/// Filter parameters for listing sessions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionFilter {
    /// Filter by metadata key-value match (optional).
    pub metadata_filter: Option<Value>,
    /// Filter by created_at range start (inclusive).
    pub created_after: Option<DateTime<Utc>>,
    /// Filter by created_at range end (exclusive).
    pub created_before: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// Session types
// ---------------------------------------------------------------------------

/// Persisted conversation session metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier.
    pub id: Uuid,
    /// Human-readable session title.
    pub title: String,
    /// Model identifier associated with this session.
    pub model: ModelName,
    /// When the session was created.
    pub created_at: DateTime<Utc>,
    /// When the session was last updated.
    pub updated_at: DateTime<Utc>,
    /// Application-specific metadata.
    pub metadata: Value,
}

impl Session {
    /// Creates a new session with a generated UUIDv7 ID and current timestamps.
    ///
    /// The session starts with no metadata (`Value::Null`) and matching
    /// `created_at` / `updated_at` timestamps.
    #[must_use]
    pub fn new(title: impl Into<String>, model: ModelName) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::now_v7(),
            title: title.into(),
            model,
            created_at: now,
            updated_at: now,
            metadata: Value::Null,
        }
    }

    /// Sets application metadata on the session, consuming and returning the session.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Metadata attached to compaction-related messages.
///
/// Compaction produces two messages:
/// 1. A user message with `is_compaction = true` and `compaction_meta.tail_start_id`
///    indicating where the retained tail begins.
/// 2. An assistant message with `is_summary = true` containing the LLM-generated summary.
///
/// The `previous_compaction_id` and `summary_text` fields enable incremental
/// summarization: prior compaction summaries are fed back to the compaction LLM
/// so it can update the summary instead of regenerating from scratch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionMeta {
    /// The first message ID retained in the tail after this compaction.
    pub tail_start_id: Option<Uuid>,
    /// The message ID of the previous compaction user message, enabling incremental updates.
    pub previous_compaction_id: Option<Uuid>,
    /// The summary text generated by the compaction LLM (stored for reuse).
    pub summary_text: Option<String>,
}

impl CompactionMeta {
    /// Creates compaction metadata for a new compaction user message.
    ///
    /// Sets `tail_start_id` to the given ID; `previous_compaction_id`
    /// and `summary_text` are left as `None`.
    #[must_use]
    pub fn new(tail_start_id: Uuid) -> Self {
        Self {
            tail_start_id: Some(tail_start_id),
            previous_compaction_id: None,
            summary_text: None,
        }
    }

    /// Sets the previous compaction ID for incremental summarization, consuming and returning self.
    #[must_use]
    pub fn with_previous(mut self, previous_id: Uuid) -> Self {
        self.previous_compaction_id = Some(previous_id);
        self
    }

    /// Sets the summary text, consuming and returning self.
    #[must_use]
    pub fn with_summary(mut self, summary: String) -> Self {
        self.summary_text = Some(summary);
        self
    }
}

/// Persisted message exchange within a session.
///
/// Each message records a single turn in the conversation: the role (user,
/// assistant, system, tool), its content parts, optional tool calls or tool
/// results, token usage, and compaction state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageRecord {
    /// Unique message identifier.
    pub id: Uuid,
    /// Session this message belongs to.
    pub session_id: Uuid,
    /// Message role.
    pub role: MessageRole,
    /// Message content parts.
    pub content: Vec<ContentPart>,
    /// Tool calls made by the assistant, if any.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Tool call ID for tool result messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool name for tool result messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Token usage associated with this exchange.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// When the message was created.
    pub created_at: DateTime<Utc>,
    /// Whether this message is a compaction task (user message triggering compression).
    #[serde(default)]
    pub is_compaction: bool,
    /// Whether this message contains a compaction summary (assistant response to compaction).
    #[serde(default)]
    pub is_summary: bool,
    /// Compaction metadata, populated only for compaction-related messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_meta: Option<CompactionMeta>,
}

impl MessageRecord {
    /// Creates a new message record with a generated UUIDv7 ID and current timestamp.
    ///
    /// All optional fields (`tool_calls`, `tool_call_id`, `tool_name`, `usage`,
    /// compaction flags) are initialized to their default/absent state.
    #[must_use]
    pub fn new(session_id: Uuid, role: MessageRole, content: Vec<ContentPart>) -> Self {
        Self {
            id: Uuid::now_v7(),
            session_id,
            role,
            content,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            usage: None,
            created_at: Utc::now(),
            is_compaction: false,
            is_summary: false,
            compaction_meta: None,
        }
    }

    /// Sets tool call metadata on a tool result message, consuming and returning self.
    ///
    /// Use this when the message represents a tool execution result rather than
    /// an assistant message that initiates tool calls.
    #[must_use]
    pub fn with_tool_result(mut self, call_id: String, name: String) -> Self {
        self.tool_call_id = Some(call_id);
        self.tool_name = Some(name);
        self
    }

    /// Sets tool calls on the message record, consuming and returning self.
    ///
    /// Use this when the assistant message initiates one or more tool calls.
    #[must_use]
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    /// Sets token usage on the message record, consuming and returning self.
    #[must_use]
    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Marks this message as a compaction task user message, consuming and returning self.
    #[must_use]
    pub fn with_compaction(mut self, meta: CompactionMeta) -> Self {
        self.is_compaction = true;
        self.compaction_meta = Some(meta);
        self
    }

    /// Marks this message as a compaction summary assistant message, consuming and returning self.
    #[must_use]
    pub fn with_summary(mut self, meta: CompactionMeta) -> Self {
        self.is_summary = true;
        self.compaction_meta = Some(meta);
        self
    }
}

impl From<&crate::provider::Message> for MessageRole {
    fn from(message: &crate::provider::Message) -> Self {
        match message {
            crate::provider::Message::System { .. } => Self::System,
            crate::provider::Message::User { .. } => Self::User,
            crate::provider::Message::Assistant { .. } => Self::Assistant,
            crate::provider::Message::Tool { .. } => Self::Tool,
        }
    }
}

/// Role tag for a persisted message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MessageRole {
    /// System instruction.
    System,
    /// User input.
    User,
    /// Assistant response.
    Assistant,
    /// Tool result.
    Tool,
}

// ---------------------------------------------------------------------------
// Embedding types
// ---------------------------------------------------------------------------

/// Embedding record with a dense vector and associated metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRecord {
    /// Unique record identifier.
    pub id: Uuid,
    /// Optional session association.
    pub session_id: Option<Uuid>,
    /// Model that produced the embedding.
    pub model: String,
    /// Dense embedding vector (list of `f32` components).
    pub vector: Vec<f32>,
    /// Application-specific metadata.
    pub metadata: Value,
    /// When the record was created.
    pub created_at: DateTime<Utc>,
}

impl EmbeddingRecord {
    /// Creates a new embedding record with a generated UUIDv7 ID and current timestamp.
    ///
    /// `session_id` and `metadata` are initialized to `None` and `Value::Null` respectively.
    #[must_use]
    pub fn new(model: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id: Uuid::now_v7(),
            session_id: None,
            model: model.into(),
            vector,
            metadata: Value::Null,
            created_at: Utc::now(),
        }
    }

    /// Associates this embedding record with a session, consuming and returning self.
    #[must_use]
    pub fn with_session(mut self, session_id: Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Sets metadata on the embedding record, consuming and returning self.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Embedding search result pairing a matching record with its similarity score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredEmbedding {
    /// The matching embedding record.
    pub record: EmbeddingRecord,
    /// Similarity score (higher means closer match).
    pub score: f32,
}

// ---------------------------------------------------------------------------
// Artifact types
// ---------------------------------------------------------------------------

/// Binary artifact stored by the agent runtime, such as files, images, or attachments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Artifact {
    /// Unique artifact identifier.
    pub id: Uuid,
    /// Optional session association.
    pub session_id: Option<Uuid>,
    /// Human-readable artifact name.
    pub name: String,
    /// MIME content type.
    pub content_type: String,
    /// Raw binary data.
    #[serde(with = "base64_bytes")]
    pub data: Vec<u8>,
    /// Application-specific metadata.
    pub metadata: Value,
    /// When the artifact was created.
    pub created_at: DateTime<Utc>,
}

impl Artifact {
    /// Creates a new artifact with a generated UUIDv7 ID and current timestamp.
    ///
    /// `session_id` and `metadata` are initialized to `None` and `Value::Null` respectively.
    #[must_use]
    pub fn new(name: impl Into<String>, content_type: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            id: Uuid::now_v7(),
            session_id: None,
            name: name.into(),
            content_type: content_type.into(),
            data,
            metadata: Value::Null,
            created_at: Utc::now(),
        }
    }

    /// Associates this artifact with a session, consuming and returning self.
    #[must_use]
    pub fn with_session(mut self, session_id: Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Sets metadata on the artifact, consuming and returning self.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Base64 encoding/decoding module for `Vec<u8>` serialization.
mod base64_bytes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S>(data: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .encode(data)
            .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        use base64::Engine as _;
        let s = String::deserialize(deserializer)?;
        base64::engine::general_purpose::STANDARD
            .decode(&s)
            .map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Persists conversation sessions and their message history.
///
/// Implementations must be `Send + Sync` to support concurrent access
/// from the agent runtime. Backends are selected at compile time via
/// Cargo feature flags.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Persists a new session and returns it with server-assigned fields.
    async fn create_session(&self, session: Session) -> StoreResult<Session>;

    /// Returns all sessions ordered by most recently updated (descending).
    async fn list_sessions(&self) -> StoreResult<Vec<Session>>;

    /// Returns a session by identifier, or `None` if not found.
    async fn get_session(&self, id: &Uuid) -> StoreResult<Option<Session>>;

    /// Deletes a session and all related data (cascading delete).
    ///
    /// Implementations MUST cascade the deletion to at minimum:
    /// - All messages belonging to the session
    /// - All tool executions belonging to the session
    /// - All usage records belonging to the session
    ///
    /// Embedding associations should be set to `NULL` or deleted.
    /// Artifacts associated with the session should be deleted.
    ///
    /// This operation is idempotent: deleting a non-existent session
    /// succeeds silently.
    async fn delete_session(&self, id: &Uuid) -> StoreResult<()>;

    /// Updates a session's title and/or model.
    ///
    /// Fields set to `None` are left unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::NotFound`] when the session does not exist.
    /// Default implementation returns [`StorageError::BackendError`]; backends
    /// should override this with a proper implementation.
    async fn update_session(
        &self,
        id: &Uuid,
        title: &str,
        model: Option<&ModelName>,
    ) -> StoreResult<Session> {
        let _ = (id, title, model);
        Err(StorageError::BackendError {
            backend: "session".to_owned(),
            message: "update_session not implemented for this backend".to_owned(),
            source: None,
        })
    }

    /// Appends a message to a session's history.
    async fn append_message(&self, message: MessageRecord) -> StoreResult<MessageRecord>;

    /// Returns all messages for a session ordered by creation time.
    async fn list_messages(&self, session_id: &Uuid) -> StoreResult<Vec<MessageRecord>>;

    /// Updates token usage on an existing message record.
    async fn update_usage(&self, message_id: &Uuid, usage: TokenUsage) -> StoreResult<()>;

    /// Returns sessions matching the filter, paginated.
    ///
    /// Default implementation calls [`list_sessions`](SessionStore::list_sessions)
    /// and filters/paginates in memory. Backends should override with native
    /// implementations for efficiency.
    async fn list_sessions_paginated(
        &self,
        pagination: Pagination,
        filter: SessionFilter,
    ) -> StoreResult<Vec<Session>> {
        let sessions = self.list_sessions().await?;
        let filtered: Vec<Session> = sessions
            .into_iter()
            .filter(|s| {
                if let Some(ref after) = filter.created_after {
                    if s.created_at < *after {
                        return false;
                    }
                }
                if let Some(ref before) = filter.created_before {
                    if s.created_at >= *before {
                        return false;
                    }
                }
                if let Some(ref meta_filter) = filter.metadata_filter {
                    return s.metadata == *meta_filter;
                }
                true
            })
            .skip(pagination.offset as usize)
            .take(pagination.limit as usize)
            .collect();
        Ok(filtered)
    }

    /// Returns messages for a session, paginated.
    ///
    /// Default implementation calls [`list_messages`](SessionStore::list_messages)
    /// and paginates in memory. Backends should override with native
    /// implementations for efficiency.
    async fn list_messages_paginated(
        &self,
        session_id: &Uuid,
        pagination: Pagination,
    ) -> StoreResult<Vec<MessageRecord>> {
        let messages = self.list_messages(session_id).await?;
        Ok(messages
            .into_iter()
            .skip(pagination.offset as usize)
            .take(pagination.limit as usize)
            .collect())
    }

    /// Checks connectivity to the backend.
    ///
    /// Returns `Ok(())` if the backend is reachable and responsive.
    /// Default implementation returns `Ok(())`.
    async fn health_check(&self) -> StoreResult<()> {
        Ok(())
    }

    /// Returns the most recent compaction user message for a session.
    ///
    /// Used for incremental summarization: prior compaction summaries are
    /// fed back to the compaction LLM so it can update the summary instead
    /// of regenerating from scratch.
    ///
    /// Default implementation iterates [`list_messages`](SessionStore::list_messages)
    /// in reverse; backends should override with a native indexed query.
    async fn get_latest_compaction(&self, session_id: &Uuid) -> StoreResult<Option<MessageRecord>> {
        let messages = self.list_messages(session_id).await?;
        Ok(messages.into_iter().rev().find(|m| m.is_compaction))
    }

    /// Marks a message as compacted, setting its `is_summary` flag.
    ///
    /// Used after the compaction LLM produces a summary to signal that
    /// older tool outputs upstream of this message are eligible for pruning.
    ///
    /// Default implementation is a no-op; backends that need to track
    /// compaction state for pruning should override.
    async fn mark_compacted(&self, _message_id: &Uuid) -> StoreResult<()> {
        Ok(())
    }
}

/// Persists embedding vectors and supports nearest-neighbor search.
///
/// Implementations use cosine similarity for ranking results (higher score = closer match).
/// Backends include in-memory HashMap, PostgreSQL with pgvector, and Qdrant.
#[async_trait]
pub trait EmbeddingStore: Send + Sync {
    /// Inserts or updates an embedding record.
    ///
    /// If a record with the same ID already exists, it is replaced.
    /// Returns the stored record on success.
    async fn upsert(&self, record: EmbeddingRecord) -> StoreResult<EmbeddingRecord>;

    /// Returns the `limit` nearest neighbors to the query vector.
    ///
    /// Each result includes the record and its similarity score (higher is closer).
    async fn search(&self, query: &[f32], limit: usize) -> StoreResult<Vec<ScoredEmbedding>>;

    /// Deletes an embedding by identifier.
    ///
    /// Returns `Ok(())` even when the ID does not exist (idempotent).
    async fn delete(&self, id: &Uuid) -> StoreResult<()>;

    /// Deletes all embeddings associated with a session.
    ///
    /// Returns the number of records deleted.
    async fn delete_by_session(&self, session_id: &Uuid) -> StoreResult<u64>;
}

/// Stores binary artifacts such as files, images, and attachments.
///
/// Backends include in-memory HashMap, local filesystem, and Amazon S3-compatible
/// object stores (via the `object_store` crate).
#[async_trait]
pub trait ArtifactStore: Send + Sync {
    /// Stores an artifact and returns it with server-assigned fields.
    async fn put(&self, artifact: Artifact) -> StoreResult<Artifact>;

    /// Retrieves an artifact by identifier, or `None` if not found.
    async fn get(&self, id: &Uuid) -> StoreResult<Option<Artifact>>;

    /// Deletes an artifact by identifier (idempotent).
    async fn delete(&self, id: &Uuid) -> StoreResult<()>;

    /// Lists all artifacts associated with a session.
    async fn list_by_session(&self, session_id: &Uuid) -> StoreResult<Vec<Artifact>>;

    /// Deletes all artifacts associated with a session.
    ///
    /// Returns the number of artifacts deleted.
    /// Default implementation is a no-op returning 0.
    async fn delete_by_session(&self, session_id: &Uuid) -> StoreResult<u64> {
        let _ = session_id;
        Ok(0)
    }
}

// ---------------------------------------------------------------------------
// Execution types
// ---------------------------------------------------------------------------

/// Persisted record of a single tool invocation.
///
/// Captures the full lifecycle: input arguments, output result, execution status,
/// wall-clock duration, and any error message on failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolExecution {
    /// Unique execution identifier.
    pub id: Uuid,
    /// Session this execution belongs to.
    pub session_id: Uuid,
    /// Message that triggered the tool call.
    pub message_id: Uuid,
    /// Tool call identifier from the provider.
    pub call_id: String,
    /// Tool name that was executed.
    pub tool_name: String,
    /// JSON arguments passed to the tool.
    pub arguments: Value,
    /// Tool output value, if execution succeeded.
    pub result: Option<Value>,
    /// Execution status.
    pub status: ToolExecutionStatus,
    /// Human-readable error message when status is `Failed`.
    pub error: Option<String>,
    /// Wall-clock execution duration.
    #[serde(with = "duration_millis")]
    pub duration: Duration,
    /// When the execution started.
    pub created_at: DateTime<Utc>,
}

impl ToolExecution {
    /// Creates a new tool execution record with a generated UUIDv7 ID and current timestamp.
    ///
    /// The execution starts with `Pending` status, no result, no error,
    /// and zero duration.
    #[must_use]
    pub fn new(
        session_id: Uuid,
        message_id: Uuid,
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: Value,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            session_id,
            message_id,
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            arguments,
            result: None,
            status: ToolExecutionStatus::Pending,
            error: None,
            duration: Duration::ZERO,
            created_at: Utc::now(),
        }
    }

    /// Marks the execution as successful with the given result and duration, consuming and returning self.
    #[must_use]
    pub fn with_success(mut self, result: Value, duration: Duration) -> Self {
        self.result = Some(result);
        self.status = ToolExecutionStatus::Success;
        self.duration = duration;
        self
    }

    /// Marks the execution as failed with the given error message and duration, consuming and returning self.
    #[must_use]
    pub fn with_failure(mut self, error: impl Into<String>, duration: Duration) -> Self {
        self.error = Some(error.into());
        self.status = ToolExecutionStatus::Failed;
        self.duration = duration;
        self
    }
}

/// Outcome of a tool execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolExecutionStatus {
    /// Execution has not started yet.
    Pending,
    /// Execution completed successfully.
    Success,
    /// Execution failed with an error.
    Failed,
}

/// Detailed token usage record for a single provider interaction.
///
/// Tracks per-request token consumption broken down by provider, model,
/// session, and message for cost accounting and analytics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageRecord {
    /// Unique record identifier.
    pub id: Uuid,
    /// Session this usage belongs to.
    pub session_id: Uuid,
    /// Message that produced this usage.
    pub message_id: Uuid,
    /// Provider that served the request.
    pub provider: String,
    /// Model that served the request.
    pub model: String,
    /// Number of input (prompt) tokens consumed.
    pub input_tokens: u64,
    /// Number of output (completion) tokens consumed.
    pub output_tokens: u64,
    /// Total token count.
    pub total_tokens: u64,
    /// When the usage was recorded.
    pub created_at: DateTime<Utc>,
}

impl UsageRecord {
    /// Creates a usage record from a provider response with a generated UUIDv7 ID and current timestamp.
    ///
    /// The token counts are extracted from the given [`TokenUsage`] struct.
    #[must_use]
    pub fn new(
        session_id: Uuid,
        message_id: Uuid,
        provider: impl Into<String>,
        model: impl Into<String>,
        usage: TokenUsage,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            session_id,
            message_id,
            provider: provider.into(),
            model: model.into(),
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
            created_at: Utc::now(),
        }
    }
}

/// Pre-computed or live-aggregated statistics for a session.
///
/// Provides a summary view including message count, tool execution counts
/// (total / success / failure), cumulative token usage, and average tool
/// execution duration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionStats {
    /// Session identifier.
    pub session_id: Uuid,
    /// Total number of messages in the session.
    pub message_count: u64,
    /// Total number of tool executions.
    pub tool_call_count: u64,
    /// Number of successful tool executions.
    pub tool_success_count: u64,
    /// Number of failed tool executions.
    pub tool_failure_count: u64,
    /// Cumulative input tokens across all provider calls.
    pub total_input_tokens: u64,
    /// Cumulative output tokens across all provider calls.
    pub total_output_tokens: u64,
    /// Cumulative total tokens.
    pub total_tokens: u64,
    /// Average tool execution duration in milliseconds.
    pub avg_tool_duration_ms: u64,
}

impl SessionStats {
    /// Creates an all-zero stats struct for a session.
    ///
    /// All counters and the average duration are initialized to zero.
    #[must_use]
    pub fn empty(session_id: Uuid) -> Self {
        Self {
            session_id,
            message_count: 0,
            tool_call_count: 0,
            tool_success_count: 0,
            tool_failure_count: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_tokens: 0,
            avg_tool_duration_ms: 0,
        }
    }
}

/// Persists tool execution records and token usage, and provides session analytics.
///
/// Implementations must be `Send + Sync`. Backends include in-memory, PostgreSQL,
/// MySQL, and SQLite (via `sqlx`).
#[async_trait]
pub trait ExecutionStore: Send + Sync {
    /// Records a tool execution and returns it with server-assigned fields.
    async fn record_execution(&self, execution: ToolExecution) -> StoreResult<ToolExecution>;

    /// Returns all tool executions for a session ordered by creation time (ascending).
    async fn list_executions(&self, session_id: &Uuid) -> StoreResult<Vec<ToolExecution>>;

    /// Returns all tool executions for a specific message ordered by creation time.
    async fn list_executions_by_message(
        &self,
        message_id: &Uuid,
    ) -> StoreResult<Vec<ToolExecution>>;

    /// Records token usage for a single provider interaction.
    async fn record_usage(&self, record: UsageRecord) -> StoreResult<UsageRecord>;

    /// Returns all usage records for a session ordered by creation time (ascending).
    async fn list_usage(&self, session_id: &Uuid) -> StoreResult<Vec<UsageRecord>>;

    /// Computes aggregate statistics for a session.
    ///
    /// Returns pre-computed or live-aggregated stats including message counts,
    /// tool call counts, and cumulative token usage.
    async fn session_stats(&self, session_id: &Uuid) -> StoreResult<SessionStats>;

    /// Deletes all executions and usage records for a session.
    ///
    /// Returns the number of records deleted.
    /// Default implementation is a no-op returning 0.
    async fn delete_by_session(&self, session_id: &Uuid) -> StoreResult<u64> {
        let _ = session_id;
        Ok(0)
    }
}

/// Serde helper for `Duration` as milliseconds.
mod duration_millis {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub(super) fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_millis().serialize(serializer)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}
