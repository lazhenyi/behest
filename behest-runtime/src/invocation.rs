//! Transport-neutral runtime invocation facade.
//!
//! This module exposes a Socket.IO-inspired `emit` / `on` interaction model
//! over [`AgentRuntime`] without binding to any wire protocol. Transport
//! adapters (Socket.IO, gRPC, SSE, ...) build on top of these types; the core
//! runtime stays free of transport concerns.
//!
//! # emit / on semantics
//!
//! [`emit`](RuntimeInvocation::emit) starts a run — it builds an
//! [`EmitRequest`], converts it to a [`RunRequest`], and hands it to
//! [`AgentRuntime::run`]. Returns [`RunOutput`] when the run completes.
//!
//! [`on`](RuntimeInvocation::on) subscribes to the runtime's event bus.
//! Every event from _any_ run managed by the same [`AgentRuntime`] is
//! received; the [`EventKind`] filter discards non-matching events before
//! the handler fires. Handlers run on independently spawned tasks so a
//! slow handler does not block event delivery for other subscribers.
//!
//! ## Event ordering
//!
//! A typical agent run produces events in this order:
//!
//! ```text
//! RunStarted
//!   → ContextBuilt
//!   → ModelStarted → TextDelta* → ToolCallStarted? → ToolCallDelta* →
//!     ToolCallCompleted? → ToolExecutionStarted? → ToolExecutionFinished?
//!   → AssistantMessageCommitted → UsageRecorded
//!   → (loop: ContextBuilt → …)
//!   → RunCompleted | RunFailed | RunCancelled
//! ```
//!
//! Every event carries a [`run_id`](AgentEvent::run_id) so handlers can
//! correlate events belonging to the same run.
//!
//! ## Event delivery guarantees
//!
//! - **At-least-once**: events are delivered via a [`tokio::sync::broadcast`]
//!   channel. Lagging receivers miss events (signaled via
//!   [`tokio::sync::broadcast::error::RecvError::Lagged`]), which is
//!   silently skipped — the receiver misses those events.
//! - **No backpressure**: `emit` does not wait for `on` handlers to
//!   complete. The runtime publishes events to the broadcast channel
//!   and proceeds immediately.
//! - **Ordered per-run**: events from a single run are always published
//!   in order. Events from concurrent runs may interleave.
//! - **Handler concurrency**: if [`Control::set_concurrency_limit`] is set,
//!   a semaphore gates how many handler tasks may run concurrently.
//!   Without a limit, every matching event spawns an unbounded task.
//!
//! ## Handler lifecycle
//!
//! [`on`](RuntimeInvocation::on) returns an [`InvocationHandle`]. The
//! underlying listener task runs until:
//!
//! - The [`InvocationHandle`] is dropped (aborts the task), or
//! - The broadcast channel is closed (all [`AgentRuntime`] senders dropped).
//!
//! Handler tasks already dispatched before the listener stops run to
//! completion.
//!
//! ## Cancellation
//!
//! Cancellation is cooperative. [`Control::cancel`] sets a flag.
//! [`emit`](RuntimeInvocation::emit) checks the flag before invoking the
//! request closure and before calling [`AgentRuntime::run`]; the listener
//! loop inside [`on`](RuntimeInvocation::on) checks it before every event
//! dispatch. The underlying runtime does not yet support hard cancellation
//! of an in-flight model call.
//!
//! # Core surface
//!
//! - [`EmitRequest`] builds a run request and hands it to [`AgentRuntime::run`].
//! - [`RuntimeInvocation::on`] subscribes to the runtime event bus and dispatches
//!   matching events to user handlers on independent tasks.
//! - [`Control`] carries cooperative cancellation/timeout/concurrency state,
//!   plus a type-erased extension map for injecting shared data into handlers.
//! - [`InvocationHandle`] aborts its listener on drop.
//!
//! # Transport-adapter surface
//!
//! [`InvocationEvent::Chat`] and the `Chat*` variants of [`EventKind`] are
//! defined here so adapters reuse the same matching logic, but the core
//! `on` implementation only surfaces [`AgentEvent`]s from
//! [`AgentRuntime::subscribe`]; chat-stream events require a streaming
//! chat adapter that populates the `Chat` variant.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;
use tokio::sync::Semaphore;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use uuid::Uuid;

use super::agent::{AgentRuntime, RunOutput};
use super::error::RuntimeError;
use super::event::AgentEvent;
use super::run::{RunId, RunRequest};
use super::stream::RuntimeEventEnvelope;
use behest_provider::{ChatStreamEvent, ModelName, ProviderId, ToolChoice};

/// Transport-neutral request for a single runtime invocation.
///
/// Converts to [`RunRequest`] via [`EmitRequest::into_run_request`]. The
/// invocation layer deliberately omits `tool_choice` and `run_id`: callers
/// needing fine-grained control over those should construct a [`RunRequest`]
/// directly and call [`AgentRuntime::run`].
#[derive(Debug, Clone)]
pub struct EmitRequest {
    /// Provider used for model calls.
    pub provider: ProviderId,
    /// Model used for generation.
    pub model: ModelName,
    /// User input message.
    pub input: String,
    /// Optional session id. When `None`, the runtime creates a new session.
    pub session_id: Option<Uuid>,
    /// Optional client-provided idempotency key.
    pub client_request_id: Option<String>,
    /// Arbitrary metadata attached to the run. Defaults to [`Value::Null`].
    pub metadata: Value,
}

impl EmitRequest {
    /// Creates a new emit request with no session, no client id, and null metadata.
    #[must_use]
    pub fn new(provider: ProviderId, model: ModelName, input: impl Into<String>) -> Self {
        Self {
            provider,
            model,
            input: input.into(),
            session_id: None,
            client_request_id: None,
            metadata: Value::Null,
        }
    }

    /// Sets the session id.
    #[must_use]
    pub fn with_session_id(mut self, session_id: Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Sets the client-provided idempotency key.
    #[must_use]
    pub fn with_client_request_id(mut self, id: impl Into<String>) -> Self {
        self.client_request_id = Some(id.into());
        self
    }

    /// Sets the metadata payload.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Converts this request into a [`RunRequest`] consumed by the runtime.
    ///
    /// `tool_choice` defaults to [`ToolChoice::Auto`] and `run_id` is left
    /// unset so the runtime allocates one.
    #[must_use]
    pub fn into_run_request(self) -> RunRequest {
        RunRequest {
            session_id: self.session_id,
            run_id: None,
            provider: self.provider,
            model: self.model,
            input: self.input,
            metadata: self.metadata,
            tool_choice: ToolChoice::Auto,
            client_request_id: self.client_request_id,
        }
    }
}

/// Errors raised by the invocation facade.
#[derive(Debug, Error)]
pub enum InvocationError {
    /// Underlying runtime error, propagated from [`AgentRuntime::run`].
    #[error(transparent)]
    Runtime(#[from] RuntimeError),

    /// Invocation task failed (for example, cancelled before completion).
    #[error("invocation task failed: {message}")]
    TaskFailed {
        /// Human-readable failure reason.
        message: String,
    },

    /// Invalid invocation request.
    #[error("invalid invocation request: {message}")]
    InvalidRequest {
        /// Human-readable validation reason.
        message: String,
    },
}

/// Errors returned by [`SessionDataStore`] implementations.
#[derive(Debug, Error)]
pub enum SessionDataError {
    /// The requested session data key was not found.
    #[error("session data key not found: {session_id}/{key}")]
    NotFound {
        /// Session id.
        session_id: Uuid,
        /// Key that was not found.
        key: String,
    },
    /// Underlying storage backend error.
    #[error("session data storage error: {message}")]
    Storage {
        /// Human-readable error description.
        message: String,
    },
}

/// Pluggable backend for per-session temporary key-value data.
///
/// Implementations store ephemeral data associated with a session id.
/// The trait is async so backends like Redis can perform non-blocking I/O.
///
/// # Built-in implementations
///
/// - [`MemorySessionDataStore`] — in-process `HashMap` (default)
/// - [`FileSessionDataStore`] — JSON files on disk (no external deps)
/// - [`RedisSessionDataStore`](crate::session_data_store::RedisSessionDataStore) — Redis hashes (feature = `redis`)
#[async_trait]
pub trait SessionDataStore: Send + Sync {
    /// Stores a value under `(session_id, key)`, overwriting any existing value.
    ///
    /// # Errors
    ///
    /// Returns [`SessionDataError::Storage`] on backend failure.
    async fn set(
        &self,
        session_id: Uuid,
        key: String,
        value: Value,
    ) -> Result<(), SessionDataError>;

    /// Retrieves the value stored under `(session_id, key)`.
    ///
    /// Returns `Ok(None)` when the key does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`SessionDataError::Storage`] on backend failure.
    async fn get(&self, session_id: Uuid, key: &str) -> Result<Option<Value>, SessionDataError>;

    /// Deletes the value stored under `(session_id, key)`.
    ///
    /// Deleting a non-existent key is a no-op (no error).
    ///
    /// # Errors
    ///
    /// Returns [`SessionDataError::Storage`] on backend failure.
    async fn delete(&self, session_id: Uuid, key: &str) -> Result<(), SessionDataError>;
}

/// In-memory [`SessionDataStore`] backed by a `HashMap`.
///
/// Suitable for single-process deployments and testing. Data does not
/// survive process restarts.
#[derive(Clone)]
pub struct MemorySessionDataStore {
    data: Arc<Mutex<HashMap<(Uuid, String), Value>>>,
}

impl Default for MemorySessionDataStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MemorySessionDataStore {
    /// Creates a new empty in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl fmt::Debug for MemorySessionDataStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemorySessionDataStore")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl SessionDataStore for MemorySessionDataStore {
    async fn set(
        &self,
        session_id: Uuid,
        key: String,
        value: Value,
    ) -> Result<(), SessionDataError> {
        let mut map = self
            .data
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        map.insert((session_id, key), value);
        Ok(())
    }

    async fn get(&self, session_id: Uuid, key: &str) -> Result<Option<Value>, SessionDataError> {
        let map = self
            .data
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(map.get(&(session_id, key.to_string())).cloned())
    }

    async fn delete(&self, session_id: Uuid, key: &str) -> Result<(), SessionDataError> {
        let mut map = self
            .data
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        map.remove(&(session_id, key.to_string()));
        Ok(())
    }
}

/// File-system [`SessionDataStore`] backed by JSON files.
///
/// Each session's data is stored in a single JSON file under the configured
/// directory. No external dependencies — uses `tokio::task::spawn_blocking`
/// with `std::fs` for blocking I/O. Per-session locking prevents concurrent
/// writes to the same file.
pub struct FileSessionDataStore {
    base_dir: PathBuf,
    locks: Arc<Mutex<HashMap<Uuid, Arc<Mutex<()>>>>>,
}

impl FileSessionDataStore {
    /// Creates a new file-backed store rooted at `base_dir`.
    ///
    /// The directory is created lazily on the first `set` call.
    #[must_use]
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            locks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn session_path(&self, session_id: Uuid) -> PathBuf {
        self.base_dir.join(format!("{session_id}.json"))
    }

    fn session_lock(&self, session_id: Uuid) -> Arc<Mutex<()>> {
        let mut map = self
            .locks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        map.entry(session_id)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }
}

impl fmt::Debug for FileSessionDataStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileSessionDataStore")
            .field("base_dir", &self.base_dir)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl SessionDataStore for FileSessionDataStore {
    async fn set(
        &self,
        session_id: Uuid,
        key: String,
        value: Value,
    ) -> Result<(), SessionDataError> {
        let path = self.session_path(session_id);
        let lock = self.session_lock(session_id);
        let base = self.base_dir.clone();

        tokio::task::spawn_blocking(move || {
            let _guard = lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::fs::create_dir_all(&base).map_err(|e| SessionDataError::Storage {
                message: format!("failed to create base dir: {e}"),
            })?;

            let mut map: HashMap<String, Value> = if path.exists() {
                let data =
                    std::fs::read_to_string(&path).map_err(|e| SessionDataError::Storage {
                        message: format!("failed to read session file: {e}"),
                    })?;
                serde_json::from_str(&data).map_err(|e| SessionDataError::Storage {
                    message: format!("failed to parse session file: {e}"),
                })?
            } else {
                HashMap::new()
            };

            map.insert(key, value);
            let json =
                serde_json::to_string_pretty(&map).map_err(|e| SessionDataError::Storage {
                    message: format!("failed to serialize session data: {e}"),
                })?;
            std::fs::write(&path, json).map_err(|e| SessionDataError::Storage {
                message: format!("failed to write session file: {e}"),
            })?;
            Ok(())
        })
        .await
        .map_err(|e| SessionDataError::Storage {
            message: format!("spawn_blocking error: {e}"),
        })?
    }

    async fn get(&self, session_id: Uuid, key: &str) -> Result<Option<Value>, SessionDataError> {
        let path = self.session_path(session_id);
        let lock = self.session_lock(session_id);
        let key = key.to_string();

        tokio::task::spawn_blocking(move || {
            let _guard = lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !path.exists() {
                return Ok(None);
            }
            let data = std::fs::read_to_string(&path).map_err(|e| SessionDataError::Storage {
                message: format!("failed to read session file: {e}"),
            })?;
            let map: HashMap<String, Value> =
                serde_json::from_str(&data).map_err(|e| SessionDataError::Storage {
                    message: format!("failed to parse session file: {e}"),
                })?;
            Ok(map.get(&key).cloned())
        })
        .await
        .map_err(|e| SessionDataError::Storage {
            message: format!("spawn_blocking error: {e}"),
        })?
    }

    async fn delete(&self, session_id: Uuid, key: &str) -> Result<(), SessionDataError> {
        let path = self.session_path(session_id);
        let lock = self.session_lock(session_id);
        let key = key.to_string();

        tokio::task::spawn_blocking(move || {
            let _guard = lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !path.exists() {
                return Ok(());
            }
            let data = std::fs::read_to_string(&path).map_err(|e| SessionDataError::Storage {
                message: format!("failed to read session file: {e}"),
            })?;
            let mut map: HashMap<String, Value> =
                serde_json::from_str(&data).map_err(|e| SessionDataError::Storage {
                    message: format!("failed to parse session file: {e}"),
                })?;
            map.remove(&key);
            let json =
                serde_json::to_string_pretty(&map).map_err(|e| SessionDataError::Storage {
                    message: format!("failed to serialize session data: {e}"),
                })?;
            std::fs::write(&path, json).map_err(|e| SessionDataError::Storage {
                message: format!("failed to write session file: {e}"),
            })?;
            Ok(())
        })
        .await
        .map_err(|e| SessionDataError::Storage {
            message: format!("spawn_blocking error: {e}"),
        })?
    }
}

/// Unified event envelope wrapping runtime and chat-stream events.
///
/// The core `on` loop only produces the [`Agent`](InvocationEvent::Agent)
/// variant from [`AgentRuntime::subscribe`]. The [`Chat`](InvocationEvent::Chat)
/// variant is populated by transport adapters that surface provider streams.
#[derive(Debug, Clone)]
pub enum InvocationEvent {
    /// Event from the agent runtime event bus.
    Agent(AgentEvent),
    /// Event from a chat stream. Populated by transport adapters.
    Chat(ChatStreamEvent),
}

impl InvocationEvent {
    /// Returns the run id when derivable from an [`AgentEvent`].
    ///
    /// Chat-stream events carry no run id; `None` is returned for them.
    #[must_use]
    pub fn run_id(&self) -> Option<RunId> {
        match self {
            InvocationEvent::Agent(e) => Some(e.run_id()),
            InvocationEvent::Chat(_) => None,
        }
    }

    /// Returns the wrapped [`AgentEvent`] when this is the [`Agent`](Self::Agent) variant.
    #[must_use]
    pub fn as_agent(&self) -> Option<&AgentEvent> {
        match self {
            InvocationEvent::Agent(e) => Some(e),
            InvocationEvent::Chat(_) => None,
        }
    }
}

/// Kind of event a caller can subscribe to via [`RuntimeInvocation::on`].
///
/// Variants are partitioned into agent-runtime events (mirroring [`AgentEvent`])
/// and chat-stream events (mirroring [`ChatStreamEvent`]). [`EventKind::Any`]
/// matches every event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// Matches any event.
    Any,
    /// Agent run started.
    RunStarted,
    /// Context built.
    ContextBuilt,
    /// Model call started.
    ModelStarted,
    /// Text delta from model.
    TextDelta,
    /// Tool call started.
    ToolCallStarted,
    /// Tool call arguments delta.
    ToolCallDelta,
    /// Tool call completed.
    ToolCallCompleted,
    /// Tool execution started.
    ToolExecutionStarted,
    /// Tool execution finished.
    ToolExecutionFinished,
    /// Assistant message committed to store.
    AssistantMessageCommitted,
    /// Tool message committed to store.
    ToolMessageCommitted,
    /// Usage recorded.
    UsageRecorded,
    /// Prompt cache metrics from a single model call.
    CacheMetrics,
    /// Run completed successfully.
    RunCompleted,
    /// Run failed.
    RunFailed,
    /// Run cancelled.
    RunCancelled,
    /// Doom loop detected.
    DoomLoopDetected,
    /// Compaction circuit breaker opened.
    CompactionCircuitOpened,
    /// Chat stream started.
    ChatStarted,
    /// Chat stream text delta.
    ChatTextDelta,
    /// Chat stream tool call started.
    ChatToolCallStarted,
    /// Chat stream tool call arguments delta.
    ChatToolCallArgumentsDelta,
    /// Chat stream tool call completed.
    ChatToolCallCompleted,
    /// Chat stream finished.
    ChatFinished,
}

impl EventKind {
    /// Returns `true` when `event` matches this kind.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn matches_agent(self, event: &AgentEvent) -> bool {
        match self {
            Self::Any => true,
            Self::RunStarted => matches!(event, AgentEvent::RunStarted(_)),
            Self::ContextBuilt => matches!(event, AgentEvent::ContextBuilt(_)),
            Self::ModelStarted => matches!(event, AgentEvent::ModelStarted(_)),
            Self::TextDelta => matches!(event, AgentEvent::TextDelta(_)),
            Self::ToolCallStarted => matches!(event, AgentEvent::ToolCallStarted(_)),
            Self::ToolCallDelta => matches!(event, AgentEvent::ToolCallDelta(_)),
            Self::ToolCallCompleted => matches!(event, AgentEvent::ToolCallCompleted(_)),
            Self::ToolExecutionStarted => matches!(event, AgentEvent::ToolExecutionStarted(_)),
            Self::ToolExecutionFinished => matches!(event, AgentEvent::ToolExecutionFinished(_)),
            Self::AssistantMessageCommitted => {
                matches!(event, AgentEvent::AssistantMessageCommitted(_))
            }
            Self::ToolMessageCommitted => matches!(event, AgentEvent::ToolMessageCommitted(_)),
            Self::UsageRecorded => matches!(event, AgentEvent::UsageRecorded(_)),
            Self::CacheMetrics => matches!(event, AgentEvent::CacheMetrics(_)),
            Self::RunCompleted => matches!(event, AgentEvent::RunCompleted(_)),
            Self::RunFailed => matches!(event, AgentEvent::RunFailed(_)),
            Self::RunCancelled => matches!(event, AgentEvent::RunCancelled(_)),
            Self::DoomLoopDetected => matches!(event, AgentEvent::DoomLoopDetected(_)),
            Self::CompactionCircuitOpened => {
                matches!(event, AgentEvent::CompactionCircuitOpened(_))
            }
            // Chat variants never match raw AgentEvent
            Self::ChatStarted
            | Self::ChatTextDelta
            | Self::ChatToolCallStarted
            | Self::ChatToolCallArgumentsDelta
            | Self::ChatToolCallCompleted
            | Self::ChatFinished => false,
        }
    }

    /// Returns `true` when `event` matches this kind.
    #[must_use]
    pub fn matches(self, event: &InvocationEvent) -> bool {
        match event {
            InvocationEvent::Agent(e) => self.matches_agent(e),
            InvocationEvent::Chat(e) => self.matches_chat(e),
        }
    }

    /// Returns `true` when `event` matches this kind's chat variant.
    #[must_use]
    fn matches_chat(self, event: &ChatStreamEvent) -> bool {
        match self {
            Self::Any => true,
            Self::ChatStarted => matches!(event, ChatStreamEvent::Started { .. }),
            Self::ChatTextDelta => matches!(event, ChatStreamEvent::TextDelta { .. }),
            Self::ChatToolCallStarted => matches!(event, ChatStreamEvent::ToolCallStarted { .. }),
            Self::ChatToolCallArgumentsDelta => {
                matches!(event, ChatStreamEvent::ToolCallArgumentsDelta { .. })
            }
            Self::ChatToolCallCompleted => {
                matches!(event, ChatStreamEvent::ToolCallCompleted { .. })
            }
            Self::ChatFinished => matches!(event, ChatStreamEvent::Finished { .. }),
            _ => false,
        }
    }
}

/// Invocation-time session context with temporary KV storage.
///
/// Passed as the second argument to `on` handlers and the first argument to
/// `emit` closures. Carries `session_id`, `run_id`, and a pluggable
/// [`SessionDataStore`] backend for ephemeral per-session data.
pub struct InvocationSession {
    /// Session id, when known.
    pub session_id: Option<Uuid>,
    /// Run id, when known.
    pub run_id: Option<RunId>,
    store: Arc<dyn SessionDataStore>,
}

impl InvocationSession {
    /// Stores a temporary value in the session data store.
    ///
    /// # Errors
    ///
    /// Returns [`SessionDataError`] when the session id is unknown or the
    /// backend fails.
    pub async fn set_data(
        &self,
        key: impl Into<String>,
        value: Value,
    ) -> Result<(), SessionDataError> {
        let session_id = self.session_id.ok_or(SessionDataError::Storage {
            message: "session_id not available".into(),
        })?;
        self.store.set(session_id, key.into(), value).await
    }

    /// Retrieves a temporary value from the session data store.
    ///
    /// Returns `Ok(None)` when the key does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`SessionDataError`] when the session id is unknown or the
    /// backend fails.
    pub async fn get_data(&self, key: &str) -> Result<Option<Value>, SessionDataError> {
        let session_id = self.session_id.ok_or(SessionDataError::Storage {
            message: "session_id not available".into(),
        })?;
        self.store.get(session_id, key).await
    }

    /// Deletes a temporary value from the session data store.
    ///
    /// Deleting a non-existent key is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`SessionDataError`] when the session id is unknown or the
    /// backend fails.
    pub async fn delete_data(&self, key: &str) -> Result<(), SessionDataError> {
        let session_id = self.session_id.ok_or(SessionDataError::Storage {
            message: "session_id not available".into(),
        })?;
        self.store.delete(session_id, key).await
    }
}

impl fmt::Debug for InvocationSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InvocationSession")
            .field("session_id", &self.session_id)
            .field("run_id", &self.run_id)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct ControlInner {
    cancelled: AtomicBool,
    timeout: Mutex<Option<Duration>>,
    concurrency_limit: Mutex<Option<usize>>,
    extensions: Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

/// Cooperative lifecycle handle shared across an invocation.
///
/// Cloning a `Control` shares the same cancellation and limit state. Cancellation
/// is cooperative: [`RuntimeInvocation::emit`] checks [`Control::is_cancelled`]
/// before invoking the closure and before calling [`AgentRuntime::run`]; the
/// underlying runtime does not yet support hard cancellation. Timeout and
/// concurrency limit are enforced by event listeners spawned through
/// [`RuntimeInvocation::on`].
///
/// ## Type-erased data
///
/// `Control` also carries an extension map for injecting arbitrary typed data
/// into handlers (similar to `actix-web::web::Data<T>`). Use [`Control::set_data`]
/// and [`Control::data`] to store and retrieve `Arc<T>` values keyed by `TypeId`.
#[derive(Debug, Clone)]
pub struct Control {
    inner: Arc<ControlInner>,
}

impl Default for Control {
    fn default() -> Self {
        Self::new()
    }
}

impl Control {
    /// Creates a fresh control handle: not cancelled, no timeout, no limit.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ControlInner {
                cancelled: AtomicBool::new(false),
                timeout: Mutex::new(None),
                concurrency_limit: Mutex::new(None),
                extensions: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Marks this invocation as cancelled.
    pub fn cancel(&self) {
        self.inner.cancelled.store(true, Ordering::Release);
    }

    /// Returns `true` once [`cancel`](Self::cancel) has been called on any clone.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    /// Sets a provider/runtime call timeout hint.
    pub fn set_timeout(&self, timeout: Duration) {
        *lock_or_recover(&self.inner.timeout) = Some(timeout);
    }

    /// Returns the configured timeout hint, if any.
    #[must_use]
    pub fn timeout(&self) -> Option<Duration> {
        *lock_or_recover(&self.inner.timeout)
    }

    /// Sets a concurrency limit hint.
    pub fn set_concurrency_limit(&self, limit: usize) {
        *lock_or_recover(&self.inner.concurrency_limit) = Some(limit.max(1));
    }

    /// Returns the configured concurrency limit hint, if any.
    #[must_use]
    pub fn concurrency_limit(&self) -> Option<usize> {
        *lock_or_recover(&self.inner.concurrency_limit)
    }

    /// Stores a type-erased value in the extension map.
    ///
    /// Values are keyed by [`TypeId`]; storing a second value of the same
    /// type overwrites the first.
    pub fn set_data<T: Send + Sync + 'static>(&self, val: T) {
        let mut ext = lock_or_recover(&self.inner.extensions);
        ext.insert(TypeId::of::<T>(), Arc::new(val));
    }

    /// Retrieves a previously stored value by type.
    ///
    /// Returns `None` when no value of type `T` has been stored.
    #[must_use]
    pub fn data<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        let ext = lock_or_recover(&self.inner.extensions);
        ext.get(&TypeId::of::<T>())
            .and_then(|arc| Arc::clone(arc).downcast::<T>().ok())
    }
}

/// Recovers from mutex poison by taking the inner guard, avoiding `unwrap`/`expect`.
fn lock_or_recover<T>(lock: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    lock.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Handle to a background listener task started by [`RuntimeInvocation::on`].
///
/// Dropping this handle aborts the listener task to prevent leaks. Inner
/// handler tasks already dispatched before drop run to completion.
#[derive(Debug)]
pub struct InvocationHandle {
    task: JoinHandle<()>,
}

impl InvocationHandle {
    /// Aborts the listener task. Idempotent.
    pub fn abort(&self) {
        self.task.abort();
    }

    /// Returns `true` once the listener task has finished (completed or aborted).
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.task.is_finished()
    }
}

impl Drop for InvocationHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Transport-neutral invocation facade over [`AgentRuntime`].
///
/// Wraps an `Arc<AgentRuntime>` and exposes `emit` / `on` semantics without
/// pulling in any wire protocol. Transport adapters build on top of this.
#[derive(Clone)]
pub struct RuntimeInvocation {
    runtime: Arc<AgentRuntime>,
    session_map: Arc<Mutex<HashMap<RunId, Uuid>>>,
    initial_data: Arc<Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>>,
    session_data_store: Arc<dyn SessionDataStore>,
}

impl RuntimeInvocation {
    /// Wraps a runtime in the invocation facade.
    ///
    /// Uses [`MemorySessionDataStore`] as the default session data backend.
    #[must_use]
    pub fn new(runtime: Arc<AgentRuntime>) -> Self {
        Self {
            runtime,
            session_map: Arc::new(Mutex::new(HashMap::new())),
            initial_data: Arc::new(Mutex::new(HashMap::new())),
            session_data_store: Arc::new(MemorySessionDataStore::new()),
        }
    }

    /// Wraps a runtime with a custom [`SessionDataStore`] backend.
    #[must_use]
    pub fn with_session_store(
        runtime: Arc<AgentRuntime>,
        store: Arc<dyn SessionDataStore>,
    ) -> Self {
        Self {
            runtime,
            session_map: Arc::new(Mutex::new(HashMap::new())),
            initial_data: Arc::new(Mutex::new(HashMap::new())),
            session_data_store: store,
        }
    }

    /// Returns a reference to the underlying runtime.
    #[must_use]
    pub fn runtime(&self) -> &AgentRuntime {
        &self.runtime
    }

    /// Registers a typed value to be injected into every [`Control`] created
    /// by [`emit`](Self::emit) and [`on`](Self::on).
    ///
    /// This is analogous to `actix-web::web::Data<T>` — call it once during
    /// setup, and every handler can retrieve the value via [`Control::data`].
    pub fn set_data<T: Send + Sync + 'static>(&self, val: T) {
        let mut map = lock_or_recover(&self.initial_data);
        map.insert(TypeId::of::<T>(), Arc::new(val));
    }

    /// Creates a [`Control`] pre-filled with [`initial_data`](Self::set_data).
    fn make_control(&self) -> Control {
        let control = Control::new();
        let extensions = lock_or_recover(&self.initial_data);
        let mut target = lock_or_recover(&control.inner.extensions);
        for (type_id, arc) in extensions.iter() {
            target.insert(*type_id, Arc::clone(arc));
        }
        drop(target);
        drop(extensions);
        control
    }

    /// Emits a run request built by `f` and executes it via [`AgentRuntime::run`].
    ///
    /// The closure receives an [`InvocationSession`] (with no session known yet)
    /// and a fresh [`Control`] handle, then returns an [`EmitRequest`]. The
    /// request is converted to a [`RunRequest`] and handed to the runtime.
    ///
    /// Cancellation is cooperative: [`Control::is_cancelled`] is checked before
    /// the closure runs and before `run` is called. If cancelled,
    /// [`InvocationError::TaskFailed`] is returned. Because `Control` is created
    /// internally, external cancellation of `emit` is not exposed by the core
    /// API — transport adapters wrap `emit` to provide that.
    ///
    /// # Errors
    ///
    /// Returns [`InvocationError::TaskFailed`] when cancelled, or
    /// [`InvocationError::Runtime`] when the underlying run fails.
    pub async fn emit<F, Fut>(&self, f: F) -> Result<RunOutput, InvocationError>
    where
        F: FnOnce(InvocationSession, Control) -> Fut + Send,
        Fut: Future<Output = EmitRequest> + Send,
    {
        let control = self.make_control();
        let session = InvocationSession {
            session_id: None,
            run_id: None,
            store: Arc::clone(&self.session_data_store),
        };
        if control.is_cancelled() {
            return Err(InvocationError::TaskFailed {
                message: "cancelled".into(),
            });
        }
        let request = f(session, control.clone()).await;
        if control.is_cancelled() {
            return Err(InvocationError::TaskFailed {
                message: "cancelled".into(),
            });
        }
        let run_request = request.into_run_request();
        let output = self.runtime.run(run_request).await?;
        Ok(output)
    }

    /// Registers an event handler for events matching `kind`.
    ///
    /// Events are sourced from [`AgentRuntime::subscribe`]. When an event
    /// matches, the handler runs on a freshly spawned task so slow handlers
    /// do not block event reception. The returned [`InvocationHandle`] aborts
    /// the listener on drop.
    ///
    /// # Errors
    ///
    /// Returns [`InvocationError::InvalidRequest`] only if subscription setup
    /// is rejected. In the current implementation this always succeeds.
    #[allow(clippy::unused_async)]
    pub async fn on<F, Fut>(
        &self,
        kind: EventKind,
        f: F,
    ) -> Result<InvocationHandle, InvocationError>
    where
        F: Fn(RuntimeEventEnvelope, InvocationSession, Control) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let receiver = self.runtime.subscribe();
        let control = self.make_control();
        let session_map = Arc::clone(&self.session_map);
        let store = Arc::clone(&self.session_data_store);
        Ok(spawn_listener(
            receiver,
            kind,
            control,
            session_map,
            store,
            f,
        ))
    }
}

/// Spawns the event-reception loop backing [`RuntimeInvocation::on`].
///
/// Kept as a free function so tests can drive it with a synthetic
/// [`broadcast::Receiver`] without constructing a full [`AgentRuntime`].
fn spawn_listener<F, Fut>(
    mut receiver: broadcast::Receiver<AgentEvent>,
    kind: EventKind,
    control: Control,
    session_map: Arc<Mutex<HashMap<RunId, Uuid>>>,
    store: Arc<dyn SessionDataStore>,
    handler: F,
) -> InvocationHandle
where
    F: Fn(RuntimeEventEnvelope, InvocationSession, Control) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let handler = Arc::new(handler);
    let concurrency_limit = control.concurrency_limit();
    let semaphore = concurrency_limit.map(|limit| Arc::new(Semaphore::new(limit)));
    let task = tokio::spawn(async move {
        loop {
            match receiver.recv().await {
                Ok(event) => {
                    if control.is_cancelled() {
                        break;
                    }
                    if !kind.matches_agent(&event) {
                        continue;
                    }
                    let run_id = event.run_id();
                    let session_id = {
                        let mut map = lock_or_recover(&session_map);
                        if let AgentEvent::RunStarted(started) = &event {
                            map.insert(run_id, started.session_id);
                            Some(started.session_id)
                        } else {
                            map.get(&run_id).copied()
                        }
                    };
                    let envelope = RuntimeEventEnvelope {
                        event_id: super::stream::RuntimeEventId::new(),
                        seq: 0,
                        run_id,
                        session_id,
                        event,
                        emitted_at: chrono::Utc::now(),
                    };
                    let session = InvocationSession {
                        session_id,
                        run_id: Some(run_id),
                        store: Arc::clone(&store),
                    };
                    let permit = if let Some(sem) = semaphore.clone() {
                        match sem.acquire_owned().await {
                            Ok(permit) => Some(permit),
                            Err(_) => break,
                        }
                    } else {
                        None
                    };
                    let h = Arc::clone(&handler);
                    let c = control.clone();
                    tokio::spawn(async move {
                        let _permit = permit;
                        h(envelope, session, c).await;
                    });
                }
                Err(broadcast::error::RecvError::Closed) => break,
                Err(broadcast::error::RecvError::Lagged(_)) => {}
            }
        }
    });
    InvocationHandle { task }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::event::{RunCompleted, RunStarted, TextDelta};
    use behest_provider::{ChatStreamEvent, FinishReason, ModelName, ProviderId, ToolChoice};
    use chrono::Utc;
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::{Notify, broadcast};
    use uuid::Uuid;

    #[test]
    fn emit_request_converts_to_run_request() {
        let sid = Uuid::new_v4();
        let req = EmitRequest::new(ProviderId::new("p"), ModelName::new("m"), "hi")
            .with_session_id(sid)
            .with_client_request_id("cid")
            .with_metadata(json!({"k": "v"}));
        let run = req.into_run_request();
        assert_eq!(run.provider, ProviderId::new("p"));
        assert_eq!(run.model, ModelName::new("m"));
        assert_eq!(run.input, "hi");
        assert_eq!(run.session_id, Some(sid));
        assert_eq!(run.client_request_id.as_deref(), Some("cid"));
        assert_eq!(run.metadata, json!({"k": "v"}));
        assert!(matches!(run.tool_choice, ToolChoice::Auto));
        assert!(run.run_id.is_none());
    }

    #[test]
    fn emit_request_default_metadata_is_null() {
        let req = EmitRequest::new(ProviderId::new("p"), ModelName::new("m"), "hi");
        assert!(req.metadata.is_null());
        let run = req.clone().into_run_request();
        assert!(run.metadata.is_null());
    }

    #[test]
    fn event_kind_matches_agent_text_delta() {
        let ev = InvocationEvent::Agent(AgentEvent::TextDelta(TextDelta {
            run_id: RunId::new(),
            delta: "x".into(),
            timestamp: Utc::now(),
        }));
        assert!(EventKind::TextDelta.matches(&ev));
        assert!(EventKind::Any.matches(&ev));
        assert!(!EventKind::RunCompleted.matches(&ev));
        assert!(!EventKind::ChatTextDelta.matches(&ev));
    }

    #[test]
    fn event_kind_matches_agent_run_completed() {
        let ev = InvocationEvent::Agent(AgentEvent::RunCompleted(RunCompleted {
            run_id: RunId::new(),
            finish_reason: FinishReason::Stop,
            iterations: 1,
            timestamp: Utc::now(),
        }));
        assert!(EventKind::RunCompleted.matches(&ev));
        assert!(EventKind::Any.matches(&ev));
        assert!(!EventKind::TextDelta.matches(&ev));
    }

    #[test]
    fn event_kind_matches_chat_text_delta() {
        let ev = InvocationEvent::Chat(ChatStreamEvent::TextDelta { delta: "x".into() });
        assert!(EventKind::ChatTextDelta.matches(&ev));
        assert!(EventKind::Any.matches(&ev));
        assert!(!EventKind::TextDelta.matches(&ev));
    }

    #[test]
    fn control_cancel_sets_flag_and_shares_state() {
        let c = Control::new();
        assert!(!c.is_cancelled());
        c.cancel();
        assert!(c.is_cancelled());
        let cloned = c.clone();
        assert!(cloned.is_cancelled(), "clone must share cancel state");
    }

    #[test]
    fn control_set_data_and_data() {
        let c = Control::new();
        c.set_data(42_u32);
        c.set_data(String::from("hello"));
        assert_eq!(*c.data::<u32>().unwrap(), 42);
        assert_eq!(*c.data::<String>().unwrap(), "hello");
        assert!(c.data::<f64>().is_none());
    }

    #[test]
    fn control_data_shared_across_clones() {
        let c = Control::new();
        c.set_data(99_u64);
        let cloned = c.clone();
        assert_eq!(*cloned.data::<u64>().unwrap(), 99);
    }

    #[tokio::test]
    async fn memory_session_data_store_round_trip() {
        let store = MemorySessionDataStore::new();
        let sid = Uuid::new_v4();
        store.set(sid, "k".into(), json!({"x": 1})).await.unwrap();
        let val = store.get(sid, "k").await.unwrap();
        assert_eq!(val, Some(json!({"x": 1})));
        store.delete(sid, "k").await.unwrap();
        assert!(store.get(sid, "k").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn file_session_data_store_round_trip() {
        let dir = std::env::temp_dir().join(format!("behest_test_{}", Uuid::new_v4()));
        let store = FileSessionDataStore::new(&dir);
        let sid = Uuid::new_v4();
        store.set(sid, "name".into(), json!("alice")).await.unwrap();
        let val = store.get(sid, "name").await.unwrap();
        assert_eq!(val, Some(json!("alice")));
        store.delete(sid, "name").await.unwrap();
        assert!(store.get(sid, "name").await.unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn invocation_session_set_get_delete() {
        let session = InvocationSession {
            session_id: Some(Uuid::new_v4()),
            run_id: None,
            store: Arc::new(MemorySessionDataStore::new()),
        };
        session.set_data("key", json!(42)).await.unwrap();
        let val = session.get_data("key").await.unwrap();
        assert_eq!(val, Some(json!(42)));
        session.delete_data("key").await.unwrap();
        assert!(session.get_data("key").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn invocation_session_no_session_id_errors() {
        let session = InvocationSession {
            session_id: None,
            run_id: None,
            store: Arc::new(MemorySessionDataStore::new()),
        };
        let result = session.set_data("key", json!(1)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn on_only_handles_matching_events() {
        let (tx, rx) = broadcast::channel::<AgentEvent>(16);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let handler = move |_, _, _| {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
            }
        };
        let session_map = Arc::new(Mutex::new(HashMap::new()));
        let store: Arc<dyn SessionDataStore> = Arc::new(MemorySessionDataStore::new());
        let handle = spawn_listener(
            rx,
            EventKind::TextDelta,
            Control::new(),
            session_map,
            store,
            handler,
        );

        let _ = tx.send(AgentEvent::TextDelta(TextDelta {
            run_id: RunId::new(),
            delta: "a".into(),
            timestamp: Utc::now(),
        }));
        let _ = tx.send(AgentEvent::RunStarted(RunStarted {
            run_id: RunId::new(),
            session_id: Uuid::new_v4(),
            provider: ProviderId::new("p"),
            model: ModelName::new("m"),
            timestamp: Utc::now(),
        }));

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "only the matching TextDelta event should be handled"
        );
        handle.abort();
    }

    #[test]
    fn control_set_concurrency_limit_clamps_zero_to_one() {
        let control = Control::new();

        control.set_concurrency_limit(0);

        assert_eq!(control.concurrency_limit(), Some(1));
    }

    #[tokio::test]
    async fn listener_applies_concurrency_backpressure_before_spawning_handlers() {
        let (tx, rx) = broadcast::channel::<AgentEvent>(1);
        let handled = Arc::new(AtomicUsize::new(0));
        let first_started = Arc::new(Notify::new());
        let release_first = Arc::new(Notify::new());
        let released = Arc::new(AtomicBool::new(false));

        let h_handled = Arc::clone(&handled);
        let h_first_started = Arc::clone(&first_started);
        let h_release_first = Arc::clone(&release_first);
        let h_released = Arc::clone(&released);
        let handler = move |_, _, _| {
            let handled = Arc::clone(&h_handled);
            let first_started = Arc::clone(&h_first_started);
            let release_first = Arc::clone(&h_release_first);
            let released = Arc::clone(&h_released);
            async move {
                let current = handled.fetch_add(1, Ordering::SeqCst) + 1;
                if current == 1 {
                    first_started.notify_waiters();
                    release_first.notified().await;
                    released.store(true, Ordering::SeqCst);
                    return;
                }

                if !released.load(Ordering::SeqCst) {
                    release_first.notified().await;
                }
            }
        };
        let session_map = Arc::new(Mutex::new(HashMap::new()));
        let store: Arc<dyn SessionDataStore> = Arc::new(MemorySessionDataStore::new());
        let control = Control::new();
        control.set_concurrency_limit(1);
        let handle = spawn_listener(
            rx,
            EventKind::TextDelta,
            control,
            session_map,
            store,
            handler,
        );

        let _ = tx.send(AgentEvent::TextDelta(TextDelta {
            run_id: RunId::new(),
            delta: "first".into(),
            timestamp: Utc::now(),
        }));
        tokio::time::timeout(Duration::from_millis(100), first_started.notified())
            .await
            .expect("first handler should start");

        for idx in 0..20 {
            let _ = tx.send(AgentEvent::TextDelta(TextDelta {
                run_id: RunId::new(),
                delta: format!("queued-{idx}"),
                timestamp: Utc::now(),
            }));
            tokio::task::yield_now().await;
        }

        assert_eq!(handled.load(Ordering::SeqCst), 1);
        release_first.notify_waiters();
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(
            handled.load(Ordering::SeqCst) <= 3,
            "listener should not pre-spawn handlers while the limit is saturated"
        );
        handle.abort();
    }

    #[tokio::test]
    async fn invocation_handle_abort_stops_listener() {
        let (tx, rx) = broadcast::channel::<AgentEvent>(16);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let handler = move |_, _, _| {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
            }
        };
        let session_map = Arc::new(Mutex::new(HashMap::new()));
        let store: Arc<dyn SessionDataStore> = Arc::new(MemorySessionDataStore::new());
        let handle = spawn_listener(
            rx,
            EventKind::Any,
            Control::new(),
            session_map,
            store,
            handler,
        );

        let _ = tx.send(AgentEvent::RunStarted(RunStarted {
            run_id: RunId::new(),
            session_id: Uuid::new_v4(),
            provider: ProviderId::new("p"),
            model: ModelName::new("m"),
            timestamp: Utc::now(),
        }));
        tokio::time::sleep(Duration::from_millis(100)).await;
        let before = counter.load(Ordering::SeqCst);
        assert!(before >= 1, "first event should be handled");

        handle.abort();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(handle.is_finished(), "listener should finish after abort");

        let _ = tx.send(AgentEvent::RunStarted(RunStarted {
            run_id: RunId::new(),
            session_id: Uuid::new_v4(),
            provider: ProviderId::new("p"),
            model: ModelName::new("m"),
            timestamp: Utc::now(),
        }));
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            counter.load(Ordering::SeqCst),
            before,
            "no events should be handled after abort"
        );
    }
}
