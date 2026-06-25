//! Transport-neutral runtime invocation facade.
//!
//! This module exposes a Socket.IO-inspired `emit` / `on` interaction model
//! over [`AgentRuntime`] without binding to any wire protocol. Transport
//! adapters (Socket.IO, gRPC, SSE, ...) build on top of these types; the core
//! runtime stays free of transport concerns.
//!
//! # Core surface
//!
//! - [`EmitRequest`] builds a run request and hands it to [`AgentRuntime::run`].
//! - [`RuntimeInvocation::on`] subscribes to the runtime event bus and dispatches
//!   matching events to user handlers on independent tasks.
//! - [`Control`] carries cooperative cancellation/timeout/concurrency state.
//! - [`InvocationHandle`] aborts its listener on drop.
//!
//! # Transport-adapter surface
//!
//! [`InvocationEvent::Chat`] and the `Chat*` variants of [`EventKind`] are
//! defined here so adapters reuse the same matching logic, but the core
//! `on` implementation only surfaces [`AgentEvent`]s from
//! [`AgentRuntime::subscribe`]; chat-stream events require a streaming
//! chat adapter that populates the `Chat` variant.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;
use thiserror::Error;
use tokio::sync::Semaphore;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::provider::{ChatStreamEvent, ModelName, ProviderId, ToolChoice};
use crate::runtime::agent::{AgentRuntime, RunOutput};
use crate::runtime::error::RuntimeError;
use crate::runtime::event::AgentEvent;
use crate::runtime::run::{RunId, RunRequest};
use crate::runtime::stream::RuntimeEventEnvelope;

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
    #[allow(clippy::too_many_lines)]
    pub fn matches(self, event: &InvocationEvent) -> bool {
        match self {
            Self::Any => true,
            Self::RunStarted => matches!(event, InvocationEvent::Agent(AgentEvent::RunStarted(_))),
            Self::ContextBuilt => {
                matches!(event, InvocationEvent::Agent(AgentEvent::ContextBuilt(_)))
            }
            Self::ModelStarted => {
                matches!(event, InvocationEvent::Agent(AgentEvent::ModelStarted(_)))
            }
            Self::TextDelta => matches!(event, InvocationEvent::Agent(AgentEvent::TextDelta(_))),
            Self::ToolCallStarted => {
                matches!(
                    event,
                    InvocationEvent::Agent(AgentEvent::ToolCallStarted(_))
                )
            }
            Self::ToolCallDelta => {
                matches!(event, InvocationEvent::Agent(AgentEvent::ToolCallDelta(_)))
            }
            Self::ToolCallCompleted => {
                matches!(
                    event,
                    InvocationEvent::Agent(AgentEvent::ToolCallCompleted(_))
                )
            }
            Self::ToolExecutionStarted => {
                matches!(
                    event,
                    InvocationEvent::Agent(AgentEvent::ToolExecutionStarted(_))
                )
            }
            Self::ToolExecutionFinished => {
                matches!(
                    event,
                    InvocationEvent::Agent(AgentEvent::ToolExecutionFinished(_))
                )
            }
            Self::AssistantMessageCommitted => {
                matches!(
                    event,
                    InvocationEvent::Agent(AgentEvent::AssistantMessageCommitted(_))
                )
            }
            Self::ToolMessageCommitted => {
                matches!(
                    event,
                    InvocationEvent::Agent(AgentEvent::ToolMessageCommitted(_))
                )
            }
            Self::UsageRecorded => {
                matches!(event, InvocationEvent::Agent(AgentEvent::UsageRecorded(_)))
            }
            Self::RunCompleted => {
                matches!(event, InvocationEvent::Agent(AgentEvent::RunCompleted(_)))
            }
            Self::RunFailed => matches!(event, InvocationEvent::Agent(AgentEvent::RunFailed(_))),
            Self::RunCancelled => {
                matches!(event, InvocationEvent::Agent(AgentEvent::RunCancelled(_)))
            }
            Self::DoomLoopDetected => {
                matches!(
                    event,
                    InvocationEvent::Agent(AgentEvent::DoomLoopDetected(_))
                )
            }
            Self::CompactionCircuitOpened => {
                matches!(
                    event,
                    InvocationEvent::Agent(AgentEvent::CompactionCircuitOpened(_))
                )
            }
            Self::ChatStarted => matches!(
                event,
                InvocationEvent::Chat(ChatStreamEvent::Started { .. })
            ),
            Self::ChatTextDelta => {
                matches!(
                    event,
                    InvocationEvent::Chat(ChatStreamEvent::TextDelta { .. })
                )
            }
            Self::ChatToolCallStarted => {
                matches!(
                    event,
                    InvocationEvent::Chat(ChatStreamEvent::ToolCallStarted { .. })
                )
            }
            Self::ChatToolCallArgumentsDelta => {
                matches!(
                    event,
                    InvocationEvent::Chat(ChatStreamEvent::ToolCallArgumentsDelta { .. })
                )
            }
            Self::ChatToolCallCompleted => {
                matches!(
                    event,
                    InvocationEvent::Chat(ChatStreamEvent::ToolCallCompleted { .. })
                )
            }
            Self::ChatFinished => {
                matches!(
                    event,
                    InvocationEvent::Chat(ChatStreamEvent::Finished { .. })
                )
            }
        }
    }
}

/// Lightweight invocation-time context passed to `emit` / `on` closures.
///
/// This is not a replacement for [`crate::store::SessionStore`]; it only
/// carries whatever run/session ids can be derived at the call site. Both
/// fields may be `None`.
#[derive(Debug, Clone, Default)]
pub struct SessionContext {
    /// Session id, when known.
    pub session_id: Option<Uuid>,
    /// Run id, when known.
    pub run_id: Option<RunId>,
}

impl SessionContext {
    /// Builds a context from an [`AgentEvent`], deriving `run_id` (and
    /// `session_id` for `RunStarted` events). Other events yield `session_id = None`.
    #[must_use]
    pub fn from_agent_event(event: &AgentEvent) -> Self {
        let session_id = match event {
            AgentEvent::RunStarted(e) => Some(e.session_id),
            _ => None,
        };
        Self {
            session_id,
            run_id: Some(event.run_id()),
        }
    }
}

#[derive(Debug)]
struct ControlInner {
    cancelled: AtomicBool,
    timeout: Mutex<Option<Duration>>,
    concurrency_limit: Mutex<Option<usize>>,
}

/// Cooperative lifecycle handle shared across an invocation.
///
/// Cloning a `Control` shares the same cancellation and limit state. Cancellation
/// is cooperative: [`RuntimeInvocation::emit`] checks [`Control::is_cancelled`]
/// before invoking the closure and before calling [`AgentRuntime::run`]; the
/// underlying runtime does not yet support hard cancellation. Timeout and
/// concurrency limit are stored as hints for transport adapters — the core
/// runtime does not enforce them.
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
        *lock_or_recover(&self.inner.concurrency_limit) = Some(limit);
    }

    /// Returns the configured concurrency limit hint, if any.
    #[must_use]
    pub fn concurrency_limit(&self) -> Option<usize> {
        *lock_or_recover(&self.inner.concurrency_limit)
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
}

impl RuntimeInvocation {
    /// Wraps a runtime in the invocation facade.
    #[must_use]
    pub fn new(runtime: Arc<AgentRuntime>) -> Self {
        Self {
            runtime,
            session_map: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Returns a reference to the underlying runtime.
    #[must_use]
    pub fn runtime(&self) -> &AgentRuntime {
        &self.runtime
    }

    /// Emits a run request built by `f` and executes it via [`AgentRuntime::run`].
    ///
    /// The closure receives an invocation-time [`SessionContext`] (with no
    /// session known yet) and a fresh [`Control`] handle, then returns an
    /// [`EmitRequest`]. The request is converted to a [`RunRequest`] and
    /// handed to the runtime.
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
        F: FnOnce(SessionContext, Control) -> Fut + Send,
        Fut: Future<Output = EmitRequest> + Send,
    {
        let control = Control::new();
        let ctx = SessionContext::default();
        if control.is_cancelled() {
            return Err(InvocationError::TaskFailed {
                message: "cancelled".into(),
            });
        }
        let request = f(ctx, control.clone()).await;
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
        F: Fn(RuntimeEventEnvelope, Control) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let receiver = self.runtime.subscribe();
        let control = Control::new();
        let session_map = Arc::clone(&self.session_map);
        Ok(spawn_listener(receiver, kind, control, session_map, f))
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
    handler: F,
) -> InvocationHandle
where
    F: Fn(RuntimeEventEnvelope, Control) -> Fut + Send + Sync + 'static,
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
                        event_id: crate::runtime::stream::RuntimeEventId::new(),
                        seq: 0,
                        run_id,
                        session_id,
                        event,
                        emitted_at: chrono::Utc::now(),
                    };
                    let h = Arc::clone(&handler);
                    let c = control.clone();
                    let sem = semaphore.clone();
                    tokio::spawn(async move {
                        if let Some(s) = sem {
                            let _permit = s.acquire().await;
                            h(envelope, c).await;
                        } else {
                            h(envelope, c).await;
                        }
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::provider::{ChatStreamEvent, FinishReason, ModelName, ProviderId, ToolChoice};
    use crate::runtime::event::{RunCompleted, RunStarted, TextDelta};
    use chrono::Utc;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::broadcast;
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

    #[tokio::test]
    async fn on_only_handles_matching_events() {
        let (tx, rx) = broadcast::channel::<AgentEvent>(16);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let handler = move |_, _| {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
            }
        };
        let session_map = Arc::new(Mutex::new(HashMap::new()));
        let handle = spawn_listener(
            rx,
            EventKind::TextDelta,
            Control::new(),
            session_map,
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

    #[tokio::test]
    async fn invocation_handle_abort_stops_listener() {
        let (tx, rx) = broadcast::channel::<AgentEvent>(16);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let handler = move |_, _| {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
            }
        };
        let session_map = Arc::new(Mutex::new(HashMap::new()));
        let handle = spawn_listener(rx, EventKind::Any, Control::new(), session_map, handler);

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
