//! Layered context traits for the behest agent runtime.
//!
//! Inspired by ADK-Rust's `ReadonlyContext → CallbackContext → InvocationContext`
//! hierarchy, this module defines progressively richer context interfaces that
//! give agents, tools, hooks, and memory operations access to exactly the
//! capabilities they need — no more, no less.
//!
//! # Hierarchy
//!
//! ```text
//! ReadonlyContext         (identity: invocation_id, session_id, user_id, app_name)
//!   ├── SessionContext    (+ session state, session store)
//!   │     ├── RunContext  (+ run_id, cancellation, deadline, event sink, budget)
//!   │     │     └── ToolContext (+ tool_call, emit_progress, search_memory)
//!   │     └── MemoryContext (+ active_window, demote, compact)
//!   └── HookContext       (+ current_state, snapshot)
//! ```
//!
//! # Design constraints
//!
//! - Each layer adds specific, well-defined capabilities.
//! - Context does not become a "global service locator" — each trait is
//!   narrowly scoped to what the consumer needs.
//! - Tool can access session info and stream progress, but cannot access
//!   internal runtime state.
//! - Every trait has clear ownership and lifetime semantics.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]

use std::time::Instant;

use behest_core::id::RunId;
use behest_core::message::Message;
use behest_core::run::RunState;
use behest_core::tool_types::ToolCall;
use serde_json::Value;
use tokio::sync::watch;

mod factory;
mod impls;
pub use factory::{
    ContextAdapter, ContextFactory, ContextInput, ContextOutput, ContextResult, FunctionAdapter,
    StaticAdapter,
};
pub use impls::{
    AppContext, HookContextImpl, MemoryContextImpl, RunContextImpl, SessionContextImpl,
    ToolContextImpl,
};

/// The minimum context available to any participant in the system.
///
/// Provides read-only identity information: who is making the request,
/// which session it belongs to, and which application is running.
pub trait ReadonlyContext {
    /// Returns the unique invocation identifier for this run.
    fn invocation_id(&self) -> &str;

    /// Returns the session identifier.
    fn session_id(&self) -> &str;

    /// Returns the authenticated user identifier.
    fn user_id(&self) -> &str;

    /// Returns the application name.
    fn app_name(&self) -> &str;
}

/// Context available during a session's lifetime.
///
/// Extends [`ReadonlyContext`] with mutable session state and access
/// to the session store for persistence.
pub trait SessionContext: ReadonlyContext {
    /// Returns the current session state (key-value store scoped to the session).
    fn session_state(&self) -> &SessionState;

    /// Returns mutable access to session state.
    fn session_state_mut(&mut self) -> &mut SessionState;
}

/// A key-value state store scoped to a single session.
///
/// Supports typed key prefixes (`user:`, `app:`, `temp:`) for
/// different scoping semantics.
#[derive(Debug, Clone, Default)]
pub struct SessionState {
    entries: std::collections::HashMap<String, Value>,
}

impl SessionState {
    /// Creates an empty session state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a value in the session state.
    pub fn set(&mut self, key: impl Into<String>, value: Value) {
        self.entries.insert(key.into(), value);
    }

    /// Gets a value from the session state.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.get(key)
    }

    /// Removes a value from the session state.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.entries.remove(key)
    }

    /// Returns all entries in the session state.
    #[must_use]
    pub fn all(&self) -> &std::collections::HashMap<String, Value> {
        &self.entries
    }
}

/// Context available during a single run invocation.
///
/// Extends [`SessionContext`] with run-level controls: cancellation,
/// deadlines, event output, and token budget tracking.
pub trait RunContext: SessionContext {
    /// Returns the unique run identifier.
    fn run_id(&self) -> &RunId;

    /// Returns a cancellation token for cooperative cancellation.
    fn cancellation_token(&self) -> &tokio_util::sync::CancellationToken;

    /// Returns the deadline for this run, if any.
    fn deadline(&self) -> Option<Instant>;

    /// Returns the event sink for emitting structured events.
    fn event_sink(&self) -> &EventSink;

    /// Returns the token budget tracker for this run.
    fn budget(&self) -> &RunBudget;
}

/// A sink for emitting events during a run.
///
/// Events are sent to all registered subscribers (local broadcast)
/// and optionally forwarded to external publishers.
#[derive(Clone)]
pub struct EventSink {
    tx: watch::Sender<Option<Value>>,
}

impl std::fmt::Debug for EventSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventSink")
            .field("receiver_count", &self.tx.receiver_count())
            .finish()
    }
}

impl EventSink {
    /// Creates a new event sink.
    #[must_use]
    pub fn new() -> Self {
        let (tx, _) = watch::channel(None);
        Self { tx }
    }

    /// Emits an event to all subscribers.
    ///
    /// Returns the number of active subscribers that received the event.
    pub fn emit(&self, event: Value) -> usize {
        self.tx.send_if_modified(|current| {
            *current = Some(event);
            true
        });
        self.tx.receiver_count()
    }

    /// Creates a receiver for this sink.
    #[must_use]
    pub fn subscribe(&self) -> watch::Receiver<Option<Value>> {
        self.tx.subscribe()
    }
}

impl Default for EventSink {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks the token budget for a single run.
#[derive(Debug, Clone)]
pub struct RunBudget {
    max_tokens: Option<usize>,
    used_tokens: usize,
}

impl RunBudget {
    /// Creates a new budget with an optional maximum.
    #[must_use]
    pub fn new(max_tokens: Option<usize>) -> Self {
        Self {
            max_tokens,
            used_tokens: 0,
        }
    }

    /// Records token usage.
    pub fn consume(&mut self, tokens: usize) {
        self.used_tokens += tokens;
    }

    /// Returns the remaining token budget.
    /// Returns `None` if there is no maximum.
    #[must_use]
    pub fn remaining(&self) -> Option<usize> {
        self.max_tokens
            .map(|max| max.saturating_sub(self.used_tokens))
    }

    /// Returns the total tokens consumed so far.
    #[must_use]
    pub fn used(&self) -> usize {
        self.used_tokens
    }
}

/// Context for tool execution.
///
/// Extends [`RunContext`] with tool-specific capabilities:
/// - Access to the current tool call
/// - Streaming progress output
/// - Memory search (for tools that need context from long-term memory)
pub trait ToolContext: RunContext {
    /// Returns the tool call being executed.
    fn tool_call(&self) -> &ToolCall;

    /// Emits a progress update for the current tool execution.
    ///
    /// Progress events are streamed to consumers so they can observe
    /// long-running tool operations in real time.
    fn emit_progress(&self, status: &str, data: Value) {
        let progress = serde_json::json!({
            "type": "tool_progress",
            "call_id": self.tool_call().id,
            "tool_name": self.tool_call().name,
            "status": status,
            "data": data,
        });
        self.event_sink().emit(progress);
    }

    /// Searches long-term memory for context relevant to the current tool call.
    ///
    /// The default implementation returns an empty result. Implementations
    /// with access to an embedding store should override this.
    fn search_memory(&self, _query: &str, _limit: usize) -> Vec<MemoryEntry> {
        Vec::new()
    }
}

/// An entry retrieved from long-term memory.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// The memory content.
    pub content: String,
    /// Relevance score (higher is more relevant).
    pub score: f32,
    /// Source identifier (e.g., session ID, document name).
    pub source: Option<String>,
}

/// Context for memory operations (demotion, compaction).
///
/// Extends [`SessionContext`] with access to the active window
/// and the ability to demote or compact messages.
pub trait MemoryContext: SessionContext {
    /// Returns the current active window (recent messages in context).
    fn active_window(&self) -> &[Message];

    /// Demotes messages from the active window to long-term storage.
    ///
    /// Returns an error if the demotion hook is not configured or
    /// the storage backend is unavailable.
    fn demote(&self, messages: Vec<Message>) -> Result<(), String>;

    /// Compacts messages into a summary and injects it back into the
    /// active window.
    ///
    /// Returns the generated summary text.
    fn compact(&self, messages: Vec<Message>) -> Result<String, String>;
}

/// Context available to hooks observing the system.
///
/// Extends [`ReadonlyContext`] with the ability to inspect the current
/// run state and take a snapshot for later replay.
pub trait HookContext: ReadonlyContext {
    /// Returns the current run state.
    fn current_state(&self) -> &RunState;

    /// Returns a snapshot of the current run for audit/replay.
    fn snapshot(&self) -> RunSnapshot;
}

/// A snapshot of the current run state for audit and replay.
#[derive(Debug, Clone)]
pub struct RunSnapshot {
    /// The state at the time of the snapshot.
    pub state: RunState,
    /// The run identifier.
    pub run_id: RunId,
    /// The session identifier.
    pub session_id: String,
    /// The number of iterations completed so far.
    pub iteration: usize,
    /// The current token usage.
    pub tokens_used: usize,
}
