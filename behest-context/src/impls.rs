//! Concrete implementations of the layered context traits.
//!
//! These structs provide the actual backing storage and behavior for each
//! context level. They are designed to be composed hierarchically:
//!
//! ```text
//! AppContext → SessionContextImpl → RunContextImpl → ToolContextImpl
//! ```

use std::time::Instant;

use behest_core::id::RunId;
use behest_core::message::Message;
use behest_core::run::RunState;
use behest_core::tool_types::ToolCall;
use tokio_util::sync::CancellationToken;

use crate::{
    EventSink, HookContext, MemoryContext, ReadonlyContext, RunBudget, RunContext, RunSnapshot,
    SessionContext, SessionState, ToolContext,
};

/// Application-level context: global configuration and identity.
#[derive(Debug, Clone)]
pub struct AppContext {
    /// Unique invocation identifier.
    pub invocation_id: String,
    /// Session identifier.
    pub session_id: String,
    /// Authenticated user identifier.
    pub user_id: String,
    /// Application name.
    pub app_name: String,
}

impl ReadonlyContext for AppContext {
    fn invocation_id(&self) -> &str {
        &self.invocation_id
    }

    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn user_id(&self) -> &str {
        &self.user_id
    }

    fn app_name(&self) -> &str {
        &self.app_name
    }
}

/// Session-level context with mutable state.
#[derive(Debug)]
pub struct SessionContextImpl {
    /// The underlying application context.
    pub app: AppContext,
    /// Mutable session state (key-value store).
    pub state: SessionState,
}

impl ReadonlyContext for SessionContextImpl {
    fn invocation_id(&self) -> &str {
        self.app.invocation_id()
    }

    fn session_id(&self) -> &str {
        self.app.session_id()
    }

    fn user_id(&self) -> &str {
        self.app.user_id()
    }

    fn app_name(&self) -> &str {
        self.app.app_name()
    }
}

impl SessionContext for SessionContextImpl {
    fn session_state(&self) -> &SessionState {
        &self.state
    }

    fn session_state_mut(&mut self) -> &mut SessionState {
        &mut self.state
    }
}

/// Run-level context with cancellation, deadline, event sink, and budget.
#[derive(Debug)]
pub struct RunContextImpl {
    /// The underlying session context.
    pub session: SessionContextImpl,
    /// The unique run identifier.
    pub run_id: RunId,
    /// Cancellation token for cooperative cancellation.
    pub cancel: CancellationToken,
    /// Deadline for this run, if any.
    pub deadline: Option<Instant>,
    /// Event sink for emitting structured events.
    pub sink: EventSink,
    /// Token budget tracker.
    pub budget: RunBudget,
}

impl ReadonlyContext for RunContextImpl {
    fn invocation_id(&self) -> &str {
        self.session.invocation_id()
    }

    fn session_id(&self) -> &str {
        self.session.session_id()
    }

    fn user_id(&self) -> &str {
        self.session.user_id()
    }

    fn app_name(&self) -> &str {
        self.session.app_name()
    }
}

impl SessionContext for RunContextImpl {
    fn session_state(&self) -> &SessionState {
        self.session.session_state()
    }

    fn session_state_mut(&mut self) -> &mut SessionState {
        self.session.session_state_mut()
    }
}

impl RunContext for RunContextImpl {
    fn run_id(&self) -> &RunId {
        &self.run_id
    }

    fn cancellation_token(&self) -> &CancellationToken {
        &self.cancel
    }

    fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    fn event_sink(&self) -> &EventSink {
        &self.sink
    }

    fn budget(&self) -> &RunBudget {
        &self.budget
    }
}

/// Tool execution context.
#[derive(Debug)]
pub struct ToolContextImpl {
    /// The underlying run context.
    pub run: RunContextImpl,
    /// The tool call being executed.
    pub tool_call: ToolCall,
}

impl ReadonlyContext for ToolContextImpl {
    fn invocation_id(&self) -> &str {
        self.run.invocation_id()
    }

    fn session_id(&self) -> &str {
        self.run.session_id()
    }

    fn user_id(&self) -> &str {
        self.run.user_id()
    }

    fn app_name(&self) -> &str {
        self.run.app_name()
    }
}

impl SessionContext for ToolContextImpl {
    fn session_state(&self) -> &SessionState {
        self.run.session_state()
    }

    fn session_state_mut(&mut self) -> &mut SessionState {
        self.run.session_state_mut()
    }
}

impl RunContext for ToolContextImpl {
    fn run_id(&self) -> &RunId {
        self.run.run_id()
    }

    fn cancellation_token(&self) -> &CancellationToken {
        self.run.cancellation_token()
    }

    fn deadline(&self) -> Option<Instant> {
        self.run.deadline()
    }

    fn event_sink(&self) -> &EventSink {
        self.run.event_sink()
    }

    fn budget(&self) -> &RunBudget {
        self.run.budget()
    }
}

impl ToolContext for ToolContextImpl {
    fn tool_call(&self) -> &ToolCall {
        &self.tool_call
    }
}

/// Memory context implementation with active window management.
pub struct MemoryContextImpl {
    /// The underlying session context.
    pub session: SessionContextImpl,
    /// Short-term active window messages.
    pub window: Vec<Message>,
}

impl ReadonlyContext for MemoryContextImpl {
    fn invocation_id(&self) -> &str {
        self.session.invocation_id()
    }

    fn session_id(&self) -> &str {
        self.session.session_id()
    }

    fn user_id(&self) -> &str {
        self.session.user_id()
    }

    fn app_name(&self) -> &str {
        self.session.app_name()
    }
}

impl SessionContext for MemoryContextImpl {
    fn session_state(&self) -> &SessionState {
        self.session.session_state()
    }

    fn session_state_mut(&mut self) -> &mut SessionState {
        self.session.session_state_mut()
    }
}

impl MemoryContext for MemoryContextImpl {
    fn active_window(&self) -> &[Message] {
        &self.window
    }

    fn demote(&self, messages: Vec<Message>) -> Result<(), String> {
        let count = messages.len();
        if count == 0 {
            return Ok(());
        }
        // Default: demotion stores messages as JSON in session state
        let mut state = self.session_state().clone();
        let key = format!("memory:demoted:{}", chrono::Utc::now().timestamp());
        let value = serde_json::to_value(&messages).map_err(|e| e.to_string())?;
        state.set(key, value);
        Ok(())
    }

    fn compact(&self, _messages: Vec<Message>) -> Result<String, String> {
        // Default: no-op compaction. Implementations should override this
        // with LLM-based summarization.
        Ok(String::new())
    }
}

/// Hook context implementation for observing run state.
pub struct HookContextImpl {
    /// The application context for identity.
    pub app: AppContext,
    /// The current run state.
    pub state: RunState,
    /// The run identifier.
    pub run_id: RunId,
    /// Current iteration count.
    pub iteration: usize,
    /// Current token usage.
    pub tokens_used: usize,
}

impl ReadonlyContext for HookContextImpl {
    fn invocation_id(&self) -> &str {
        &self.app.invocation_id
    }

    fn session_id(&self) -> &str {
        &self.app.session_id
    }

    fn user_id(&self) -> &str {
        &self.app.user_id
    }

    fn app_name(&self) -> &str {
        &self.app.app_name
    }
}

impl HookContext for HookContextImpl {
    fn current_state(&self) -> &RunState {
        &self.state
    }

    fn snapshot(&self) -> RunSnapshot {
        RunSnapshot {
            state: self.state.clone(),
            run_id: self.run_id,
            session_id: self.session_id().to_string(),
            iteration: self.iteration,
            tokens_used: self.tokens_used,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn session_state_set_and_get() {
        let mut state = SessionState::new();
        state.set("user:name", Value::String("Alice".to_string()));
        assert_eq!(
            state.get("user:name"),
            Some(&Value::String("Alice".to_string()))
        );
        assert!(state.get("nonexistent").is_none());
    }

    #[test]
    fn session_state_remove() {
        let mut state = SessionState::new();
        state.set("temp:key", Value::Bool(true));
        assert!(state.remove("temp:key").is_some());
        assert!(state.get("temp:key").is_none());
    }

    #[test]
    fn run_budget_tracking() {
        let mut budget = RunBudget::new(Some(1000));
        assert_eq!(budget.remaining(), Some(1000));
        budget.consume(300);
        assert_eq!(budget.remaining(), Some(700));
        assert_eq!(budget.used(), 300);
    }

    #[test]
    fn run_budget_unlimited() {
        let budget = RunBudget::new(None);
        assert_eq!(budget.remaining(), None);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn event_sink_emit_and_subscribe() {
        let sink = EventSink::new();
        let mut rx = sink.subscribe();
        sink.emit(serde_json::json!({"type": "test"}));
        // The receiver should have the latest value
        let val = rx.borrow_and_update().clone();
        assert!(val.is_some());
    }

    #[test]
    fn tool_context_emits_progress() {
        let app = AppContext {
            invocation_id: "inv-1".to_string(),
            session_id: "sess-1".to_string(),
            user_id: "user-1".to_string(),
            app_name: "test".to_string(),
        };
        let session = SessionContextImpl {
            app,
            state: SessionState::new(),
        };
        let sink = EventSink::new();
        let mut sub = sink.subscribe();
        let run = RunContextImpl {
            session,
            run_id: RunId::new(),
            cancel: CancellationToken::new(),
            deadline: None,
            sink,
            budget: RunBudget::new(None),
        };
        let ctx = ToolContextImpl {
            run,
            tool_call: ToolCall::new("call_1", "test_tool", Value::Null),
        };

        ctx.emit_progress("working", serde_json::json!({"percent": 50}));

        let emitted = sub.borrow_and_update().clone();
        assert!(emitted.is_some());
        let event = emitted.unwrap();
        assert_eq!(event["type"], "tool_progress");
        assert_eq!(event["call_id"], "call_1");
    }
}
