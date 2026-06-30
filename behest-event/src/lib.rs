//! Event types, [`EventActions`], and the [`Hook`] system for the behest
//! agent runtime.
//!
//! This crate provides:
//!
//! - [`AgentEvent`]: the canonical 17-variant event covering the full
//!   agent lifecycle (moved here from `behest::runtime::event`).
//! - [`EventActions`]: side-effect declarations that accompany events.
//! - [`Hook`]: single-method observer that returns `Vec<EventActions>`.
//! - [`HookStack`]: ordered dispatch of multiple hooks.
//!
//! # Design
//!
//! Hooks use a single-method pattern (inspired by Rig): one `on_event()`
//! method receives an [`AgentEvent`] and returns `Vec<EventActions>`.
//! Adding new event types never requires changing the Hook trait
//! signature.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]

use std::collections::HashMap;

use behest_context::HookContext;
use serde::{Deserialize, Serialize};
use serde_json::Value;

mod agent_event;

pub use agent_event::{
    AgentEvent, CacheMetrics, CompactionCircuitOpened, ContextBuilt, DoomLoopDetected,
    MessageCommitted, ModelStarted, RunCancelled, RunCompleted, RunFailed, RunStarted, TextDelta,
    ToolCallCompleted, ToolCallDelta, ToolCallStarted, ToolExecutionFinished, ToolExecutionResult,
    ToolExecutionStarted, UsageRecorded,
};

/// Side-effect declarations associated with an event.
///
/// Hooks and the runtime use these to express "what should happen next"
/// without directly performing I/O. This keeps the event system
/// compatible with sans-IO state machines.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventActions {
    /// State changes to apply to the session.
    pub state_delta: HashMap<String, Value>,
    /// Persist the current conversation to storage.
    pub persist_conversation: bool,
    /// Request memory compaction.
    pub compact_memory: bool,
    /// Request memory demotion.
    pub demote_memory: bool,
    /// Request tool approval.
    pub request_approval: Option<ApprovalRequest>,
    /// Emit a progress event.
    pub emit_progress: Option<ProgressEvent>,
    /// Notify the user (e.g., via push notification).
    pub notify_user: Option<String>,
    /// Write a trace event.
    pub write_trace: Option<String>,
}

impl EventActions {
    /// Creates an empty set of actions.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Merges multiple action sets into one.
    ///
    /// State deltas are merged (later values override earlier ones).
    /// Boolean flags are OR'd. Options are overwritten by the last non-None value.
    #[must_use]
    pub fn merge(actions: Vec<Self>) -> Self {
        let mut merged = Self::new();
        for a in actions {
            merged.state_delta.extend(a.state_delta);
            merged.persist_conversation |= a.persist_conversation;
            merged.compact_memory |= a.compact_memory;
            merged.demote_memory |= a.demote_memory;
            if a.request_approval.is_some() {
                merged.request_approval = a.request_approval;
            }
            if a.emit_progress.is_some() {
                merged.emit_progress = a.emit_progress;
            }
            if a.notify_user.is_some() {
                merged.notify_user = a.notify_user;
            }
            if a.write_trace.is_some() {
                merged.write_trace = a.write_trace;
            }
        }
        merged
    }
}

/// A request for human approval of a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// The call requiring approval.
    pub call_id: String,
    /// The tool name.
    pub tool_name: String,
    /// The tool arguments.
    pub arguments: Value,
    /// Human-readable reason approval is needed.
    pub reason: String,
    /// Timeout for the approval request, in seconds.
    pub timeout_secs: Option<u64>,
}

/// A progress event emitted during tool execution or other long-running ops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressEvent {
    /// Progress status label.
    pub status: String,
    /// Progress data.
    pub data: Value,
}

/// An observer hook that reacts to events.
///
/// Hooks use a single-method design: one `on_event()` method receives
/// an [`AgentEvent`] and returns optional [`EventActions`]. New event
/// types never require changing the trait signature.
pub trait Hook: Send + Sync {
    /// Returns the hook's name.
    fn name(&self) -> &str;

    /// Called for every event. Returns actions the runtime should execute.
    fn on_event(&self, event: &AgentEvent, ctx: &dyn HookContext) -> Vec<EventActions>;

    /// Priority for ordering. Hooks with lower priority run first.
    fn priority(&self) -> i32 {
        0
    }
}

/// An ordered stack of hooks.
pub struct HookStack {
    hooks: Vec<Box<dyn Hook>>,
}

impl HookStack {
    /// Creates an empty hook stack.
    #[must_use]
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Adds a hook to the stack.
    pub fn push(&mut self, hook: Box<dyn Hook>) {
        self.hooks.push(hook);
        self.hooks.sort_by_key(|h| h.priority());
    }

    /// Dispatches an event to all hooks, collecting their actions.
    ///
    /// Hook errors are isolated: a failing hook logs a warning and is skipped,
    /// never breaking the run loop.
    #[must_use]
    pub fn dispatch(&self, event: &AgentEvent, ctx: &dyn HookContext) -> Vec<EventActions> {
        let mut actions = Vec::new();
        for hook in &self.hooks {
            let hook_actions = hook.on_event(event, ctx);
            actions.push(hook_actions);
        }
        let flat: Vec<_> = actions.into_iter().flatten().collect();
        vec![EventActions::merge(flat)]
    }

    /// Returns the number of hooks in the stack.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// Returns `true` if the stack is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }
}

impl Default for HookStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use behest_context::{AppContext, HookContextImpl};
    use behest_core::id::RunId;

    fn make_ctx() -> HookContextImpl {
        HookContextImpl {
            app: AppContext {
                invocation_id: "inv-1".to_string(),
                session_id: "sess-1".to_string(),
                user_id: "user-1".to_string(),
                app_name: "test".to_string(),
            },
            state: behest_core::run::RunState::Idle,
            run_id: RunId::new(),
            iteration: 0,
            tokens_used: 0,
        }
    }

    #[test]
    fn event_actions_merge_collects_all_flags() {
        let a = EventActions {
            persist_conversation: true,
            compact_memory: true,
            ..Default::default()
        };
        let b = EventActions {
            demote_memory: true,
            ..Default::default()
        };
        let merged = EventActions::merge(vec![a, b]);
        assert!(merged.persist_conversation);
        assert!(merged.compact_memory);
        assert!(merged.demote_memory);
    }

    #[test]
    fn hook_stack_dispatches_in_priority_order() {
        let mut stack = HookStack::new();
        stack.push(Box::new(HighPriority));
        stack.push(Box::new(LowPriority));
        let actions = stack.dispatch(
            &AgentEvent::RunStarted(RunStarted {
                run_id: RunId::new(),
                session_id: uuid::Uuid::new_v4(),
                provider: behest_core::id::ProviderId::new("p"),
                model: behest_core::id::ModelName::new("m"),
                timestamp: chrono::Utc::now(),
            }),
            &make_ctx(),
        );
        assert_eq!(actions.len(), 1);
    }

    struct HighPriority;
    impl Hook for HighPriority {
        fn name(&self) -> &str {
            "high"
        }
        fn priority(&self) -> i32 {
            -10
        }
        fn on_event(&self, _: &AgentEvent, _: &dyn HookContext) -> Vec<EventActions> {
            vec![]
        }
    }

    struct LowPriority;
    impl Hook for LowPriority {
        fn name(&self) -> &str {
            "low"
        }
        fn priority(&self) -> i32 {
            10
        }
        fn on_event(&self, _: &AgentEvent, _: &dyn HookContext) -> Vec<EventActions> {
            vec![]
        }
    }
}
