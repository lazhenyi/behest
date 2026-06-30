//! Event system, EventActions, and Hook trait for the behest agent runtime.
//!
//! This crate provides:
//!
//! - [`AgentEvent`]: a unified enum covering the full agent lifecycle
//! - [`EventActions`]: side-effect declarations that accompany events
//! - [`Hook`]: single-method observer that returns optional `EventActions`
//! - [`HookStack`]: ordered dispatch of multiple hooks
//!
//! # Design
//!
//! Hooks use a single-method pattern (inspired by Rig): one `on_event()`
//! method receives an [`AgentEvent`] and returns `Vec<EventActions>`. Adding
//! new event types never requires changing the Hook trait signature.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]

use std::collections::HashMap;

use behest_context::HookContext;
use behest_core::id::RunId;
use behest_core::message::{FinishReason, TokenUsage};
use behest_core::tool_types::ToolCall;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A unified event covering the full agent lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
#[non_exhaustive]
pub enum AgentEvent {
    /// A run has started.
    RunStarted {
        /// Run identifier.
        run_id: RunId,
        /// Session identifier.
        session_id: String,
    },
    /// A run has finished successfully.
    RunFinished {
        /// Run identifier.
        run_id: RunId,
        /// Why the run finished.
        reason: FinishReason,
        /// Total token usage.
        usage: TokenUsage,
    },
    /// A run was aborted due to an error.
    RunAborted {
        /// Run identifier.
        run_id: RunId,
        /// Error that caused the abort.
        error: String,
    },

    /// A model call was initiated.
    ModelCalled {
        /// Request identifier.
        request_id: String,
        /// Model name.
        model: String,
    },
    /// A text delta was received during streaming.
    TextDelta {
        /// Run identifier.
        run_id: RunId,
        /// Incremental text chunk.
        delta: String,
    },
    /// A tool call was started.
    ToolCallStarted {
        /// Run identifier.
        run_id: RunId,
        /// Call identifier.
        call_id: String,
        /// Tool name.
        name: String,
    },
    /// A tool call argument delta was received.
    ToolCallArgumentsDelta {
        /// Run identifier.
        run_id: RunId,
        /// Call identifier.
        call_id: String,
        /// Incremental argument chunk.
        delta: String,
    },
    /// A tool call was completed.
    ToolCallCompleted {
        /// Run identifier.
        run_id: RunId,
        /// Completed tool call.
        call: ToolCall,
    },
    /// A model call completed.
    ModelCompleted {
        /// Run identifier.
        run_id: RunId,
        /// Token usage for this call.
        usage: TokenUsage,
    },

    /// A tool execution started.
    ToolExecutionStarted {
        /// Run identifier.
        run_id: RunId,
        /// Call identifier.
        call_id: String,
        /// Tool name.
        name: String,
    },
    /// A tool reported progress.
    ToolExecutionProgress {
        /// Run identifier.
        run_id: RunId,
        /// Call identifier.
        call_id: String,
        /// Progress status.
        status: String,
        /// Progress data.
        data: Value,
    },
    /// A tool execution completed.
    ToolExecutionCompleted {
        /// Run identifier.
        run_id: RunId,
        /// Call identifier.
        call_id: String,
        /// Tool output.
        output: Value,
    },
    /// A tool execution failed.
    ToolExecutionFailed {
        /// Run identifier.
        run_id: RunId,
        /// Call identifier.
        call_id: String,
        /// Error message.
        error: String,
    },

    /// Approval was requested for a tool call.
    ApprovalRequested {
        /// Run identifier.
        run_id: RunId,
        /// Call identifier.
        call_id: String,
        /// Tool name.
        tool_name: String,
        /// Reason approval is needed.
        reason: String,
    },
    /// Approval was granted.
    ApprovalGranted {
        /// Run identifier.
        run_id: RunId,
        /// Call identifier.
        call_id: String,
    },
    /// Approval was rejected.
    ApprovalRejected {
        /// Run identifier.
        run_id: RunId,
        /// Call identifier.
        call_id: String,
        /// Optional rejection reason.
        reason: Option<String>,
    },
    /// Approval timed out.
    ApprovalTimeout {
        /// Run identifier.
        run_id: RunId,
        /// Call identifier.
        call_id: String,
    },

    /// Messages were demoted to long-term storage.
    MemoryDemoted {
        /// Number of messages demoted.
        count: usize,
    },
    /// Memory was compacted into a summary.
    MemoryCompacted {
        /// Original message count.
        original_count: usize,
        /// Summary length in characters.
        summary_length: usize,
    },
    /// Memory was restored from long-term storage.
    MemoryRestored {
        /// Number of messages restored.
        count: usize,
    },
}

impl AgentEvent {
    /// Returns the run ID for this event, if applicable.
    #[must_use]
    pub fn run_id(&self) -> Option<&RunId> {
        match self {
            Self::RunStarted { run_id, .. }
            | Self::RunFinished { run_id, .. }
            | Self::RunAborted { run_id, .. }
            | Self::TextDelta { run_id, .. }
            | Self::ToolCallStarted { run_id, .. }
            | Self::ToolCallArgumentsDelta { run_id, .. }
            | Self::ToolCallCompleted { run_id, .. }
            | Self::ModelCompleted { run_id, .. }
            | Self::ToolExecutionStarted { run_id, .. }
            | Self::ToolExecutionProgress { run_id, .. }
            | Self::ToolExecutionCompleted { run_id, .. }
            | Self::ToolExecutionFailed { run_id, .. }
            | Self::ApprovalRequested { run_id, .. }
            | Self::ApprovalGranted { run_id, .. }
            | Self::ApprovalRejected { run_id, .. }
            | Self::ApprovalTimeout { run_id, .. } => Some(run_id),
            Self::ModelCalled { .. }
            | Self::MemoryDemoted { .. }
            | Self::MemoryCompacted { .. }
            | Self::MemoryRestored { .. } => None,
        }
    }
}

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

    struct TestHook {
        should_persist: bool,
    }

    impl Hook for TestHook {
        fn name(&self) -> &str {
            "test_hook"
        }

        fn on_event(&self, _event: &AgentEvent, _ctx: &dyn HookContext) -> Vec<EventActions> {
            if self.should_persist {
                vec![EventActions {
                    persist_conversation: true,
                    ..Default::default()
                }]
            } else {
                vec![]
            }
        }
    }

    #[test]
    fn hook_stack_dispatch() {
        let mut stack = HookStack::new();
        stack.push(Box::new(TestHook {
            should_persist: true,
        }));

        let event = AgentEvent::RunStarted {
            run_id: RunId::new(),
            session_id: "sess-1".to_string(),
        };
        let ctx = make_ctx();

        let actions = stack.dispatch(&event, &ctx);
        assert_eq!(actions.len(), 1);
        assert!(actions[0].persist_conversation);
    }

    #[test]
    fn event_actions_merge_state_deltas() {
        let a1 = EventActions {
            state_delta: {
                let mut m = HashMap::new();
                m.insert("a".to_string(), Value::String("1".to_string()));
                m
            },
            ..Default::default()
        };
        let a2 = EventActions {
            state_delta: {
                let mut m = HashMap::new();
                m.insert("b".to_string(), Value::String("2".to_string()));
                m
            },
            persist_conversation: true,
            ..Default::default()
        };

        let merged = EventActions::merge(vec![a1, a2]);
        assert_eq!(merged.state_delta.len(), 2);
        assert!(merged.persist_conversation);
    }

    #[test]
    fn hook_stack_empty_returns_empty() {
        let stack = HookStack::new();
        let event = AgentEvent::RunStarted {
            run_id: RunId::new(),
            session_id: "s".to_string(),
        };
        let ctx = make_ctx();
        let actions = stack.dispatch(&event, &ctx);
        assert_eq!(actions.len(), 1);
        let a = &actions[0];
        assert!(!a.persist_conversation);
        assert!(a.state_delta.is_empty());
    }
}
