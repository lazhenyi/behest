//! Server/Client tool dispatch — explicit split for agent loops.
//!
//! When a model returns tool calls, the runtime splits them into `Server`
//! tools (executed by the runtime, results injected back into the
//! conversation) and `Client` tools (surfaced to the caller, the loop pauses).
//!
//! All logic here is pure: the user feeds in a list of tool calls and a
//! source map, and gets back a [`ToolDispatchPlan`] describing what to do.
//! The user then drives their own loop using the plan and resumes the
//! runtime with the client-side results.
//!
//! # Example
//!
//! ```ignore
//! use behest_runtime::tool_dispatch::{split_tool_calls, ToolDispatchPlan};
//! use behest_core::tool_types::ToolSource;
//! use std::collections::HashMap;
//!
//! // source_map built from registered tools: name → ToolSource
//! let mut source_map = HashMap::new();
//! source_map.insert("read_file".into(), ToolSource::Server);
//! source_map.insert("ask_user".into(), ToolSource::Client);
//!
//! let plan = split_tool_calls(&model_tool_calls, &source_map);
//!
//! // Server tools: execute and continue the loop
//! for call in &plan.server_calls {
//!     let result = tool_runtime.execute(ctx, call).await?;
//!     // inject result into conversation
//! }
//!
//! // Client tools: surface to caller, loop pauses
//! if !plan.client_calls.is_empty() {
//!     return TurnWithToolsOutcome::ClientToolsPending {
//!         calls: plan.client_calls.clone(),
//!     };
//! }
//! ```

use std::collections::HashMap;

use behest_core::tool_types::{ToolCall, ToolSource};

/// A plan describing how to dispatch a batch of tool calls.
///
/// Built by [`split_tool_calls`] from a model-returned list of tool calls
/// and a source map (tool name → where it executes).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ToolDispatchPlan {
    /// Server-side calls: the runtime executes these and consumes the
    /// results internally. The loop continues after they complete.
    pub server_calls: Vec<ToolCall>,
    /// Client-side calls: surfaced to the caller. The loop pauses until
    /// the caller resumes it with results via `submit_tool_results`.
    pub client_calls: Vec<ToolCall>,
}

impl ToolDispatchPlan {
    /// Returns `true` when there are no calls of either kind.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.server_calls.is_empty() && self.client_calls.is_empty()
    }

    /// Returns the total number of calls (server + client).
    #[must_use]
    pub fn total_calls(&self) -> usize {
        self.server_calls.len() + self.client_calls.len()
    }

    /// Returns `true` when at least one client call is present — the
    /// caller must execute these before the loop can continue.
    #[must_use]
    pub fn has_client_calls(&self) -> bool {
        !self.client_calls.is_empty()
    }

    /// Returns `true` when at least one server call is present — the
    /// runtime must execute these before continuing.
    #[must_use]
    pub fn has_server_calls(&self) -> bool {
        !self.server_calls.is_empty()
    }
}

/// Splits a list of model-returned tool calls into server and client groups
/// based on a source map.
///
/// Tools not present in the source map default to `Server` (the runtime
/// handles them). This matches the convention that registered runtime tools
/// are server-side unless explicitly tagged `Client`.
///
/// Order is preserved within each group — server calls appear in the same
/// relative order as in the input, and so do client calls.
#[must_use]
pub fn split_tool_calls(
    calls: &[ToolCall],
    source_map: &HashMap<String, ToolSource>,
) -> ToolDispatchPlan {
    let mut server_calls = Vec::new();
    let mut client_calls = Vec::new();

    for call in calls {
        let source = source_map
            .get(&call.name)
            .copied()
            .unwrap_or(ToolSource::Server);
        if source.is_client() {
            client_calls.push(call.clone());
        } else {
            server_calls.push(call.clone());
        }
    }

    ToolDispatchPlan {
        server_calls,
        client_calls,
    }
}

/// Builds a source map from a list of `(name, source)` pairs.
///
/// Convenience for callers that don't already have a `HashMap`.
#[must_use]
pub fn build_source_map(pairs: Vec<(String, ToolSource)>) -> HashMap<String, ToolSource> {
    pairs.into_iter().collect()
}

/// Outcome of a turn that produced tool calls, after the split.
///
/// The user inspects this and decides how to drive their loop:
/// - `ServerOnly` → execute server tools, then continue the loop
/// - `ClientOnly` → surface client tools to the caller, pause the loop
/// - `Mixed` → execute server tools, then surface client tools, then pause
/// - `Empty` → no tools at all (the model returned an empty list)
#[derive(Debug, Clone, PartialEq)]
pub enum TurnWithToolsOutcome {
    /// Only server tools — execute them and continue the loop.
    ServerOnly {
        /// Server-side tool calls to execute.
        calls: Vec<ToolCall>,
    },
    /// Only client tools — surface to the caller, pause the loop.
    ClientOnly {
        /// Client-side tool calls the caller must execute.
        calls: Vec<ToolCall>,
    },
    /// Both server and client tools — execute server tools first, then
    /// surface the client tools and pause.
    Mixed {
        /// Server-side calls (execute first).
        server_calls: Vec<ToolCall>,
        /// Client-side calls (surface after server tools complete).
        client_calls: Vec<ToolCall>,
    },
    /// The model returned no tool calls.
    Empty,
}

impl TurnWithToolsOutcome {
    /// Builds the outcome from a dispatch plan.
    #[must_use]
    pub fn from_plan(plan: ToolDispatchPlan) -> Self {
        match (plan.server_calls.is_empty(), plan.client_calls.is_empty()) {
            (false, true) => Self::ServerOnly {
                calls: plan.server_calls,
            },
            (true, false) => Self::ClientOnly {
                calls: plan.client_calls,
            },
            (false, false) => Self::Mixed {
                server_calls: plan.server_calls,
                client_calls: plan.client_calls,
            },
            (true, true) => Self::Empty,
        }
    }

    /// Returns `true` if the loop should pause and wait for the caller
    /// (i.e., there are client tools to surface).
    #[must_use]
    pub fn requires_caller(&self) -> bool {
        match self {
            Self::ClientOnly { .. } | Self::Mixed { .. } => true,
            Self::ServerOnly { .. } | Self::Empty => false,
        }
    }

    /// Returns the client-side calls to surface, if any.
    #[must_use]
    pub fn client_calls(&self) -> &[ToolCall] {
        match self {
            Self::ClientOnly { calls } => calls,
            Self::Mixed { client_calls, .. } => client_calls,
            Self::ServerOnly { .. } | Self::Empty => &[],
        }
    }

    /// Returns the server-side calls to execute, if any.
    #[must_use]
    pub fn server_calls(&self) -> &[ToolCall] {
        match self {
            Self::ServerOnly { calls } => calls,
            Self::Mixed { server_calls, .. } => server_calls,
            Self::ClientOnly { .. } | Self::Empty => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use behest_core::tool_types::ToolCall;
    use serde_json::json;

    fn call(name: &str) -> ToolCall {
        ToolCall::new(format!("id_{name}"), name, json!({}))
    }

    fn source_map(pairs: &[(&str, ToolSource)]) -> HashMap<String, ToolSource> {
        pairs.iter().map(|(n, s)| (n.to_string(), *s)).collect()
    }

    #[test]
    fn split_all_server_when_no_map() {
        let calls = vec![call("a"), call("b"), call("c")];
        let plan = split_tool_calls(&calls, &HashMap::new());
        assert_eq!(plan.server_calls.len(), 3);
        assert!(plan.client_calls.is_empty());
    }

    #[test]
    fn split_separates_client_and_server() {
        let calls = vec![call("a"), call("b"), call("c")];
        let map = source_map(&[("b", ToolSource::Client)]);
        let plan = split_tool_calls(&calls, &map);
        assert_eq!(plan.server_calls.len(), 2);
        assert_eq!(plan.client_calls.len(), 1);
        assert_eq!(plan.client_calls[0].name, "b");
    }

    #[test]
    fn split_preserves_order_within_groups() {
        let calls = vec![call("s1"), call("c1"), call("s2"), call("c2")];
        let map = source_map(&[("c1", ToolSource::Client), ("c2", ToolSource::Client)]);
        let plan = split_tool_calls(&calls, &map);
        assert_eq!(plan.server_calls.len(), 2);
        assert_eq!(plan.server_calls[0].name, "s1");
        assert_eq!(plan.server_calls[1].name, "s2");
        assert_eq!(plan.client_calls.len(), 2);
        assert_eq!(plan.client_calls[0].name, "c1");
        assert_eq!(plan.client_calls[1].name, "c2");
    }

    #[test]
    fn split_empty_input() {
        let plan = split_tool_calls(&[], &HashMap::new());
        assert!(plan.is_empty());
        assert_eq!(plan.total_calls(), 0);
    }

    #[test]
    fn from_plan_server_only() {
        let plan = ToolDispatchPlan {
            server_calls: vec![call("a")],
            client_calls: Vec::new(),
        };
        let outcome = TurnWithToolsOutcome::from_plan(plan);
        assert!(matches!(outcome, TurnWithToolsOutcome::ServerOnly { .. }));
        assert!(!outcome.requires_caller());
        assert_eq!(outcome.server_calls().len(), 1);
        assert!(outcome.client_calls().is_empty());
    }

    #[test]
    fn from_plan_client_only() {
        let plan = ToolDispatchPlan {
            server_calls: Vec::new(),
            client_calls: vec![call("a")],
        };
        let outcome = TurnWithToolsOutcome::from_plan(plan);
        assert!(matches!(outcome, TurnWithToolsOutcome::ClientOnly { .. }));
        assert!(outcome.requires_caller());
        assert_eq!(outcome.client_calls().len(), 1);
        assert!(outcome.server_calls().is_empty());
    }

    #[test]
    fn from_plan_mixed() {
        let plan = ToolDispatchPlan {
            server_calls: vec![call("s")],
            client_calls: vec![call("c")],
        };
        let outcome = TurnWithToolsOutcome::from_plan(plan);
        assert!(matches!(outcome, TurnWithToolsOutcome::Mixed { .. }));
        assert!(outcome.requires_caller());
        assert_eq!(outcome.server_calls().len(), 1);
        assert_eq!(outcome.client_calls().len(), 1);
    }

    #[test]
    fn from_plan_empty() {
        let plan = ToolDispatchPlan::default();
        let outcome = TurnWithToolsOutcome::from_plan(plan);
        assert!(matches!(outcome, TurnWithToolsOutcome::Empty));
        assert!(!outcome.requires_caller());
    }

    #[test]
    fn build_source_map_collects_pairs() {
        let map = build_source_map(vec![
            ("a".to_string(), ToolSource::Server),
            ("b".to_string(), ToolSource::Client),
        ]);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("a"), Some(&ToolSource::Server));
        assert_eq!(map.get("b"), Some(&ToolSource::Client));
    }

    #[test]
    fn unknown_tool_defaults_to_server() {
        let calls = vec![call("mystery")];
        let plan = split_tool_calls(&calls, &HashMap::new());
        assert_eq!(plan.server_calls.len(), 1);
        assert!(plan.client_calls.is_empty());
    }
}
