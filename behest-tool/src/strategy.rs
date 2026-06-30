//! Tool execution strategies.
//!
//! Controls how multiple tool calls are executed:
//! - [`Sequential`](ToolExecutionStrategy::Sequential): one at a time
//! - [`Parallel`](ToolExecutionStrategy::Parallel): concurrently with a cap
//! - [`Auto`](ToolExecutionStrategy::Auto): intelligently groups based on tool metadata

use std::sync::Arc;

use behest_core::tool_types::ToolCall;

use crate::Tool;

/// How tool calls should be executed.
#[derive(Debug, Clone, Default)]
pub enum ToolExecutionStrategy {
    /// Execute tool calls one at a time, in order.
    Sequential,
    /// Execute tool calls concurrently, up to the given limit.
    Parallel {
        /// Maximum number of concurrent tool executions.
        max_concurrency: usize,
    },
    /// Automatically decide based on tool metadata:
    /// - Read-only + concurrency-safe tools → parallel
    /// - Write/destructive tools → sequential
    /// - Requires-approval tools → held until approved
    #[default]
    Auto,
}

/// A plan for executing tool calls.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Groups of tool calls. Each group runs sequentially,
    /// but calls within a group may run in parallel if `parallel` is true.
    pub groups: Vec<ExecutionGroup>,
}

/// A group of tool calls to execute together.
#[derive(Debug, Clone)]
pub struct ExecutionGroup {
    /// Whether calls in this group can run in parallel.
    pub parallel: bool,
    /// The tool calls in this group.
    pub calls: Vec<ToolCall>,
}

impl ExecutionPlan {
    /// Creates a sequential plan (all calls in one non-parallel group).
    #[must_use]
    pub fn sequential(calls: Vec<ToolCall>) -> Self {
        Self {
            groups: vec![ExecutionGroup {
                parallel: false,
                calls,
            }],
        }
    }

    /// Creates a parallel plan (all calls in one parallel group).
    #[must_use]
    pub fn parallel(calls: Vec<ToolCall>) -> Self {
        Self {
            groups: vec![ExecutionGroup {
                parallel: true,
                calls,
            }],
        }
    }
}

impl ToolExecutionStrategy {
    /// Produces an execution plan for the given tool calls.
    #[must_use]
    pub fn plan(&self, calls: &[ToolCall], tools: &[Arc<dyn Tool>]) -> ExecutionPlan {
        match self {
            Self::Sequential => ExecutionPlan::sequential(calls.to_vec()),
            Self::Parallel { .. } => ExecutionPlan::parallel(calls.to_vec()),
            Self::Auto => Self::auto_plan(calls, tools),
        }
    }

    fn auto_plan(calls: &[ToolCall], tools: &[Arc<dyn Tool>]) -> ExecutionPlan {
        if calls.len() <= 1 {
            return ExecutionPlan::sequential(calls.to_vec());
        }

        // Build a name → tool lookup
        let tool_map: std::collections::HashMap<&str, &Arc<dyn Tool>> =
            tools.iter().map(|t| (t.name(), t)).collect();

        // Partition into parallel-safe and sequential calls
        let mut parallel_safe: Vec<ToolCall> = Vec::new();
        let mut sequential: Vec<ToolCall> = Vec::new();
        let mut pending_approval: Vec<ToolCall> = Vec::new();

        for call in calls {
            let tool = tool_map.get(call.name.as_str());
            let is_read_only = tool.is_some_and(|t| t.is_read_only());
            let is_concurrency_safe = tool.is_some_and(|t| t.is_concurrency_safe());
            let requires_approval = tool.is_some_and(|t| t.requires_approval());

            if requires_approval {
                pending_approval.push(call.clone());
            } else if is_read_only || is_concurrency_safe {
                parallel_safe.push(call.clone());
            } else {
                sequential.push(call.clone());
            }
        }

        let mut groups: Vec<ExecutionGroup> = Vec::new();

        // Pending approval tools each get their own group (must wait for approval)
        for call in pending_approval {
            groups.push(ExecutionGroup {
                parallel: false,
                calls: vec![call],
            });
        }

        // Parallel-safe tools run together
        if !parallel_safe.is_empty() {
            groups.push(ExecutionGroup {
                parallel: true,
                calls: parallel_safe,
            });
        }

        // Sequential tools each get their own group
        for call in sequential {
            groups.push(ExecutionGroup {
                parallel: false,
                calls: vec![call],
            });
        }

        if groups.is_empty() {
            ExecutionPlan::sequential(calls.to_vec())
        } else {
            ExecutionPlan { groups }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FunctionTool, ToolRegistry};

    fn make_ro_tool(name: &str) -> Arc<dyn Tool> {
        Arc::new(
            FunctionTool::new(name, format!("{name} tool"), serde_json::json!({}), |_| {
                Box::pin(async { Ok(crate::ToolOutput::text("ok")) })
            })
            .read_only(),
        )
    }

    fn make_rw_tool(name: &str) -> Arc<dyn Tool> {
        Arc::new(FunctionTool::new(
            name,
            format!("{name} tool"),
            serde_json::json!({}),
            |_| Box::pin(async { Ok(crate::ToolOutput::text("ok")) }),
        ))
    }

    fn make_call(name: &str) -> ToolCall {
        ToolCall::new(format!("call_{name}"), name, serde_json::Value::Null)
    }

    #[test]
    fn auto_parallelizes_ro_tools() {
        let tools: Vec<Arc<dyn Tool>> = vec![make_ro_tool("search"), make_ro_tool("fetch")];
        let calls = vec![make_call("search"), make_call("fetch")];

        let plan = ToolExecutionStrategy::Auto.plan(&calls, &tools);
        assert_eq!(plan.groups.len(), 1);
        assert!(plan.groups[0].parallel);
        assert_eq!(plan.groups[0].calls.len(), 2);
    }

    #[test]
    fn auto_serializes_rw_tools() {
        let tools: Vec<Arc<dyn Tool>> = vec![make_rw_tool("create"), make_rw_tool("delete")];
        let calls = vec![make_call("create"), make_call("delete")];

        let plan = ToolExecutionStrategy::Auto.plan(&calls, &tools);
        assert_eq!(plan.groups.len(), 2);
        assert!(!plan.groups[0].parallel);
        assert!(!plan.groups[1].parallel);
    }

    #[test]
    fn auto_mixed_partitions() {
        let ro1 = make_ro_tool("search");
        let rw1 = make_rw_tool("save");
        let ro2 = make_ro_tool("fetch");
        let tools: Vec<Arc<dyn Tool>> = vec![ro1, rw1, ro2];
        let calls = vec![make_call("search"), make_call("save"), make_call("fetch")];

        let plan = ToolExecutionStrategy::Auto.plan(&calls, &tools);
        // Parallel group first (search + fetch), then sequential save
        assert_eq!(plan.groups.len(), 2);
        assert!(plan.groups[0].parallel);
        assert_eq!(plan.groups[0].calls.len(), 2);
        assert!(!plan.groups[1].parallel);
        assert_eq!(plan.groups[1].calls.len(), 1);
    }

    #[test]
    fn sequential_strategy_always_one_group() {
        let tools: Vec<Arc<dyn Tool>> = vec![make_ro_tool("a"), make_ro_tool("b")];
        let calls = vec![make_call("a"), make_call("b")];

        let plan = ToolExecutionStrategy::Sequential.plan(&calls, &tools);
        assert_eq!(plan.groups.len(), 1);
        assert!(!plan.groups[0].parallel);
        assert_eq!(plan.groups[0].calls.len(), 2);
    }

    #[test]
    fn parallel_strategy_always_one_group() {
        let tools: Vec<Arc<dyn Tool>> = vec![make_ro_tool("a"), make_ro_tool("b")];
        let calls = vec![make_call("a"), make_call("b")];

        let plan = ToolExecutionStrategy::Parallel { max_concurrency: 4 }.plan(&calls, &tools);
        assert_eq!(plan.groups.len(), 1);
        assert!(plan.groups[0].parallel);
    }

    #[test]
    fn single_call_uses_sequential() {
        let tools: Vec<Arc<dyn Tool>> = vec![make_ro_tool("a")];
        let calls = vec![make_call("a")];

        let plan = ToolExecutionStrategy::Auto.plan(&calls, &tools);
        assert_eq!(plan.groups.len(), 1);
        assert!(!plan.groups[0].parallel);
    }
}
