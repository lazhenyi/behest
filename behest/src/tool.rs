//! Re-export of the tool system from [`behest_tool`].
//! This module is preserved for facade compatibility.

pub use behest_tool::{
    ExecutionPlan, FunctionTool, SideEffects, Tool, ToolExecutionStrategy, ToolOutput,
    ToolRegistry, ToolResult,
};
