//! Tool trait, registry, and execution strategies for the behest agent runtime.
//!
//! # Key types
//!
//! - [`Tool`] trait: the fundamental abstraction for callable capabilities
//! - [`ToolRegistry`]: thread-safe dynamic registration and dispatch
//! - [`ToolExecutionStrategy`]: controls serial vs parallel execution
//! - [`FunctionTool`]: closure-based tool with metadata
//! - [`ToolOutput`]: wrapped JSON output from tool execution
//!
//! # ToolExecutionStrategy
//!
//! - **Sequential**: strict one-at-a-time execution
//! - **Parallel**: concurrent execution with a max concurrency cap
//! - **Auto**: intelligently groups read-only/concurrency-safe tools for
//!   parallel execution while serializing stateful tools

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]

use std::sync::{Arc, RwLockReadGuard, RwLockWriteGuard};

use async_trait::async_trait;
use behest_context::ToolContext;
use behest_core::tool_types::{ToolCall, ToolSpec};
use serde::{Deserialize, Serialize};
use serde_json::Value;

mod strategy;
pub use strategy::{ExecutionPlan, ToolExecutionStrategy};

/// Canonical tool error type. Aliased here so that downstream callers
/// can write `behest_tool::ToolError` without taking a direct
/// dependency on `behest_core`.
pub use behest_core::error::ToolError;

/// Alias for a tool execution result.
pub type ToolResult<T = ToolOutput> = std::result::Result<T, ToolError>;

/// The output of a tool execution, wrapping a JSON value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// The tool's output as a JSON value.
    pub value: Value,
}

impl ToolOutput {
    /// Creates a tool output from a JSON value.
    #[must_use]
    pub fn new(value: Value) -> Self {
        Self { value }
    }

    /// Creates a tool output from a text string.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            value: Value::String(text.into()),
        }
    }

    /// Creates a tool output representing an error.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            value: serde_json::json!({"error": message.into()}),
        }
    }
}

/// Side effects a tool may produce.
///
/// Used by [`ToolExecutionStrategy::Auto`] to determine safe concurrency.
#[derive(Debug, Clone, Copy, Default)]
pub struct SideEffects {
    /// Tool performs read operations (safe to run concurrently).
    pub read: bool,
    /// Tool performs write operations (requires serialization).
    pub write: bool,
    /// Tool makes external API calls.
    pub external: bool,
    /// Tool may delete or destroy resources.
    pub destructive: bool,
}

impl SideEffects {
    /// Returns `true` if the tool is safe to run concurrently with other tools.
    #[must_use]
    pub const fn is_concurrency_safe(&self) -> bool {
        !self.write && !self.destructive
    }
}

/// An executable tool that can be called by an LLM.
///
/// Implementations must provide at minimum: `name()`, `description()`,
/// `parameters_schema()`, and `execute()`.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the tool's name (visible to the model).
    fn name(&self) -> &str;

    /// Returns a human-readable description of the tool.
    fn description(&self) -> &str;

    /// Returns the JSON Schema for the tool's parameters.
    fn parameters_schema(&self) -> Value;

    /// Returns the JSON Schema for the tool's response (if any).
    fn response_schema(&self) -> Option<Value> {
        None
    }

    /// Executes the tool with the given arguments.
    async fn execute(&self, args: Value) -> ToolResult<ToolOutput>;

    /// Executes the tool with the given [`ToolContext`] and arguments.
    ///
    /// Defaults to calling [`Self::execute`] without context. Override
    /// to consume the context (e.g., for memory / cancellation / sink).
    async fn execute_with_ctx(
        &self,
        _ctx: &dyn ToolContext,
        args: Value,
    ) -> ToolResult<ToolOutput> {
        self.execute(args).await
    }

    /// Returns `true` if the tool is read-only (no side effects).
    fn is_read_only(&self) -> bool {
        false
    }

    /// Returns `true` if the tool is safe to run concurrently.
    fn is_concurrency_safe(&self) -> bool {
        false
    }

    /// Returns `true` if the tool is expected to run for a long time.
    fn is_long_running(&self) -> bool {
        false
    }

    /// Returns the side effects the tool may produce.
    fn side_effects(&self) -> SideEffects {
        SideEffects::default()
    }

    /// Returns `true` if the tool requires human approval before execution.
    fn requires_approval(&self) -> bool {
        false
    }

    /// Returns the reason approval is needed, if applicable.
    fn approval_reason(&self) -> Option<String> {
        None
    }

    /// Returns the scopes required to use this tool.
    fn required_scopes(&self) -> &[&str] {
        &[]
    }

    /// Produces a [`ToolSpec`] for this tool.
    #[must_use]
    fn to_spec(&self) -> ToolSpec {
        ToolSpec::new(self.name(), self.description(), self.parameters_schema())
    }
}

/// Type alias for the async handler function used by [`FunctionTool`].
///
/// The handler returns a JSON [`Value`] (not a [`ToolOutput`]); the
/// wrapper internally converts it to a [`ToolOutput`]. This matches
/// the long-standing convention from the facade's `FunctionTool::new`.
pub type ToolHandler = dyn Fn(Value) -> std::pin::Pin<Box<dyn std::future::Future<Output = ToolResult<Value>> + Send>>
    + Send
    + Sync;

/// A closure-based tool.
pub struct FunctionTool {
    name: String,
    description: String,
    parameters_schema: Value,
    read_only: bool,
    concurrency_safe: bool,
    long_running: bool,
    side_effects: SideEffects,
    approval_required: bool,
    approval_reason: Option<String>,
    handler: Box<ToolHandler>,
}

impl std::fmt::Debug for FunctionTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionTool")
            .field("name", &self.name)
            .field("read_only", &self.read_only)
            .finish()
    }
}

impl FunctionTool {
    /// Creates a new function-based tool.
    ///
    /// The handler is expected to return a JSON [`Value`]; the
    /// implementation wraps it in a [`ToolOutput`].
    #[must_use]
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters_schema: Value,
        handler: F,
    ) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ToolResult<Value>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            parameters_schema,
            read_only: false,
            concurrency_safe: false,
            long_running: false,
            side_effects: SideEffects::default(),
            approval_required: false,
            approval_reason: None,
            handler: Box::new(move |args| Box::pin(handler(args))),
        }
    }

    /// Marks the tool as read-only.
    #[must_use]
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self.concurrency_safe = true;
        self.side_effects.read = true;
        self
    }

    /// Marks the tool as safe for concurrent execution.
    #[must_use]
    pub fn concurrency_safe(mut self) -> Self {
        self.concurrency_safe = true;
        self
    }

    /// Marks the tool as long-running.
    #[must_use]
    pub fn long_running(mut self) -> Self {
        self.long_running = true;
        self
    }

    /// Sets the side effects for this tool.
    #[must_use]
    pub fn side_effects(mut self, effects: SideEffects) -> Self {
        self.side_effects = effects;
        self
    }

    /// Marks the tool as requiring human approval.
    #[must_use]
    pub fn requires_approval(mut self, reason: impl Into<String>) -> Self {
        self.approval_required = true;
        self.approval_reason = Some(reason.into());
        self
    }
}

#[async_trait]
impl Tool for FunctionTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        self.parameters_schema.clone()
    }

    async fn execute(&self, args: Value) -> ToolResult<ToolOutput> {
        let value = (self.handler)(args).await?;
        Ok(ToolOutput::new(value))
    }

    fn is_read_only(&self) -> bool {
        self.read_only
    }

    fn is_concurrency_safe(&self) -> bool {
        self.concurrency_safe
    }

    fn is_long_running(&self) -> bool {
        self.long_running
    }

    fn side_effects(&self) -> SideEffects {
        self.side_effects
    }

    fn requires_approval(&self) -> bool {
        self.approval_required
    }

    fn approval_reason(&self) -> Option<String> {
        self.approval_reason.clone()
    }
}

/// A thread-safe registry of tools.
pub struct ToolRegistry {
    tools: std::sync::RwLock<std::collections::HashMap<String, Arc<dyn Tool>>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names = self.names();
        f.debug_struct("ToolRegistry")
            .field("tools", &names)
            .finish()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        // `RwLock` doesn't implement `Clone`, but the inner map is
        // cheaply cloneable. We take a read lock to snapshot.
        let map = self.read_tools().clone();
        Self {
            tools: std::sync::RwLock::new(map),
        }
    }
}

impl ToolRegistry {
    /// Creates an empty tool registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Registers a tool, returning any previous tool with the same name.
    pub fn register<T: Tool + 'static>(&self, tool: T) -> Option<Arc<dyn Tool>> {
        self.write_tools()
            .insert(tool.name().to_string(), Arc::new(tool))
    }

    /// Registers an already-shared tool, returning any previous tool
    /// with the same name.
    pub fn register_arc(&self, tool: Arc<dyn Tool>) -> Option<Arc<dyn Tool>> {
        self.write_tools().insert(tool.name().to_string(), tool)
    }

    /// Unregisters a tool by name.
    pub fn unregister(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.write_tools().remove(name)
    }

    /// Gets a tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.read_tools().get(name).cloned()
    }

    /// Returns all registered tool names.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.read_tools().keys().cloned().collect()
    }

    /// Returns the number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.read_tools().len()
    }

    /// Returns `true` if no tools are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.read_tools().is_empty()
    }

    /// Generates [`ToolSpec`]s for all registered tools, sorted by name.
    #[must_use]
    pub fn specs(&self) -> Vec<ToolSpec> {
        let mut specs: Vec<_> = self.read_tools().values().map(|t| t.to_spec()).collect();
        specs.sort_by(|a, b| a.name.cmp(&b.name));
        specs
    }

    /// Looks up and executes a tool call.
    pub async fn execute(&self, ctx: &dyn ToolContext, call: &ToolCall) -> ToolResult<ToolOutput> {
        let tool = self.get(&call.name).ok_or_else(|| ToolError::NotFound {
            name: call.name.clone(),
        })?;
        tool.execute_with_ctx(ctx, call.arguments.clone()).await
    }

    fn read_tools(&self) -> RwLockReadGuard<'_, std::collections::HashMap<String, Arc<dyn Tool>>> {
        match self.tools.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn write_tools(
        &self,
    ) -> RwLockWriteGuard<'_, std::collections::HashMap<String, Arc<dyn Tool>>> {
        match self.tools.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool(name: &str, read_only: bool) -> FunctionTool {
        let n = name.to_string();
        FunctionTool::new(
            name,
            format!("Tool {name}"),
            serde_json::json!({"type": "object", "properties": {}}),
            move |_args| {
                let n = n.clone();
                Box::pin(async move { Ok(serde_json::Value::String(format!("{n} done"))) })
            },
        )
        .let_ro(read_only)
    }

    impl FunctionTool {
        fn let_ro(mut self, read_only: bool) -> Self {
            if read_only {
                self.read_only = true;
                self.concurrency_safe = true;
            }
            self
        }
    }

    #[test]
    fn tool_registry_register_and_get() {
        let reg = ToolRegistry::new();
        reg.register(make_tool("echo", true));
        assert!(reg.get("echo").is_some());
        assert!(reg.get("nonexistent").is_none());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn tool_registry_unregister() {
        let reg = ToolRegistry::new();
        reg.register(make_tool("echo", true));
        assert!(reg.unregister("echo").is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn tool_specs_sorted() {
        let reg = ToolRegistry::new();
        reg.register(make_tool("zulu", true));
        reg.register(make_tool("alpha", true));
        let specs = reg.specs();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "alpha");
        assert_eq!(specs[1].name, "zulu");
    }

    #[test]
    fn tool_registry_recovers_from_poisoned_lock() {
        let reg = Arc::new(ToolRegistry::new());
        reg.register(make_tool("before", true));

        let cloned = Arc::clone(&reg);
        let _ = std::thread::spawn(move || {
            let _guard = cloned.tools.write();
            panic!("poison registry");
        })
        .join();

        reg.register(make_tool("after", true));

        assert_eq!(reg.len(), 2);
        assert!(reg.get("before").is_some());
        assert!(reg.get("after").is_some());
    }

    #[test]
    fn side_effects_concurrency_safety() {
        let read_only = SideEffects {
            read: true,
            ..Default::default()
        };
        assert!(read_only.is_concurrency_safe());

        let write = SideEffects {
            write: true,
            ..Default::default()
        };
        assert!(!write.is_concurrency_safe());

        let destructive = SideEffects {
            destructive: true,
            ..Default::default()
        };
        assert!(!destructive.is_concurrency_safe());
    }
}
