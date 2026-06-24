//! Runtime tool registry and execution primitives.
//!
//! The [`Tool`] trait defines the contract for executable tools that can be
//! invoked by a chat provider. The [`ToolRegistry`] manages tool registration,
//! schema generation, and execution dispatch.
//!
//! # Example
//!
//! ```rust
//! use behest::tool::{FunctionTool, ToolRegistry};
//! use serde_json::{Value, json};
//!
//! let registry = ToolRegistry::new();
//!
//! let echo_tool = FunctionTool::new(
//!     "echo",
//!     "Echoes the input message",
//!     json!({
//!         "type": "object",
//!         "properties": {
//!             "message": { "type": "string" }
//!         },
//!         "required": ["message"]
//!     }),
//!     |args: Value| async move {
//!         Ok(args.get("message").cloned().unwrap_or(Value::Null))
//!     },
//! );
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(echo_tool);
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ToolError;
use crate::provider::{Message, ToolCall, ToolSpec};

/// Result type for tool execution.
pub type ToolResult<T> = std::result::Result<T, ToolError>;

/// Output produced by a tool execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolOutput {
    /// JSON value returned by the tool.
    pub value: Value,
}

impl ToolOutput {
    /// Creates a tool output from a JSON value.
    #[must_use]
    pub const fn new(value: Value) -> Self {
        Self { value }
    }

    /// Creates a tool output from a string.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            value: Value::String(text.into()),
        }
    }

    /// Creates a tool output indicating an error occurred.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            value: serde_json::json!({ "error": message.into() }),
        }
    }
}

/// Executable tool that can be invoked by a chat provider.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the tool name visible to the model.
    fn name(&self) -> &str;

    /// Returns the tool description visible to the model.
    fn description(&self) -> &str;

    /// Returns the JSON schema describing accepted arguments.
    fn parameters_schema(&self) -> Value;

    /// Executes the tool with the given arguments.
    async fn execute(&self, arguments: Value) -> ToolResult<ToolOutput>;

    /// Returns `true` if the tool does not modify any external state.
    /// Defaults to `false` (fail-closed).
    fn is_read_only(&self) -> bool {
        false
    }

    /// Returns `true` if the tool can be safely executed concurrently with
    /// other concurrent-safe tools. Defaults to `false` (fail-closed).
    fn is_concurrency_safe(&self) -> bool {
        false
    }

    /// Converts this tool into a [`ToolSpec`] for provider requests.
    fn to_spec(&self) -> ToolSpec {
        ToolSpec::new(self.name(), self.description(), self.parameters_schema())
    }
}

/// Function-based tool implementation.
pub struct FunctionTool<F> {
    name: String,
    description: String,
    parameters_schema: Value,
    handler: F,
    read_only: bool,
    concurrency_safe: bool,
}

impl<F, Fut> FunctionTool<F>
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ToolResult<Value>> + Send + 'static,
{
    /// Creates a function-based tool.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters_schema: Value,
        handler: F,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters_schema,
            handler,
            read_only: false,
            concurrency_safe: false,
        }
    }

    /// Marks this tool as read-only (does not modify external state).
    #[must_use]
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Marks this tool as safe for concurrent execution.
    #[must_use]
    pub fn concurrency_safe(mut self) -> Self {
        self.concurrency_safe = true;
        self
    }
}

#[async_trait]
impl<F, Fut> Tool for FunctionTool<F>
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ToolResult<Value>> + Send + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        self.parameters_schema.clone()
    }

    async fn execute(&self, arguments: Value) -> ToolResult<ToolOutput> {
        let value = (self.handler)(arguments).await?;
        Ok(ToolOutput::new(value))
    }

    fn is_read_only(&self) -> bool {
        self.read_only
    }

    fn is_concurrency_safe(&self) -> bool {
        self.concurrency_safe
    }
}

/// External tool that delegates execution to an external system.
///
/// This is a **schema-only placeholder** for future integration with external
/// tool systems such as MCP (Model Context Protocol) or HTTP-based tool servers.
///
/// # Warning
///
/// Calling [`Tool::execute`] on an `ExternalTool` **always** returns
/// [`ToolError::NotImplemented`]. It should only be registered when the caller
/// handles execution externally (e.g., by intercepting tool calls before
/// dispatching to the registry).
pub struct ExternalTool {
    name: String,
    description: String,
    parameters_schema: Value,
    endpoint: Option<String>,
    read_only: bool,
    concurrency_safe: bool,
}

impl ExternalTool {
    /// Creates an external tool definition.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters_schema,
            endpoint: None,
            read_only: false,
            concurrency_safe: false,
        }
    }

    /// Sets the external endpoint for this tool.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Returns the configured endpoint, if any.
    #[must_use]
    pub fn endpoint(&self) -> Option<&str> {
        self.endpoint.as_deref()
    }

    /// Marks this tool as read-only (does not modify external state).
    #[must_use]
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Marks this tool as safe for concurrent execution.
    #[must_use]
    pub fn concurrency_safe(mut self) -> Self {
        self.concurrency_safe = true;
        self
    }
}

#[async_trait]
impl Tool for ExternalTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        self.parameters_schema.clone()
    }

    async fn execute(&self, _arguments: Value) -> ToolResult<ToolOutput> {
        Err(ToolError::NotImplemented {
            name: self.name.clone(),
        })
    }

    fn is_read_only(&self) -> bool {
        self.read_only
    }

    fn is_concurrency_safe(&self) -> bool {
        self.concurrency_safe
    }
}

/// Runtime registry for executable tools.
///
/// The registry maintains tool definitions and provides execution dispatch
/// for tool calls returned by chat providers.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Creates an empty tool registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tool and returns the replaced tool, if any.
    pub fn register<T>(&mut self, tool: T) -> Option<Arc<dyn Tool>>
    where
        T: Tool + 'static,
    {
        let name = tool.name().to_owned();
        self.tools.insert(name, Arc::new(tool))
    }

    /// Registers an already shared tool.
    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) -> Option<Arc<dyn Tool>> {
        let name = tool.name().to_owned();
        self.tools.insert(name, tool)
    }

    /// Returns a registered tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).map(Arc::clone)
    }

    /// Returns all registered tool names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }

    /// Returns the number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns `true` when no tools are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Generates [`ToolSpec`] definitions for all registered tools,
    /// sorted alphabetically by tool name.
    ///
    /// The sorted output ensures deterministic prompt caching across
    /// turns and provider calls.
    #[must_use]
    pub fn specs(&self) -> Vec<ToolSpec> {
        let mut specs: Vec<ToolSpec> = self.tools.values().map(|t| t.to_spec()).collect();
        specs.sort_by(|a, b| a.name.cmp(&b.name));
        specs
    }

    /// Executes a tool call and returns the output.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::NotFound`] when the tool is not registered.
    pub async fn execute(&self, call: &ToolCall) -> ToolResult<ToolOutput> {
        let tool = self.get(&call.name).ok_or_else(|| ToolError::NotFound {
            name: call.name.clone(),
        })?;
        tool.execute(call.arguments.clone()).await
    }

    /// Executes a tool call and converts the result to a [`Message::Tool`].
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::NotFound`] when the tool is not registered.
    pub async fn execute_to_message(&self, call: &ToolCall) -> ToolResult<Message> {
        let output = self.execute(call).await?;
        Ok(Message::tool_text(
            call.id.clone(),
            call.name.clone(),
            output.value.to_string(),
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    async fn echo_handler(args: Value) -> ToolResult<Value> {
        Ok(args.get("message").cloned().unwrap_or(Value::Null))
    }

    #[test]
    fn tool_registry_should_be_empty_when_new() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn tool_registry_should_register_function_tool() {
        let mut registry = ToolRegistry::new();
        let tool = FunctionTool::new(
            "echo",
            "Echoes input",
            json!({"type": "object"}),
            echo_handler,
        );
        registry.register(tool);

        assert_eq!(registry.len(), 1);
        assert!(registry.get("echo").is_some());
    }

    #[test]
    fn tool_registry_should_return_none_for_unknown_tool() {
        let registry = ToolRegistry::new();
        assert!(registry.get("unknown").is_none());
    }

    #[test]
    fn tool_registry_should_generate_specs() {
        let mut registry = ToolRegistry::new();
        let tool = FunctionTool::new(
            "echo",
            "Echoes input",
            json!({"type": "object"}),
            echo_handler,
        );
        registry.register(tool);

        let specs = registry.specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "echo");
        assert_eq!(specs[0].description, "Echoes input");
    }

    #[test]
    fn tool_registry_should_replace_existing_tool() {
        let mut registry = ToolRegistry::new();
        let tool1 = FunctionTool::new("echo", "First", json!({}), echo_handler);
        let tool2 = FunctionTool::new("echo", "Second", json!({}), echo_handler);

        registry.register(tool1);
        let replaced = registry.register(tool2);

        assert!(replaced.is_some());
        assert_eq!(registry.len(), 1);
    }

    #[tokio::test]
    async fn tool_registry_should_execute_tool_call() {
        let mut registry = ToolRegistry::new();
        let tool = FunctionTool::new(
            "echo",
            "Echoes input",
            json!({"type": "object"}),
            echo_handler,
        );
        registry.register(tool);

        let call = ToolCall::new("call_1", "echo", json!({"message": "hello"}));
        let output = registry.execute(&call).await.unwrap();

        assert_eq!(output.value, json!("hello"));
    }

    #[tokio::test]
    async fn tool_registry_should_return_error_for_unknown_tool() {
        let registry = ToolRegistry::new();
        let call = ToolCall::new("call_1", "unknown", json!({}));
        let result = registry.execute(&call).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::NotFound { .. }));
    }

    #[tokio::test]
    async fn tool_registry_should_convert_output_to_message() {
        let mut registry = ToolRegistry::new();
        let tool = FunctionTool::new(
            "echo",
            "Echoes input",
            json!({"type": "object"}),
            echo_handler,
        );
        registry.register(tool);

        let call = ToolCall::new("call_1", "echo", json!({"message": "hello"}));
        let message = registry.execute_to_message(&call).await.unwrap();

        match message {
            Message::Tool {
                tool_call_id,
                name,
                content,
            } => {
                assert_eq!(tool_call_id, "call_1");
                assert_eq!(name, "echo");
                assert!(!content.is_empty());
            }
            _ => panic!("expected Message::Tool"),
        }
    }

    #[test]
    fn external_tool_should_return_not_implemented() {
        let tool = ExternalTool::new("external", "External tool", json!({}));
        assert_eq!(tool.name(), "external");
        assert!(tool.endpoint().is_none());
    }

    #[test]
    fn external_tool_should_accept_endpoint() {
        let tool = ExternalTool::new("external", "External tool", json!({}))
            .with_endpoint("https://example.com/tool");
        assert_eq!(tool.endpoint(), Some("https://example.com/tool"));
    }

    #[tokio::test]
    async fn external_tool_execute_should_return_not_implemented() {
        let tool = ExternalTool::new("external", "External tool", json!({}));
        let result = tool.execute(json!({})).await;
        assert!(matches!(result, Err(ToolError::NotImplemented { .. })));
    }

    #[test]
    fn specs_should_return_sorted_by_name() {
        let mut registry = ToolRegistry::new();
        registry.register(FunctionTool::new(
            "zebra",
            "Zebra tool",
            json!({}),
            echo_handler,
        ));
        registry.register(FunctionTool::new(
            "alpha",
            "Alpha tool",
            json!({}),
            echo_handler,
        ));
        registry.register(FunctionTool::new(
            "mike",
            "Mike tool",
            json!({}),
            echo_handler,
        ));

        let specs = registry.specs();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].name, "alpha");
        assert_eq!(specs[1].name, "mike");
        assert_eq!(specs[2].name, "zebra");
    }

    #[test]
    fn function_tool_default_classification_is_false() {
        let tool = FunctionTool::new("test", "desc", json!({}), |_| async { Ok(json!(null)) });
        assert!(!tool.is_read_only());
        assert!(!tool.is_concurrency_safe());
    }

    #[test]
    fn function_tool_classification_builder() {
        let tool = FunctionTool::new("test", "desc", json!({}), |_| async { Ok(json!(null)) })
            .read_only()
            .concurrency_safe();
        assert!(tool.is_read_only());
        assert!(tool.is_concurrency_safe());
    }

    #[test]
    fn external_tool_default_classification_is_false() {
        let tool = ExternalTool::new("test", "desc", json!({}));
        assert!(!tool.is_read_only());
        assert!(!tool.is_concurrency_safe());
    }

    #[test]
    fn external_tool_classification_builder() {
        let tool = ExternalTool::new("test", "desc", json!({}))
            .read_only()
            .concurrency_safe();
        assert!(tool.is_read_only());
        assert!(tool.is_concurrency_safe());
    }
}
