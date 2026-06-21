//! Tool definitions and tool-call routing primitives.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-schema-backed tool definition exposed to a chat provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Stable tool name visible to the model.
    pub name: String,
    /// Human-readable tool description.
    pub description: String,
    /// JSON schema describing accepted arguments.
    pub parameters_schema: Value,
}

impl ToolSpec {
    /// Creates a new tool definition.
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
        }
    }
}

/// Tool selection policy for a chat request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ToolChoice {
    /// Let the provider or model decide whether a tool is needed.
    #[default]
    Auto,
    /// Disable tool calls for this request.
    None,
    /// Require at least one tool call.
    Required,
    /// Force a specific tool by name.
    Tool {
        /// Tool name to force.
        name: String,
    },
}

/// Tool call emitted by an assistant message or stream event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-generated call identifier.
    pub id: String,
    /// Tool name requested by the provider.
    pub name: String,
    /// JSON arguments for the tool invocation.
    pub arguments: Value,
}

impl ToolCall {
    /// Creates a tool call.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}
