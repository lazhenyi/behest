//! Tool definitions, tool-call routing primitives, and runtime limits.
//!
//! All limit types use `Option<...>` semantics: `None` means "no limit".
//! Nothing is enforced unless the user explicitly configures it.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::cache::CacheControl;

/// JSON-schema-backed tool definition exposed to a chat provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Stable tool name visible to the model.
    pub name: String,
    /// Human-readable tool description.
    pub description: String,
    /// JSON schema describing accepted arguments.
    pub parameters_schema: Value,
    /// Optional cache marker applied to this tool's prefix.
    ///
    /// When set, providers that support explicit cache markers (Anthropic)
    /// will cache the tool-definition prefix up to and including this
    /// tool. Ignored by providers that perform automatic caching.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
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
            cache_control: None,
        }
    }

    /// Returns a copy of this tool spec with the given cache control marker.
    #[must_use]
    pub fn with_cache_control(mut self, ctrl: CacheControl) -> Self {
        self.cache_control = Some(ctrl);
        self
    }
}

/// Tool selection policy for a chat request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
#[non_exhaustive]
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

// ── Runtime limits & sandbox (all optional — None = no enforcement) ──

/// Where a tool is executed.
///
/// `Server` tools run inside the agent runtime (the loop consumes results
/// and continues). `Client` tools are surfaced to the caller and the loop
/// pauses — the caller executes them and resumes the loop with the results.
///
/// Use this tag when registering a tool to control the dispatch loop:
/// - Register MCP-provided tools as `Server` — the runtime handles them.
/// - Register application-specific callbacks as `Client` when the caller
///   wants to intercept, audit, or transform tool results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSource {
    /// Executed by the runtime. Results are injected into the conversation
    /// and the loop continues automatically.
    #[default]
    Server,
    /// Surfaced to the caller. The loop pauses, returning the call to the
    /// caller, who must execute it and resume via `submit_tool_results`.
    Client,
}

impl ToolSource {
    /// Returns `true` for the server variant.
    #[must_use]
    pub const fn is_server(self) -> bool {
        matches!(self, Self::Server)
    }

    /// Returns `true` for the client variant.
    #[must_use]
    pub const fn is_client(self) -> bool {
        matches!(self, Self::Client)
    }
}

/// Hierarchical sandbox profile for tool execution.
///
/// Tools declare their required profile; the runtime enforces a configured
/// maximum.  Profiles are ordered: `Inherited < ReadOnly < WorkspaceWrite
/// < NetworkEnabled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxProfile {
    /// Inherit the runtime default.  Least privilege.
    Inherited = 0,
    /// Read-only file access.  No writes, no network.
    ReadOnly = 1,
    /// Workspace writes allowed (but no escape).  No network.
    WorkspaceWrite = 2,
    /// Full network access + workspace writes.
    NetworkEnabled = 3,
}

impl SandboxProfile {
    /// Returns `true` when `self` profile allows the `requested` profile.
    #[must_use]
    pub fn allows(self, requested: Self) -> bool {
        self >= requested
    }
}

/// Optional per-tool runtime limits.
///
/// Every field is `Option` — `None` means "unlimited / default behaviour".
/// Users construct limits via `ToolRuntimeLimits::default()` and override
/// the fields they care about.
///
/// # Example
///
/// ```ignore
/// let limits = ToolRuntimeLimits {
///     max_timeout: Some(Duration::from_secs(120)),
///     max_sandbox: Some(SandboxProfile::ReadOnly),
///     ..ToolRuntimeLimits::default()
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ToolRuntimeLimits {
    /// Maximum serialized JSON bytes for one tool input (`None` = no limit).
    pub max_input_bytes: Option<usize>,
    /// Maximum serialized JSON bytes for one tool output (`None` = no limit).
    pub max_output_bytes: Option<usize>,
    /// Maximum execution time for one tool call (`None` = no limit).
    pub max_timeout: Option<Duration>,
    /// Maximum number of tools executing concurrently (`None` = no limit).
    pub max_parallel_tools: Option<usize>,
    /// Maximum number of tool calls in one turn (`None` = no limit).
    pub max_calls_per_turn: Option<usize>,
    /// Maximum cumulative output bytes in one turn (`None` = no limit).
    pub max_cumulative_output_bytes: Option<usize>,
    /// Highest sandbox profile the runtime will execute (`None` = no limit).
    pub max_sandbox: Option<SandboxProfile>,
}

impl ToolRuntimeLimits {
    /// Creates limits with all fields unset (no enforcement).
    #[must_use]
    pub const fn none() -> Self {
        Self {
            max_input_bytes: None,
            max_output_bytes: None,
            max_timeout: None,
            max_parallel_tools: None,
            max_calls_per_turn: None,
            max_cumulative_output_bytes: None,
            max_sandbox: None,
        }
    }

    /// Returns `true` when no limit is configured (everything is `None`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.max_input_bytes.is_none()
            && self.max_output_bytes.is_none()
            && self.max_timeout.is_none()
            && self.max_parallel_tools.is_none()
            && self.max_calls_per_turn.is_none()
            && self.max_cumulative_output_bytes.is_none()
            && self.max_sandbox.is_none()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_profile_ordering() {
        assert!(SandboxProfile::ReadOnly > SandboxProfile::Inherited);
        assert!(SandboxProfile::WorkspaceWrite > SandboxProfile::ReadOnly);
        assert!(SandboxProfile::NetworkEnabled > SandboxProfile::WorkspaceWrite);
    }

    #[test]
    fn sandbox_readonly_allows_readonly() {
        assert!(SandboxProfile::ReadOnly.allows(SandboxProfile::ReadOnly));
    }

    #[test]
    fn sandbox_readonly_rejects_workspace_write() {
        assert!(!SandboxProfile::ReadOnly.allows(SandboxProfile::WorkspaceWrite));
    }

    #[test]
    fn sandbox_network_allows_everything() {
        assert!(SandboxProfile::NetworkEnabled.allows(SandboxProfile::Inherited));
        assert!(SandboxProfile::NetworkEnabled.allows(SandboxProfile::ReadOnly));
        assert!(SandboxProfile::NetworkEnabled.allows(SandboxProfile::WorkspaceWrite));
        assert!(SandboxProfile::NetworkEnabled.allows(SandboxProfile::NetworkEnabled));
    }

    #[test]
    fn empty_limits_is_empty() {
        assert!(ToolRuntimeLimits::none().is_empty());
        assert!(ToolRuntimeLimits::default().is_empty());
    }

    #[test]
    fn partial_limits_is_not_empty() {
        let limits = ToolRuntimeLimits {
            max_timeout: Some(std::time::Duration::from_secs(30)),
            ..ToolRuntimeLimits::none()
        };
        assert!(!limits.is_empty());
    }

    #[test]
    fn tool_source_default_is_server() {
        assert_eq!(ToolSource::default(), ToolSource::Server);
        assert!(ToolSource::Server.is_server());
        assert!(!ToolSource::Server.is_client());
        assert!(ToolSource::Client.is_client());
        assert!(!ToolSource::Client.is_server());
    }

    #[test]
    fn tool_source_serialize_roundtrip() {
        let server = serde_json::to_value(ToolSource::Server).unwrap();
        assert_eq!(server, serde_json::json!("server"));
        let client: ToolSource = serde_json::from_value(serde_json::json!("client")).unwrap();
        assert_eq!(client, ToolSource::Client);
    }
}
