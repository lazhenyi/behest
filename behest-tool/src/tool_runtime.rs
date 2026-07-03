//! ToolRuntime — optional enforcement of runtime limits, hooks, and scoping.
//!
//! Unlike [`ToolRegistry`](crate::ToolRegistry), which is a bare lookup table, [`ToolRuntime`]
//! enforces the configured [`ToolRuntimeLimits`] and runs registered hooks.
//! Use [`ToolRuntime::scoped`] to create filtered views with allow/deny lists.
//!
//! Everything is opt-in: construct `ToolRuntime::default()` for zero enforcement,
//! or use [`ToolRuntime::builder()`] to add limits and hooks explicitly.
//!
//! # Example
//!
//! ```ignore
//! use behest_tool::tool_runtime::ToolRuntime;
//! use behest_core::tool_types::{ToolRuntimeLimits, SandboxProfile};
//!
//! let mut runtime = ToolRuntime::builder()
//!     .limit_max_timeout(Duration::from_secs(120))
//!     .limit_max_sandbox(SandboxProfile::WorkspaceWrite)
//!     .hook(my_hook)
//!     .build();
//!
//! runtime.register(my_tool);
//! let result = runtime.execute(&ctx, &tool_call).await?;
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use behest_context::ToolContext;
use behest_core::tool_types::{SandboxProfile, ToolCall, ToolRuntimeLimits};
use serde_json::Value;

use crate::{Tool, ToolOutput, ToolResult};

/// Lifecycle hook run before and after tool execution.
///
/// Hooks are purely additive — they cannot block execution. Return `Err`
/// to convert the tool result into an error result; return `Ok` to pass
/// through (or modify) the tool outcome.
#[async_trait]
pub trait ToolHook: Send + Sync {
    /// Runs before the tool executes.
    ///
    /// Returning `Err` short-circuits execution — the tool never runs.
    async fn before_execute(
        &self,
        _tool_name: &str,
        _tool_call_id: &str,
        _input: &Value,
    ) -> Result<(), crate::ToolError> {
        Ok(())
    }

    /// Runs after the tool executes (even if the tool returned an error).
    ///
    /// Use this for auditing, logging, or modifying the final result.
    async fn after_execute(
        &self,
        _tool_name: &str,
        _tool_call_id: &str,
        result: &mut ToolResult<ToolOutput>,
    ) {
        let _ = result;
    }
}

/// A registered tool together with its source metadata.
struct ToolEntry {
    tool: Arc<dyn Tool>,
}

impl ToolEntry {
    fn tool(&self) -> &Arc<dyn Tool> {
        &self.tool
    }
}

/// Runtime that enforces optional limits and hooks on tool execution.
pub struct ToolRuntime {
    registry: BTreeMap<String, ToolEntry>,
    hooks: Vec<Arc<dyn ToolHook>>,
    limits: ToolRuntimeLimits,
}

impl std::fmt::Debug for ToolRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.registry.keys().map(String::as_str).collect();
        f.debug_struct("ToolRuntime")
            .field("tools", &names)
            .field("limits", &self.limits)
            .field("hooks_count", &self.hooks.len())
            .finish()
    }
}

impl Default for ToolRuntime {
    fn default() -> Self {
        Self {
            registry: BTreeMap::new(),
            hooks: Vec::new(),
            limits: ToolRuntimeLimits::none(),
        }
    }
}

impl ToolRuntime {
    /// Creates an empty runtime with no limits.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a builder for constructing a runtime with limits and hooks.
    #[must_use]
    pub fn builder() -> ToolRuntimeBuilder {
        ToolRuntimeBuilder::new()
    }

    /// Registers a tool, returning any previous tool with the same name.
    pub fn register<T: Tool + 'static>(&mut self, tool: T) -> Option<Arc<dyn Tool>> {
        self.registry
            .insert(
                tool.name().to_string(),
                ToolEntry {
                    tool: Arc::new(tool),
                },
            )
            .map(|entry| entry.tool)
    }

    /// Registers an already-shared tool.
    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) -> Option<Arc<dyn Tool>> {
        self.registry
            .insert(tool.name().to_string(), ToolEntry { tool })
            .map(|entry| entry.tool)
    }

    /// Unregisters a tool by name.
    pub fn unregister(&mut self, name: &str) -> Option<Arc<dyn Tool>> {
        self.registry.remove(name).map(|entry| entry.tool)
    }

    /// Returns a reference to a tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.registry.get(name).map(|entry| entry.tool())
    }

    /// Returns all registered tool names, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.registry.keys().cloned().collect();
        names.sort();
        names
    }

    /// Returns the number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.registry.len()
    }

    /// Returns `true` if no tools are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.registry.is_empty()
    }

    /// Registers a hook.
    pub fn add_hook<H: ToolHook + 'static>(&mut self, hook: H) {
        self.hooks.push(Arc::new(hook));
    }

    /// Returns the current runtime limits.
    #[must_use]
    pub fn limits(&self) -> &ToolRuntimeLimits {
        &self.limits
    }

    /// Creates a scoped view with a tool allowlist.
    ///
    /// Calls to a tool not in the allowlist return a "tool not available" error.
    #[must_use]
    pub fn scoped(&self, filter: ScopedFilter) -> ScopedToolRuntime<'_> {
        ScopedToolRuntime {
            inner: self,
            filter,
        }
    }

    /// Executes a single tool call, enforcing limits and running hooks.
    pub async fn execute(&self, ctx: &dyn ToolContext, call: &ToolCall) -> ToolResult<ToolOutput> {
        let entry = self.registry.get(&call.name).ok_or_else(|| {
            behest_core::error::ToolError::NotFound {
                name: call.name.clone(),
            }
        })?;
        let tool = entry.tool();
        self.execute_tool(tool, ctx, call).await
    }

    /// Executes a batch of tool calls sequentially.
    ///
    /// Each call goes through the full enforcement pipeline (limits, hooks, scoping).
    /// For parallel execution strategies, use [`ToolExecutionStrategy`](crate::ToolExecutionStrategy)
    /// at a higher level.
    pub async fn execute_batch(
        &self,
        ctx: &dyn ToolContext,
        calls: &[ToolCall],
    ) -> Vec<ToolResult<ToolOutput>> {
        if calls.is_empty() {
            return Vec::new();
        }

        // Check per-turn call limit
        let call_limit = self.limits.max_calls_per_turn.unwrap_or(usize::MAX);
        if calls.len() > call_limit {
            return calls
                .iter()
                .map(|call| {
                    Err(behest_core::error::ToolError::InvalidArguments {
                        name: call.name.clone(),
                        message: format!(
                            "tool call batch ({}) exceeds runtime limit ({})",
                            calls.len(),
                            call_limit
                        ),
                    })
                })
                .collect();
        }

        let mut results = Vec::with_capacity(calls.len());
        let mut cumulative_output_bytes: usize = 0;

        for call in calls {
            let result = self.execute(ctx, call).await;

            // Track cumulative output
            if let Ok(ref output) = result {
                cumulative_output_bytes += serde_json::to_vec(&output.value)
                    .map(|v| v.len())
                    .unwrap_or(0);
            }

            // Check cumulative output limit
            if let Some(max_cumulative) = self.limits.max_cumulative_output_bytes
                && cumulative_output_bytes > max_cumulative
            {
                // Truncate: mark remaining calls as errors
                let remaining = calls.len() - results.len();
                results.push(Err(behest_core::error::ToolError::Execution {
                    name: "batch".to_string(),
                    message: format!(
                        "cumulative tool output ({cumulative_output_bytes} bytes) \
                         exceeded runtime limit ({max_cumulative} bytes)"
                    ),
                }));
                // Ensure we don't put the current result twice
                if remaining > 0 {
                    continue; // skip pushing current result, handled above
                }
            }

            results.push(result);
        }

        results
    }

    /// Internal: execute one tool through the full pipeline.
    async fn execute_tool(
        &self,
        tool: &Arc<dyn Tool>,
        ctx: &dyn ToolContext,
        call: &ToolCall,
    ) -> ToolResult<ToolOutput> {
        let name = tool.name();
        let tool_call_id = &call.id;

        // ── Input size check ──
        if let Some(max_input) = self.limits.max_input_bytes {
            let input_bytes = serde_json::to_vec(&call.arguments)
                .map(|v| v.len())
                .unwrap_or(0);
            if input_bytes > max_input {
                return Err(behest_core::error::ToolError::InvalidArguments {
                    name: name.to_string(),
                    message: format!(
                        "tool input ({input_bytes} bytes) exceeds limit ({max_input} bytes)"
                    ),
                });
            }
        }

        // ── Sandbox check ──
        if let Some(max_sandbox) = self.limits.max_sandbox {
            let required = tool.sandbox_profile();
            if !max_sandbox.allows(required) {
                return Err(behest_core::error::ToolError::Execution {
                    name: name.to_string(),
                    message: format!(
                        "tool sandbox {required:?} exceeds runtime limit {max_sandbox:?}"
                    ),
                });
            }
        }

        // ── Pre-execution hooks ──
        for hook in &self.hooks {
            hook.before_execute(name, tool_call_id, &call.arguments)
                .await?;
        }

        // ── Execution (with optional timeout) ──
        let effective_timeout = tool.timeout().or(self.limits.max_timeout);

        let started = Instant::now();
        let mut result = if let Some(timeout) = effective_timeout {
            match tokio::time::timeout(timeout, tool.execute_with_ctx(ctx, call.arguments.clone()))
                .await
            {
                Ok(inner) => inner,
                Err(_elapsed) => Err(behest_core::error::ToolError::Execution {
                    name: name.to_string(),
                    message: format!("tool execution timed out after {}ms", timeout.as_millis()),
                }),
            }
        } else {
            tool.execute_with_ctx(ctx, call.arguments.clone()).await
        };

        // ── Output size check ──
        if let Ok(ref output) = result
            && let Some(max_output) = self.limits.max_output_bytes
        {
            let output_bytes = serde_json::to_vec(&output.value)
                .map(|v| v.len())
                .unwrap_or(0);
            if output_bytes > max_output {
                result = Err(behest_core::error::ToolError::Execution {
                    name: name.to_string(),
                    message: format!(
                        "tool output ({output_bytes} bytes) exceeds limit ({max_output} bytes)"
                    ),
                });
            }
        }

        let elapsed = started.elapsed();

        // ── Post-execution hooks ──
        for hook in &self.hooks {
            hook.after_execute(name, tool_call_id, &mut result).await;
        }

        let _ = elapsed; // available for future metrics
        result
    }
}

// ── Builder ──

/// Builder for [`ToolRuntime`].
pub struct ToolRuntimeBuilder {
    limits: ToolRuntimeLimits,
    hooks: Vec<Arc<dyn ToolHook>>,
}

impl ToolRuntimeBuilder {
    /// Creates a new builder with no limits or hooks.
    #[must_use]
    pub fn new() -> Self {
        Self {
            limits: ToolRuntimeLimits::none(),
            hooks: Vec::new(),
        }
    }

    /// Set the maximum tool input size in bytes.
    #[must_use]
    pub fn limit_max_input_bytes(mut self, bytes: usize) -> Self {
        self.limits.max_input_bytes = Some(bytes);
        self
    }

    /// Set the maximum tool output size in bytes.
    #[must_use]
    pub fn limit_max_output_bytes(mut self, bytes: usize) -> Self {
        self.limits.max_output_bytes = Some(bytes);
        self
    }

    /// Set the maximum tool execution time.
    #[must_use]
    pub fn limit_max_timeout(mut self, timeout: Duration) -> Self {
        self.limits.max_timeout = Some(timeout);
        self
    }

    /// Set the maximum number of concurrent tool executions.
    #[must_use]
    pub fn limit_max_parallel_tools(mut self, count: usize) -> Self {
        self.limits.max_parallel_tools = Some(count);
        self
    }

    /// Set the maximum number of tool calls per turn.
    #[must_use]
    pub fn limit_max_calls_per_turn(mut self, count: usize) -> Self {
        self.limits.max_calls_per_turn = Some(count);
        self
    }

    /// Set the maximum cumulative output bytes per turn.
    #[must_use]
    pub fn limit_max_cumulative_output_bytes(mut self, bytes: usize) -> Self {
        self.limits.max_cumulative_output_bytes = Some(bytes);
        self
    }

    /// Set the highest sandbox profile allowed.
    #[must_use]
    pub fn limit_max_sandbox(mut self, profile: SandboxProfile) -> Self {
        self.limits.max_sandbox = Some(profile);
        self
    }

    /// Register a tool hook.
    #[must_use]
    pub fn hook<H: ToolHook + 'static>(mut self, hook: H) -> Self {
        self.hooks.push(Arc::new(hook));
        self
    }

    /// Build the runtime.
    #[must_use]
    pub fn build(self) -> ToolRuntime {
        ToolRuntime {
            registry: BTreeMap::new(),
            hooks: self.hooks,
            limits: self.limits,
        }
    }
}

impl Default for ToolRuntimeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Scoped runtime ──

/// A filtered view over a [`ToolRuntime`].
///
/// Tool calls to names not in the allowlist (or names in the denylist)
/// are rejected with "tool not available" before reaching the tool.
pub struct ScopedToolRuntime<'a> {
    inner: &'a ToolRuntime,
    filter: ScopedFilter,
}

/// Filter applied by [`ScopedToolRuntime`].
///
/// Empty allowlist means all tools are allowed (subject to denylist).
/// Denylist is checked first — if a tool is in both lists, it is denied.
#[derive(Debug, Clone, Default)]
pub struct ScopedFilter {
    /// If non-empty, only tools in this list are visible.
    pub allowlist: Vec<String>,
    /// Tools in this list are never visible.
    pub denylist: Vec<String>,
}

impl ScopedFilter {
    /// Creates a filter that allows only the given tools.
    #[must_use]
    pub fn allow_only(names: Vec<String>) -> Self {
        Self {
            allowlist: names,
            denylist: Vec::new(),
        }
    }

    /// Creates a filter that denies the given tools but allows everything else.
    #[must_use]
    pub fn deny(names: Vec<String>) -> Self {
        Self {
            allowlist: Vec::new(),
            denylist: names,
        }
    }

    /// Returns `true` when the named tool passes this filter.
    #[must_use]
    pub fn allows(&self, name: &str) -> bool {
        if self.denylist.iter().any(|d| d == name) {
            return false;
        }
        if self.allowlist.is_empty() {
            return true;
        }
        self.allowlist.iter().any(|a| a == name)
    }
}

impl ScopedToolRuntime<'_> {
    /// Executes a tool call if the tool passes the scope filter.
    pub async fn execute(&self, ctx: &dyn ToolContext, call: &ToolCall) -> ToolResult<ToolOutput> {
        if !self.filter.allows(&call.name) {
            return Err(behest_core::error::ToolError::Execution {
                name: call.name.clone(),
                message: format!("tool `{}` is not available for this agent", call.name),
            });
        }
        self.inner.execute(ctx, call).await
    }

    /// Executes a batch of tool calls through the scope filter.
    pub async fn execute_batch(
        &self,
        ctx: &dyn ToolContext,
        calls: &[ToolCall],
    ) -> Vec<ToolResult<ToolOutput>> {
        let calls: Vec<ToolCall> = calls.to_vec();

        // Filter out denied calls
        let mut results: Vec<Option<ToolResult<ToolOutput>>> =
            (0..calls.len()).map(|_| None).collect();

        let mut allowed_indices: Vec<usize> = Vec::new();
        let mut allowed_calls: Vec<ToolCall> = Vec::new();

        for (i, call) in calls.iter().enumerate() {
            if !self.filter.allows(&call.name) {
                results[i] = Some(Err(behest_core::error::ToolError::Execution {
                    name: call.name.clone(),
                    message: format!("tool `{}` is not available for this agent", call.name),
                }));
            } else {
                allowed_indices.push(i);
                allowed_calls.push(call.clone());
            }
        }

        if allowed_calls.is_empty() {
            // Every slot was pre-filled with a denied-call error
            return results.into_iter().flatten().collect();
        }

        let allowed_results = self.inner.execute_batch(ctx, &allowed_calls).await;

        for (i, result) in allowed_indices.into_iter().zip(allowed_results) {
            results[i] = Some(result);
        }

        // Every slot is now either a denied error or an execution result
        results.into_iter().flatten().collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::FunctionTool;
    use behest_context::{
        AppContext, EventSink, RunBudget, RunContextImpl, SessionContextImpl, SessionState,
        ToolContextImpl,
    };
    use behest_core::id::RunId;
    use behest_core::tool_types::ToolCall;
    use tokio_util::sync::CancellationToken;

    fn make_tool(name: &str) -> FunctionTool {
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
    }

    fn make_long_tool(name: &str, delay_ms: u64) -> FunctionTool {
        let n = name.to_string();
        FunctionTool::new(
            name,
            format!("Tool {name}"),
            serde_json::json!({"type": "object", "properties": {}}),
            move |_args| {
                let n = n.clone();
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    Ok(serde_json::Value::String(format!("{n} done")))
                })
            },
        )
        .timeout(Duration::from_millis(100))
    }

    fn tool_ctx(call_id: &str, tool_name: &str) -> ToolContextImpl {
        let app = AppContext {
            invocation_id: "inv-test".to_string(),
            session_id: "sess-test".to_string(),
            user_id: "user-test".to_string(),
            app_name: "test".to_string(),
        };
        let session = SessionContextImpl {
            app,
            state: SessionState::new(),
        };
        let run = RunContextImpl {
            session,
            run_id: RunId::new(),
            cancel: CancellationToken::new(),
            deadline: None,
            sink: EventSink::new(),
            budget: RunBudget::new(None),
        };
        ToolContextImpl {
            run,
            tool_call: ToolCall::new(call_id, tool_name, serde_json::Value::Null),
        }
    }

    #[tokio::test]
    async fn empty_runtime_returns_not_found() {
        let rt = ToolRuntime::new();
        let ctx = tool_ctx("c1", "echo");
        let call = ToolCall::new("c1", "echo", serde_json::Value::Null);
        let result = rt.execute(&ctx, &call).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn registered_tool_executes_successfully() {
        let mut rt = ToolRuntime::new();
        rt.register(make_tool("echo"));
        let ctx = tool_ctx("c1", "echo");
        let call = ToolCall::new("c1", "echo", serde_json::Value::Null);
        let result = rt.execute(&ctx, &call).await.unwrap();
        assert_eq!(result.value, "echo done");
    }

    #[tokio::test]
    async fn input_size_limit_rejects_overly_large_input() {
        let mut rt = ToolRuntime::builder().limit_max_input_bytes(10).build();
        rt.register(make_tool("echo"));
        let ctx = tool_ctx("c1", "echo");
        let call = ToolCall::new("c1", "echo", serde_json::json!("a very long input string"));
        let result = rt.execute(&ctx, &call).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exceeds limit"));
    }

    #[tokio::test]
    async fn timeout_enforced() {
        let mut rt = ToolRuntime::builder()
            .limit_max_timeout(Duration::from_millis(50))
            .build();
        rt.register(make_long_tool("slow", 500));
        let ctx = tool_ctx("c1", "slow");
        let call = ToolCall::new("c1", "slow", serde_json::Value::Null);
        let result = rt.execute(&ctx, &call).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn sandbox_check_enforced() {
        let mut rt = ToolRuntime::builder()
            .limit_max_sandbox(SandboxProfile::ReadOnly)
            .build();
        let web_tool = make_tool("web_fetch").sandbox(SandboxProfile::NetworkEnabled);
        rt.register(web_tool);
        let ctx = tool_ctx("c1", "web_fetch");
        let call = ToolCall::new("c1", "web_fetch", serde_json::Value::Null);
        let result = rt.execute(&ctx, &call).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("sandbox"));
    }

    #[tokio::test]
    async fn hook_before_and_after_fires() {
        use std::sync::Mutex;

        struct AuditHook {
            calls: Mutex<Vec<String>>,
        }

        #[async_trait]
        impl ToolHook for AuditHook {
            async fn before_execute(
                &self,
                name: &str,
                _call_id: &str,
                _input: &Value,
            ) -> Result<(), crate::ToolError> {
                self.calls.lock().unwrap().push(format!("before:{name}"));
                Ok(())
            }

            async fn after_execute(
                &self,
                name: &str,
                _call_id: &str,
                _result: &mut ToolResult<ToolOutput>,
            ) {
                self.calls.lock().unwrap().push(format!("after:{name}"));
            }
        }

        let hook = Arc::new(AuditHook {
            calls: Mutex::new(Vec::new()),
        });
        let hook_clone = Arc::clone(&hook);

        struct HookWrapper {
            inner: Arc<AuditHook>,
        }

        #[async_trait]
        impl ToolHook for HookWrapper {
            async fn before_execute(
                &self,
                name: &str,
                call_id: &str,
                input: &Value,
            ) -> Result<(), crate::ToolError> {
                self.inner.before_execute(name, call_id, input).await
            }

            async fn after_execute(
                &self,
                name: &str,
                call_id: &str,
                result: &mut ToolResult<ToolOutput>,
            ) {
                self.inner.after_execute(name, call_id, result).await;
            }
        }

        let mut rt = ToolRuntime::builder()
            .hook(HookWrapper { inner: hook_clone })
            .build();
        rt.register(make_tool("echo"));
        let ctx = tool_ctx("c1", "echo");
        let call = ToolCall::new("c1", "echo", serde_json::Value::Null);
        let result = rt.execute(&ctx, &call).await;
        assert!(result.is_ok());

        let calls = hook.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], "before:echo");
        assert_eq!(calls[1], "after:echo");
    }

    #[tokio::test]
    async fn scoped_filter_allows_only_listed() {
        let mut rt = ToolRuntime::new();
        rt.register(make_tool("read"));
        rt.register(make_tool("write"));

        let scoped = rt.scoped(ScopedFilter::allow_only(vec!["read".to_string()]));
        let ctx = tool_ctx("c1", "read");

        let ok = scoped
            .execute(&ctx, &ToolCall::new("c1", "read", serde_json::Value::Null))
            .await;
        assert!(ok.is_ok());

        let err = scoped
            .execute(&ctx, &ToolCall::new("c2", "write", serde_json::Value::Null))
            .await;
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("not available"));
    }

    #[tokio::test]
    async fn scoped_denylist_blocks_unlisted_allows() {
        let mut rt = ToolRuntime::new();
        rt.register(make_tool("read"));
        rt.register(make_tool("write"));

        let scoped = rt.scoped(ScopedFilter::deny(vec!["write".to_string()]));
        let ctx = tool_ctx("c1", "read");

        assert!(
            scoped
                .execute(&ctx, &ToolCall::new("c1", "read", serde_json::Value::Null))
                .await
                .is_ok()
        );
        assert!(
            scoped
                .execute(&ctx, &ToolCall::new("c2", "write", serde_json::Value::Null))
                .await
                .is_err()
        );
    }
}
