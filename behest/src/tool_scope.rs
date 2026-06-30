//! LIFO scoped tool registry with shadow-stack semantics.
//!
//! [`ScopedToolRegistry`] wraps a base [`ToolRegistry`] with a stack of
//! named scopes. Tools registered in higher scopes shadow those with the
//! same name in lower scopes. When a scope is popped, all tools registered
//! in that scope are removed, revealing any previously-shadowed tools.
//!
//! # Scope model
//!
//! ```text
//!   Scope 2 (Turn)     ── top
//!   Scope 1 (Run)
//!   Scope 0 (Agent)    ── bottom
//!   Base registry       ── always present
//! ```
//!
//! Resolution order: top scope → ... → bottom scope → base registry.
//! First match wins.
//!
//! # Thread safety
//!
//! The scope stack uses interior mutability (`Mutex`) so that scopes can
//! be pushed/popped through a shared reference. The base registry is
//! lock-free. Cloning captures the current scope snapshot; cloned copies
//! operate independently.
//!
//! # Example
//!
//! ```rust
//! use behest::tool::{FunctionTool, ToolRegistry};
//! use behest::tool_scope::ScopedToolRegistry;
//! use serde_json::json;
//!
//! let base = ToolRegistry::new();
//! let scoped = ScopedToolRegistry::new(base);
//!
//! let scope = scoped.push_scope();
//! scoped.register_in_scope(scope, FunctionTool::new(
//!     "my_tool", "desc", json!({}),
//!     |args| async move { Ok(args) },
//! ));
//!
//! assert!(scoped.get("my_tool").is_some());
//!
//! scoped.pop_scope(scope);
//! assert!(scoped.get("my_tool").is_none());
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::provider::ToolSpec;
use crate::tool::{Tool, ToolRegistry};

/// Scope identifier returned by [`ScopedToolRegistry::push_scope`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(pub(crate) usize);

/// Well-known scope levels for documentation and ordering.
///
/// These are advisory labels — the actual ordering is determined by
/// push/pop sequence, not by the enum value. Use these as convention
/// when building runtime integrations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopeLevel {
    /// Global tools available to all agents (e.g., shell, file system).
    Global = 0,
    /// Agent-level tools registered when an agent definition is loaded.
    Agent = 1,
    /// Run-level tools for the lifetime of a single `run()` invocation.
    Run = 2,
    /// Turn-level tools for a single iteration within a run.
    Turn = 3,
}

/// A layer in the tool scope stack, holding tools registered at that scope.
#[derive(Clone)]
struct ScopeLayer {
    tools: HashMap<String, Arc<dyn Tool>>,
}

/// Interior state protected by a mutex.
struct ScopeState {
    layers: Vec<(ScopeId, ScopeLayer)>,
    next_id: usize,
}

impl Clone for ScopeState {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            next_id: self.next_id,
        }
    }
}

/// Result of registering an already-Arc'd tool in a scope.
type RegisterArcResult = Result<Option<Arc<dyn Tool>>, (String, Arc<dyn Tool>)>;

/// LIFO scoped tool registry with shadow-stack semantics.
///
/// Wraps a base [`ToolRegistry`] with a stack of scopes. Tools in
/// higher scopes shadow identically-named tools in lower scopes.
/// Popping a scope removes all its tools and restores shadows.
///
/// The scope stack is protected by a `Mutex` for interior mutability,
/// enabling push/pop through `&self`. The base registry is lock-free.
pub struct ScopedToolRegistry {
    base: ToolRegistry,
    state: Mutex<ScopeState>,
}

impl Clone for ScopedToolRegistry {
    fn clone(&self) -> Self {
        let state = self.lock_state_poison_safe();
        Self {
            base: self.base.clone(),
            state: Mutex::new(state.clone()),
        }
    }
}

impl ScopedToolRegistry {
    /// Creates a new scoped registry with the given base.
    ///
    /// The base registry provides tools at the lowest resolution level.
    #[must_use]
    pub fn new(base: ToolRegistry) -> Self {
        Self {
            base,
            state: Mutex::new(ScopeState {
                layers: Vec::new(),
                next_id: 0,
            }),
        }
    }

    /// Pushes a new empty scope onto the stack and returns its [`ScopeId`].
    #[must_use]
    pub fn push_scope(&self) -> ScopeId {
        let mut state = self.lock_state_poison_safe();
        let id = ScopeId(state.next_id);
        state.next_id += 1;
        state.layers.push((
            id,
            ScopeLayer {
                tools: HashMap::new(),
            },
        ));
        id
    }

    /// Pops the given scope from the stack.
    ///
    /// Returns `true` if the scope was found and removed, `false` if it
    /// was not the top scope or does not exist. Only the topmost scope
    /// can be popped — attempting to pop a non-top scope is a no-op.
    pub fn pop_scope(&self, id: ScopeId) -> bool {
        let mut state = self.lock_state_poison_safe();
        if let Some(&(top_id, _)) = state.layers.last()
            && top_id == id
        {
            state.layers.pop();
            return true;
        }
        false
    }

    /// Registers a tool in the specified scope.
    ///
    /// If a tool with the same name already exists in that scope, it is
    /// replaced and the old tool is returned. Tools in other scopes or
    /// the base registry are not affected — they are simply shadowed.
    ///
    /// # Errors
    ///
    /// Returns the tool back if the scope does not exist.
    pub fn register_in_scope<T: Tool + 'static>(
        &self,
        scope: ScopeId,
        tool: T,
    ) -> Result<Option<Arc<dyn Tool>>, T> {
        let name = tool.name().to_owned();
        let mut state = self.lock_state_poison_safe();
        for (id, layer) in &mut state.layers {
            if *id == scope {
                let arc: Arc<dyn Tool> = Arc::new(tool);
                return Ok(layer.tools.insert(name, arc));
            }
        }
        Err(tool)
    }

    /// Registers an already-`Arc`'d tool in the specified scope.
    ///
    /// Returns `Ok(old)` if the scope exists, or `Err((name, tool))` if not.
    ///
    /// # Errors
    ///
    /// Returns `Err((name, tool))` when the requested `scope` does not exist
    /// in the layer stack. The returned tuple lets the caller recover the
    /// name and `Arc`'d tool without an extra allocation.
    pub fn register_arc_in_scope(
        &self,
        scope: ScopeId,
        name: String,
        tool: Arc<dyn Tool>,
    ) -> RegisterArcResult {
        let mut state = self.lock_state_poison_safe();
        for (id, layer) in &mut state.layers {
            if *id == scope {
                return Ok(layer.tools.insert(name, tool));
            }
        }
        Err((name, tool))
    }

    /// Returns a tool by searching from top scope to base.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let state = self.lock_state_poison_safe();
        // Search scopes top-down (last = topmost).
        for (_, layer) in state.layers.iter().rev() {
            if let Some(tool) = layer.tools.get(name) {
                return Some(Arc::clone(tool));
            }
        }
        // Fall back to base.
        self.base.get(name)
    }

    /// Generates merged [`ToolSpec`]s from all scopes and base,
    /// sorted alphabetically by tool name.
    ///
    /// Tools in higher scopes shadow those with the same name in lower
    /// scopes or the base. The returned list contains one entry per
    /// unique tool name.
    ///
    /// The sorted output ensures deterministic prompt caching across
    /// turns and provider calls.
    #[must_use]
    pub fn specs(&self) -> Vec<ToolSpec> {
        let state = self.lock_state_poison_safe();
        let mut merged: HashMap<String, Arc<dyn Tool>> = HashMap::new();

        // Start with base (lowest priority).
        for spec in self.base.specs() {
            if let Some(tool) = self.base.get(&spec.name) {
                merged.insert(spec.name, tool);
            }
        }

        // Overlay each scope bottom-up (top scope wins).
        for (_, layer) in &state.layers {
            for (name, tool) in &layer.tools {
                merged.insert(name.clone(), Arc::clone(tool));
            }
        }

        let mut specs: Vec<ToolSpec> = merged.values().map(|t| t.to_spec()).collect();
        specs.sort_by(|a, b| a.name.cmp(&b.name));
        specs
    }

    /// Returns the number of active scopes.
    #[must_use]
    pub fn scope_depth(&self) -> usize {
        let state = self.lock_state_poison_safe();
        state.layers.len()
    }

    /// Returns a reference to the base registry.
    #[must_use]
    pub fn base(&self) -> &ToolRegistry {
        &self.base
    }

    /// Returns a mutable reference to the base registry.
    pub fn base_mut(&mut self) -> &mut ToolRegistry {
        &mut self.base
    }

    /// Removes a tool from the base registry by name.
    pub fn unregister_from_base(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.base.unregister(name)
    }

    /// Returns the total number of unique tools visible (base + all scopes).
    #[must_use]
    pub fn len(&self) -> usize {
        self.specs().len()
    }

    /// Returns `true` if no tools are visible at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.specs().is_empty()
    }

    /// Executes a tool call using the merged tool view.
    ///
    /// This delegates to the first matching tool from top scope to base.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::NotFound`](crate::error::ToolError::NotFound)
    /// when the tool is not in any scope or the base.
    pub async fn execute(
        &self,
        call: &crate::provider::ToolCall,
    ) -> crate::tool::ToolResult<crate::tool::ToolOutput> {
        let tool = self
            .get(&call.name)
            .ok_or_else(|| crate::error::ToolError::NotFound {
                name: call.name.clone(),
            })?;
        tool.execute(call.arguments.clone()).await
    }

    /// Pushes a new scope and returns a RAII guard that pops it on drop.
    ///
    /// This is the preferred way to manage scope lifetimes. The returned
    /// [`ScopeGuard`] automatically calls [`pop_scope`](Self::pop_scope)
    /// when dropped, ensuring cleanup even in the presence of early
    /// returns or panics.
    #[must_use]
    pub fn push_scope_guarded(self: &Arc<Self>) -> ScopeGuard {
        let id = self.push_scope();
        ScopeGuard {
            scope: Arc::clone(self),
            id,
        }
    }

    fn lock_state_poison_safe(&self) -> MutexGuard<'_, ScopeState> {
        match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

/// RAII guard that pops a scope from a [`ScopedToolRegistry`] on drop.
///
/// Created by [`ScopedToolRegistry::push_scope_guarded`]. The scope is
/// popped when the guard is dropped, regardless of whether the scope is
/// still the top of the stack (non-top pops are silently ignored by
/// the underlying [`pop_scope`](ScopedToolRegistry::pop_scope)).
pub struct ScopeGuard {
    scope: Arc<ScopedToolRegistry>,
    id: ScopeId,
}

impl ScopeGuard {
    /// Returns the [`ScopeId`] of the scope this guard will pop.
    #[must_use]
    pub fn id(&self) -> ScopeId {
        self.id
    }

    /// Consumes the guard without popping. The scope remains on the stack.
    ///
    /// Returns the underlying [`ScopedToolRegistry`] reference.
    #[must_use]
    pub fn into_inner(self) -> Arc<ScopedToolRegistry> {
        let scope = Arc::clone(&self.scope);
        std::mem::forget(self);
        scope
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        self.scope.pop_scope(self.id);
    }
}

impl std::fmt::Debug for ScopeGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopeGuard")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::type_complexity, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::tool::FunctionTool;
    use serde_json::{Value, json};

    fn make_tool(
        name: &'static str,
    ) -> FunctionTool<
        impl Fn(
            Value,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = crate::tool::ToolResult<Value>> + Send>,
        > + Send
        + Sync
        + 'static,
    > {
        FunctionTool::new(
            name,
            format!("{name} desc"),
            json!({"type": "object"}),
            |args| -> std::pin::Pin<
                Box<dyn std::future::Future<Output = crate::tool::ToolResult<Value>> + Send>,
            > { Box::pin(async move { Ok(args) }) },
        )
    }

    #[test]
    fn new_should_have_base_tools() {
        let base = ToolRegistry::new();
        base.register(make_tool("base_tool"));
        let scoped = ScopedToolRegistry::new(base);

        assert!(scoped.get("base_tool").is_some());
        assert!(scoped.get("missing").is_none());
        assert_eq!(scoped.scope_depth(), 0);
    }

    #[test]
    fn push_scope_should_create_empty_scope() {
        let base = ToolRegistry::new();
        let scoped = ScopedToolRegistry::new(base);
        let scope = scoped.push_scope();

        assert_eq!(scoped.scope_depth(), 1);
        assert!(scoped.get("anything").is_none());
        let _ = scope;
    }

    #[test]
    fn register_in_scope_should_shadow_base() {
        let base = ToolRegistry::new();
        base.register(make_tool("shared"));
        let scoped = ScopedToolRegistry::new(base);

        let scope = scoped.push_scope();
        scoped
            .register_in_scope(scope, make_tool("shared"))
            .ok()
            .unwrap();

        assert!(scoped.get("shared").is_some());
        assert_eq!(scoped.specs().len(), 1);
    }

    #[test]
    fn pop_scope_should_unshadow_base_tool() {
        let base = ToolRegistry::new();
        base.register(make_tool("shared"));
        let scoped = ScopedToolRegistry::new(base);

        let scope = scoped.push_scope();
        scoped
            .register_in_scope(scope, make_tool("shared"))
            .ok()
            .unwrap();
        assert!(scoped.get("shared").is_some());

        scoped.pop_scope(scope);
        assert!(scoped.get("shared").is_some());
        assert_eq!(scoped.scope_depth(), 0);
    }

    #[test]
    fn pop_scope_should_remove_scope_only_tools() {
        let base = ToolRegistry::new();
        let scoped = ScopedToolRegistry::new(base);

        let scope = scoped.push_scope();
        scoped
            .register_in_scope(scope, make_tool("scope_only"))
            .ok()
            .unwrap();
        assert!(scoped.get("scope_only").is_some());

        scoped.pop_scope(scope);
        assert!(scoped.get("scope_only").is_none());
    }

    #[test]
    fn pop_scope_non_top_should_return_false() {
        let base = ToolRegistry::new();
        let registry = ScopedToolRegistry::new(base);

        let scope1 = registry.push_scope();
        let scope2 = registry.push_scope();

        assert!(!registry.pop_scope(scope1));
        assert_eq!(registry.scope_depth(), 2);

        assert!(registry.pop_scope(scope2));
        assert_eq!(registry.scope_depth(), 1);
    }

    #[test]
    fn pop_nonexistent_scope_should_return_false() {
        let base = ToolRegistry::new();
        let scoped = ScopedToolRegistry::new(base);
        assert!(!scoped.pop_scope(ScopeId(999)));
    }

    #[test]
    fn multiple_scopes_shadow_correctly() {
        let base = ToolRegistry::new();
        let registry = ScopedToolRegistry::new(base);

        let scope1 = registry.push_scope();
        registry
            .register_in_scope(scope1, make_tool("tool"))
            .ok()
            .unwrap();

        let scope2 = registry.push_scope();
        registry
            .register_in_scope(scope2, make_tool("tool"))
            .ok()
            .unwrap();

        assert_eq!(registry.specs().len(), 1);

        registry.pop_scope(scope2);
        assert!(registry.get("tool").is_some());
        assert_eq!(registry.specs().len(), 1);

        registry.pop_scope(scope1);
        assert!(registry.get("tool").is_none());
    }

    #[test]
    fn register_in_nonexistent_scope_returns_error() {
        let base = ToolRegistry::new();
        let scoped = ScopedToolRegistry::new(base);
        let result = scoped.register_in_scope(ScopeId(999), make_tool("nope"));
        assert!(result.is_err());
    }

    #[test]
    fn register_replaces_within_same_scope() {
        let base = ToolRegistry::new();
        let scoped = ScopedToolRegistry::new(base);

        let scope = scoped.push_scope();
        let old = scoped
            .register_in_scope(scope, make_tool("tool"))
            .ok()
            .unwrap();
        assert!(old.is_none());

        let old = scoped
            .register_in_scope(scope, make_tool("tool"))
            .ok()
            .unwrap();
        assert!(old.is_some());
        assert_eq!(scoped.specs().len(), 1);
    }

    #[tokio::test]
    async fn execute_should_resolve_from_scoped_registry() {
        let base = ToolRegistry::new();
        let scoped = ScopedToolRegistry::new(base);

        let scope = scoped.push_scope();
        scoped
            .register_in_scope(scope, make_tool("tool"))
            .ok()
            .unwrap();

        let call = crate::provider::ToolCall::new("c1", "tool", json!({"x": 1}));
        let output = scoped.execute(&call).await.unwrap();
        assert_eq!(output.value, json!({"x": 1}));
    }

    #[tokio::test]
    async fn execute_unknown_tool_returns_not_found() {
        let base = ToolRegistry::new();
        let scoped = ScopedToolRegistry::new(base);

        let call = crate::provider::ToolCall::new("c1", "missing", json!({}));
        let result = scoped.execute(&call).await;
        assert!(result.is_err());
    }

    #[test]
    fn scope_id_increments_monotonically() {
        let base = ToolRegistry::new();
        let scoped = ScopedToolRegistry::new(base);
        let s1 = scoped.push_scope();
        let s2 = scoped.push_scope();
        assert!(s2.0 > s1.0);
    }

    #[test]
    fn base_mut_allows_modifying_base() {
        let base = ToolRegistry::new();
        let mut scoped = ScopedToolRegistry::new(base);
        scoped.base_mut().register(make_tool("new_base"));
        assert!(scoped.get("new_base").is_some());
    }

    #[test]
    fn clone_captures_scope_snapshot() {
        let base = ToolRegistry::new();
        let scoped = ScopedToolRegistry::new(base);

        let scope = scoped.push_scope();
        scoped
            .register_in_scope(scope, make_tool("tool"))
            .ok()
            .unwrap();

        let cloned = scoped.clone();
        assert!(cloned.get("tool").is_some());

        // Pop original scope — clone unaffected.
        scoped.pop_scope(scope);
        assert!(scoped.get("tool").is_none());
        assert!(cloned.get("tool").is_some());
    }

    #[test]
    fn scope_guard_pops_on_drop() {
        let base = ToolRegistry::new();
        let scoped = Arc::new(ScopedToolRegistry::new(base));

        {
            let _guard = scoped.push_scope_guarded();
            assert_eq!(scoped.scope_depth(), 1);
        }
        // Guard dropped — scope popped.
        assert_eq!(scoped.scope_depth(), 0);
    }

    #[test]
    fn scope_guard_id_matches_pushed_scope() {
        let base = ToolRegistry::new();
        let scoped = Arc::new(ScopedToolRegistry::new(base));

        let guard = scoped.push_scope_guarded();
        let id = guard.id();
        // Verify the scope exists.
        assert_eq!(scoped.scope_depth(), 1);
        // Pop manually to verify the id matches.
        assert!(scoped.pop_scope(id));
    }

    #[test]
    fn scope_guard_into_inner_prevents_drop_cleanup() {
        let base = ToolRegistry::new();
        let scoped = Arc::new(ScopedToolRegistry::new(base));

        let guard = scoped.push_scope_guarded();
        let id = guard.id();

        // Consume guard without dropping it.
        let _ = guard.into_inner();
        // Scope should still be present (drop was skipped).
        assert_eq!(scoped.scope_depth(), 1);
        // Clean up manually.
        scoped.pop_scope(id);
    }

    #[test]
    fn scope_guard_tools_within_scope_are_visible() {
        let base = ToolRegistry::new();
        let scoped = Arc::new(ScopedToolRegistry::new(base));

        let guard = scoped.push_scope_guarded();
        scoped
            .register_in_scope(guard.id(), make_tool("scoped_tool"))
            .ok()
            .unwrap();

        assert!(scoped.get("scoped_tool").is_some());
        assert_eq!(scoped.specs().len(), 1);

        drop(guard);
        // Tool should be gone after guard drop.
        assert!(scoped.get("scoped_tool").is_none());
        assert_eq!(scoped.specs().len(), 0);
    }

    #[test]
    fn push_scope_is_thread_safe() {
        use std::thread;

        let base = ToolRegistry::new();
        let scoped = Arc::new(ScopedToolRegistry::new(base));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let s = Arc::clone(&scoped);
                thread::spawn(move || {
                    let _scope = s.push_scope();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(scoped.scope_depth(), 4);
    }

    #[test]
    fn specs_should_return_sorted_by_name() {
        let base = ToolRegistry::new();
        base.register(make_tool("zebra"));
        base.register(make_tool("alpha"));
        let scoped = ScopedToolRegistry::new(base);

        let scope = scoped.push_scope();
        scoped
            .register_in_scope(scope, make_tool("mike"))
            .ok()
            .unwrap();

        let specs = scoped.specs();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].name, "alpha");
        assert_eq!(specs[1].name, "mike");
        assert_eq!(specs[2].name, "zebra");
    }
}
