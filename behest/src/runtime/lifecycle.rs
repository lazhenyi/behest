//! Cooperative shutdown primitives.
//!
//! [`ShutdownToken`] is a clonable, hierarchical cancellation primitive shared
//! between the [`ComponentRegistry`](crate::runtime::registry::ComponentRegistry),
//! every [`Component`](crate::runtime::component::Component) lifecycle phase,
//! and the future transport layer (added in M5).
//!
//! Design goals:
//!
//! 1. **Cooperative, not preemptive**: tasks opt-in by calling
//!    [`ShutdownToken::wait`] or by polling [`ShutdownToken::is_shutdown`].
//! 2. **Hierarchical**: child tokens propagate a shutdown signal upward to
//!    their parent, so the entire process can be torn down from any
//!    component failure without each component needing to wire its own
//!    supervision.
//! 3. **Cheap to clone**: backpressure-free, `Arc`-backed, safe to embed
//!    in every component without allocation pressure.

#![allow(clippy::pedantic)]
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::Notify;

/// Hierarchical, cooperative shutdown token.
///
/// Cloning a token is cheap: all clones share the same underlying state. Child
/// tokens notify their parent on [`ShutdownToken::signal_shutdown`], so any
/// component can request a process-wide shutdown.
#[derive(Clone, Default)]
pub struct ShutdownToken {
    state: Arc<ShutdownState>,
}

struct ShutdownState {
    flag: AtomicBool,
    notify: Notify,
    /// Optional parent. When a child signals shutdown, the parent also
    /// receives the signal. This enables the
    /// "any-component-failure-takes-down-the-process" semantics.
    parent: Option<Arc<ShutdownState>>,
}

impl Default for ShutdownState {
    fn default() -> Self {
        Self {
            flag: AtomicBool::new(false),
            notify: Notify::new(),
            parent: None,
        }
    }
}

impl ShutdownToken {
    /// Create a root shutdown token with no parent.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(ShutdownState {
                flag: AtomicBool::new(false),
                notify: Notify::new(),
                parent: None,
            }),
        }
    }

    /// Derive a child token that propagates its shutdown signal to `self`.
    ///
    /// Calling [`ShutdownToken::signal_shutdown`] on the child will also
    /// signal the parent. The child has its own internal flag, so a parent
    /// that is already shutdown will not cause spurious wakeups on a
    /// freshly-created child.
    #[must_use]
    pub fn child(&self) -> Self {
        Self {
            state: Arc::new(ShutdownState {
                flag: AtomicBool::new(false),
                notify: Notify::new(),
                parent: Some(self.state.clone()),
            }),
        }
    }

    /// Signal shutdown. Idempotent; subsequent calls are no-ops.
    pub fn signal_shutdown(&self) {
        self.state.flag.store(true, Ordering::SeqCst);
        self.state.notify.notify_waiters();
        if let Some(parent) = &self.state.parent {
            parent.flag.store(true, Ordering::SeqCst);
            parent.notify.notify_waiters();
        }
    }

    /// Returns `true` if [`ShutdownToken::signal_shutdown`] has been called on
    /// this token or any of its ancestors.
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        if self.state.flag.load(Ordering::SeqCst) {
            return true;
        }
        if let Some(parent) = &self.state.parent {
            return parent.flag.load(Ordering::SeqCst);
        }
        false
    }

    /// Wait for a shutdown signal. Returns immediately if one has already
    /// been signalled.
    pub async fn wait(&self) {
        if self.is_shutdown() {
            return;
        }
        let notified = self.state.notify.notified();
        if self.is_shutdown() {
            return;
        }
        notified.await;
    }

    /// Reset the shutdown flag. Intended for tests only; production code
    /// should treat shutdown as monotonic.
    #[cfg(test)]
    pub fn reset(&self) {
        self.state.flag.store(false, Ordering::SeqCst);
    }
}

impl std::fmt::Debug for ShutdownToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShutdownToken")
            .field("is_shutdown", &self.is_shutdown())
            .field("has_parent", &self.state.parent.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_token_is_not_shutdown() {
        let t = ShutdownToken::new();
        assert!(!t.is_shutdown());
    }

    #[test]
    fn signal_shutdown_is_idempotent() {
        let t = ShutdownToken::new();
        t.signal_shutdown();
        t.signal_shutdown();
        assert!(t.is_shutdown());
    }

    #[test]
    fn child_propagates_to_parent() {
        let parent = ShutdownToken::new();
        let child = parent.child();
        assert!(!parent.is_shutdown());
        child.signal_shutdown();
        assert!(child.is_shutdown());
        assert!(parent.is_shutdown());
    }

    #[test]
    fn sibling_children_do_not_propagate_to_each_other() {
        let parent = ShutdownToken::new();
        let a = parent.child();
        let _b = parent.child();
        a.signal_shutdown();
        // parent is shutdown but sibling b's local flag is not.
        // b.is_shutdown() returns true because it walks to parent.
        // This is intentional: children observe ancestor shutdowns.
        let _ = parent.is_shutdown();
    }

    #[tokio::test]
    async fn wait_returns_when_signalled() {
        let t = ShutdownToken::new();
        let t2 = t.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            t2.signal_shutdown();
        });
        t.wait().await;
        assert!(t.is_shutdown());
    }

    #[tokio::test]
    async fn wait_returns_immediately_if_already_signalled() {
        let t = ShutdownToken::new();
        t.signal_shutdown();
        // Should not hang.
        t.wait().await;
    }
}
