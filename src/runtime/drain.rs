//! Drain guard for graceful reference draining during hot-swap.
//!
//! A [`DrainGuard`] wraps an `Arc<T>` and provides utilities to wait
//! for all other holders to drop their references before proceeding
//! with cleanup or final stop.
//!
//! # Example
//!
//! ```rust,ignore
//! use behest::runtime::drain::DrainGuard;
//! use std::time::Duration;
//!
//! // After replace_instance returns the old component:
//! let old = registry.replace_instance("db", new_instance).await?;
//! let guard = DrainGuard::new(old);
//!
//! // Wait for in-flight references to drain (up to 30s).
//! match guard.wait_for_drain(Duration::from_secs(30)).await {
//!     Ok(()) => { /* all references dropped */ }
//!     Err(remaining) => {
//!         tracing::warn!(remaining, "drain timeout; some references still held");
//!     }
//! }
//! ```

#![allow(clippy::pedantic)]

use std::sync::Arc;
use std::time::Duration;

/// Result of a drain operation: either all references were dropped,
/// or the timeout expired with some references still outstanding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainResult {
    /// All references were successfully drained.
    Drained,
    /// The timeout expired with the given number of outstanding
    /// references (excluding the guard itself).
    Timeout {
        /// Number of outstanding references that were not dropped
        /// before the timeout.
        remaining: usize,
    },
}

impl DrainResult {
    /// Returns `true` if all references were drained.
    #[must_use]
    pub fn is_drained(&self) -> bool {
        matches!(self, DrainResult::Drained)
    }
}

/// A guard wrapping an `Arc<T>` that provides drain-wait utilities.
///
/// After a hot-swap replaces a component in the registry, the caller
/// receives the old `Arc<dyn AnyComponent>`. Wrapping it in a
/// `DrainGuard` lets the caller explicitly wait for all other
/// holders (e.g. in-flight requests, background tasks) to drop their
/// clones before performing final cleanup.
///
/// The guard itself holds one strong reference; `wait_for_drain`
/// considers the drain complete when `Arc::strong_count` equals 1
/// (only the guard's reference remains).
pub struct DrainGuard<T: ?Sized + Send + Sync> {
    inner: Arc<T>,
}

impl<T: ?Sized + Send + Sync> DrainGuard<T> {
    /// Wrap an existing `Arc<T>` in a drain guard.
    #[must_use]
    pub fn new(inner: Arc<T>) -> Self {
        Self { inner }
    }

    /// Returns the current number of outstanding references *beyond*
    /// this guard. A return value of 0 means only the guard holds a
    /// reference.
    #[must_use]
    pub fn outstanding_refs(&self) -> usize {
        Arc::strong_count(&self.inner).saturating_sub(1)
    }

    /// Wait until all other holders drop their `Arc<T>` clones, or
    /// the timeout expires.
    ///
    /// Polls at 50 ms intervals. Returns [`DrainResult::Drained`]
    /// when the strong count reaches 1, or
    /// [`DrainResult::Timeout`] if the deadline passes.
    pub async fn wait_for_drain(&self, timeout: Duration) -> DrainResult {
        let deadline = tokio::time::Instant::now() + timeout;
        let poll_interval = Duration::from_millis(50);

        loop {
            if self.outstanding_refs() == 0 {
                return DrainResult::Drained;
            }
            if tokio::time::Instant::now() >= deadline {
                return DrainResult::Timeout {
                    remaining: self.outstanding_refs(),
                };
            }
            tokio::time::sleep(poll_interval).await;
        }
    }

    /// Borrow the inner `Arc<T>`.
    #[must_use]
    pub fn arc(&self) -> &Arc<T> {
        &self.inner
    }

    /// Consume the guard and return the inner `Arc<T>`.
    #[must_use]
    pub fn into_inner(self) -> Arc<T> {
        self.inner
    }
}

impl<T: ?Sized + Send + Sync + std::fmt::Debug> std::fmt::Debug for DrainGuard<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrainGuard")
            .field("outstanding_refs", &self.outstanding_refs())
            .field("inner", &self.inner)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn drain_completes_when_only_guard_holds_ref() {
        let arc = Arc::new(42_u32);
        let guard = DrainGuard::new(arc);
        // Only the guard holds a reference now.
        let result = guard.wait_for_drain(Duration::from_millis(100)).await;
        assert!(result.is_drained());
    }

    #[tokio::test]
    async fn drain_waits_for_other_holders() {
        let arc = Arc::new(42_u32);
        let guard = DrainGuard::new(Arc::clone(&arc));
        let extra = Arc::clone(&arc);

        // Drop the original; only guard + extra remain.
        drop(arc);
        assert_eq!(guard.outstanding_refs(), 1);

        // Spawn a task that drops the extra ref after 50ms.
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(extra);
        });

        let result = guard.wait_for_drain(Duration::from_secs(2)).await;
        assert!(result.is_drained());
    }

    #[tokio::test]
    async fn drain_times_out_when_refs_held() {
        let arc = Arc::new(42_u32);
        let guard = DrainGuard::new(Arc::clone(&arc));
        let _extra = Arc::clone(&arc);

        let result = guard.wait_for_drain(Duration::from_millis(100)).await;
        assert!(!result.is_drained());
        assert_eq!(result, DrainResult::Timeout { remaining: 2 });
    }

    #[tokio::test]
    async fn into_inner_returns_arc() {
        let arc = Arc::new("hello");
        let guard = DrainGuard::new(Arc::clone(&arc));
        let returned = guard.into_inner();
        // Both arcs should point to the same data.
        assert_eq!(*returned, "hello");
    }
}
