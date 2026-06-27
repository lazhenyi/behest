//! [`ExtensionPoint<T>`]: a typed, name-indexed, hot-swappable collection.
//!
//! In `behest`'s composable model, every pluggable category of runtime
//! element — chat providers, embedding providers, tools, context adapters,
//! stores, publishers, transports — is exposed as an
//! [`ExtensionPoint<T>`]. Operators compose a runtime by registering
//! implementations by name, and can replace any registered instance at
//! runtime.
//!
//! # Design
//!
//! - **Name-indexed**: every entry is keyed by a stable string. The same
//!   name space is shared with the component registry, so config files can
//!   reference extensions by name (e.g. `"primary"`, `"fallback-eu"`).
//! - **Clonable**: the inner state is wrapped in an [`Arc`], so cloning an
//!   `ExtensionPoint` is cheap. A clone observes every registration
//!   performed on the original.
//! - **Hot-swappable**: [`ExtensionPoint::replace`] atomically swaps the
//!   stored `Arc<T>` and returns the previous one. Callers holding the old
//!   `Arc` continue to use it; new `get` calls return the new instance.
//! - **In-use detection**: [`ExtensionPoint::unregister`] refuses to drop
//!   an entry whose strong count is above the registry's reference (one
//!   reference for the storage slot). This catches the common bug of
//!   removing a provider that is still serving a run.
//!
//! # Example
//!
//! ```rust
//! use std::sync::Arc;
//! use behest::runtime::extension::ExtensionPoint;
//!
//! let ep: ExtensionPoint<dyn String> = ExtensionPoint::new();
//! ep.register("greeting", Arc::new("hello".to_string())).unwrap();
//! assert_eq!(ep.get("greeting").map(|s| (*s).clone()), Some("hello".to_string()));
//! ```

#![allow(clippy::pedantic)]
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use thiserror::Error;

use super::replace::{DEFAULT_DRAIN_TIMEOUT, ReplaceError, ReplaceToken};

/// Errors raised by [`ExtensionPoint`] operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ExtensionError {
    /// Tried to register a name that is already in use.
    #[error("extension `{name}` is already registered")]
    AlreadyRegistered {
        /// The conflicting name.
        name: String,
    },
    /// Tried to unregister or replace a name that is not present.
    #[error("extension `{name}` not found")]
    NotFound {
        /// The missing name.
        name: String,
    },
    /// Tried to unregister a name whose strong count is greater than
    /// one (i.e. external callers are still holding the `Arc<T>`).
    #[error("extension `{name}` has {strong_count} live references and cannot be removed")]
    InUse {
        /// The contended name.
        name: String,
        /// Number of strong references observed at the time of the call.
        strong_count: usize,
    },
    /// Internal lock acquisition failed.
    #[error("extension registry lock poisoned")]
    LockPoisoned,
    /// Drain-aware replace protocol error.
    #[error("replace failed: {0}")]
    Replace(#[from] ReplaceError),
}

/// Typed, name-indexed collection of `Arc<T>`.
///
/// Cheap to clone; clones share the same underlying entries.
pub struct ExtensionPoint<T: ?Sized> {
    inner: Arc<ExtensionInner<T>>,
}

struct ExtensionInner<T: ?Sized> {
    entries: RwLock<HashMap<String, Arc<T>>>,
}

impl<T: ?Sized> Default for ExtensionPoint<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ?Sized> Clone for ExtensionPoint<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: ?Sized> ExtensionPoint<T> {
    /// Construct an empty extension point.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ExtensionInner {
                entries: RwLock::new(HashMap::new()),
            }),
        }
    }

    /// Number of registered entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.read().map(|m| m.len()).unwrap_or_default()
    }

    /// Returns `true` if the extension point has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a sorted list of registered names. Sorting makes the result
    /// stable for snapshot tests and log output.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let mut names = self
            .read()
            .map(|m| m.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        names.sort_unstable();
        names
    }

    /// Take a consistent snapshot of `(name, Arc<T>)` pairs.
    #[must_use]
    pub fn snapshot(&self) -> Vec<(String, Arc<T>)> {
        self.read()
            .map(|m| {
                let mut entries: Vec<(String, Arc<T>)> =
                    m.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                entries
            })
            .unwrap_or_default()
    }

    /// Begin a drain-aware replace.
    ///
    /// Returns a [`ReplaceToken`] that the caller must hand to
    /// [`ExtensionPoint::complete_replace`] together with the new
    /// value. The token commits before the previous `Arc<T>` is removed
    /// from the map, which lets the caller cancel a pending replace
    /// before the swap becomes visible.
    ///
    /// This call only checks that `name` is currently registered; it
    /// does not write anything. If the name is removed between
    /// `begin_replace` and `complete_replace`, the latter returns
    /// [`ExtensionError::NotFound`].
    ///
    /// # Errors
    /// - [`ExtensionError::NotFound`] if the name is not registered.
    pub fn begin_replace(&self, name: &str) -> Result<ReplaceToken, ExtensionError> {
        self.begin_replace_with_timeout(name, DEFAULT_DRAIN_TIMEOUT)
    }

    /// Begin a drain-aware replace with a custom drain timeout.
    ///
    /// Behaves like [`ExtensionPoint::begin_replace`] but uses
    /// `timeout` as the maximum wait for in-flight `Arc<T>` holders to
    /// release the previous value. When the deadline elapses, the
    /// previous value is still swapped out (force swap) and a
    /// `tracing::warn!` is emitted.
    ///
    /// # Errors
    /// - [`ExtensionError::NotFound`] if the name is not registered.
    pub fn begin_replace_with_timeout(
        &self,
        name: &str,
        timeout: Duration,
    ) -> Result<ReplaceToken, ExtensionError> {
        let map = self.read()?;
        if !map.contains_key(name) {
            return Err(ExtensionError::NotFound {
                name: name.to_string(),
            });
        }
        Ok(ReplaceToken::new(timeout))
    }

    fn read(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, HashMap<String, Arc<T>>>, ExtensionError> {
        self.inner
            .entries
            .read()
            .map_err(|_| ExtensionError::LockPoisoned)
    }

    fn write(
        &self,
    ) -> Result<std::sync::RwLockWriteGuard<'_, HashMap<String, Arc<T>>>, ExtensionError> {
        self.inner
            .entries
            .write()
            .map_err(|_| ExtensionError::LockPoisoned)
    }
}

impl<T: ?Sized + Send + Sync + 'static> ExtensionPoint<T> {
    /// Register a new entry. Errors if the name is already in use.
    ///
    /// # Errors
    /// - [`ExtensionError::AlreadyRegistered`] if the name is taken.
    /// - [`ExtensionError::LockPoisoned`] if the internal lock was
    ///   poisoned by a panic.
    pub fn register(&self, name: impl Into<String>, value: Arc<T>) -> Result<(), ExtensionError> {
        let name = name.into();
        let mut map = self.write()?;
        if map.contains_key(&name) {
            return Err(ExtensionError::AlreadyRegistered { name });
        }
        map.insert(name, value);
        Ok(())
    }

    /// Register a new entry, replacing any existing one with the same
    /// name. Returns the previous `Arc<T>`, or `None` if there was none.
    pub fn register_or_replace(&self, name: impl Into<String>, value: Arc<T>) -> Option<Arc<T>> {
        let name = name.into();
        self.write().ok().and_then(|mut m| m.insert(name, value))
    }

    /// Remove an entry. Returns the removed `Arc<T>`, or `None` if the
    /// name was not present.
    ///
    /// Refuses to remove an entry whose strong count is greater than one
    /// (i.e. external callers still hold a reference). Use
    /// [`ExtensionPoint::force_unregister`] to bypass the check.
    pub fn unregister(&self, name: &str) -> Result<Option<Arc<T>>, ExtensionError> {
        let map = self.read()?;
        if let Some(existing) = map.get(name) {
            if Arc::strong_count(existing) > 1 {
                return Err(ExtensionError::InUse {
                    name: name.to_string(),
                    strong_count: Arc::strong_count(existing),
                });
            }
        }
        drop(map);
        Ok(self.write()?.remove(name))
    }

    /// Remove an entry without checking for live references. The strong
    /// count of the returned `Arc` after this call is exactly the number
    /// of external holders (potentially zero, if this is the last one).
    pub fn force_unregister(&self, name: &str) -> Option<Arc<T>> {
        self.write().ok().and_then(|mut m| m.remove(name))
    }

    /// Atomically replace an entry. Returns the previous `Arc<T>`, or
    /// [`ExtensionError::NotFound`] if the name was not present.
    ///
    /// After this call, new [`ExtensionPoint::get`] calls return `new`.
    /// Callers that already hold the old `Arc<T>` continue to operate on
    /// the old instance until they drop it.
    ///
    /// # Errors
    /// - [`ExtensionError::NotFound`] if the name was not present.
    pub fn replace(&self, name: &str, new: Arc<T>) -> Result<Arc<T>, ExtensionError> {
        let mut map = self.write()?;
        let previous = map.remove(name).ok_or_else(|| ExtensionError::NotFound {
            name: name.to_string(),
        })?;
        map.insert(name.to_string(), new);
        Ok(previous)
    }

    /// Look up an entry by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<T>> {
        self.read().ok().and_then(|m| m.get(name).cloned())
    }

    /// Look up an entry by name, returning [`ExtensionError::NotFound`]
    /// if missing.
    pub fn get_required(&self, name: &str) -> Result<Arc<T>, ExtensionError> {
        self.get(name).ok_or_else(|| ExtensionError::NotFound {
            name: name.to_string(),
        })
    }

    /// Returns `true` if the given name is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.read().map(|m| m.contains_key(name)).unwrap_or(false)
    }

    /// Strong reference count for the registered entry, or `None` if
    /// the name is not registered. Useful for diagnostics and for
    /// deciding whether an in-use removal is safe.
    #[must_use]
    pub fn strong_count(&self, name: &str) -> Option<usize> {
        self.read()
            .ok()
            .and_then(|m| m.get(name).map(Arc::strong_count))
    }

    /// Complete a drain-aware replace previously initiated by
    /// [`ExtensionPoint::begin_replace`].
    ///
    /// The protocol is:
    ///
    /// 1. The token is committed (`Pending -> Committed`). If the
    ///    token was already finalized, the call returns
    ///    [`ExtensionError::Replace`] without touching the map.
    /// 2. Under the write lock, the previous `Arc<T>` is removed and
    ///    the new `Arc<T>` is inserted atomically. The write lock is
    ///    then released, so concurrent [`ExtensionPoint::get`] calls
    ///    never return `None` during the drain phase.
    /// 3. The function polls `Arc::strong_count` of the previous
    ///    value outside the lock, sleeping up to the token's timeout
    ///    for in-flight holders to drop their references. Once the
    ///    strong count falls to one (only the local `old` reference
    ///    remains) the wait completes early.
    /// 4. When the deadline elapses before drain, the swap still
    ///    happened (force swap) and a `tracing::warn!` is emitted.
    ///
    /// Returns the previous `Arc<T>` so callers can run teardown
    /// hooks on the old instance.
    ///
    /// # Errors
    /// - [`ExtensionError::Replace`] if the token was already
    ///   committed, aborted, or the wait was otherwise rejected.
    /// - [`ExtensionError::NotFound`] if the name was removed between
    ///   `begin_replace` and this call.
    pub async fn complete_replace(
        self: Arc<Self>,
        name: &str,
        new: Arc<T>,
        token: ReplaceToken,
    ) -> Result<Arc<T>, ExtensionError> {
        token.try_commit()?;

        let old = {
            let mut map = self.write()?;
            let old = map.remove(name).ok_or_else(|| ExtensionError::NotFound {
                name: name.to_string(),
            })?;
            map.insert(name.to_string(), new);
            old
        };

        let deadline = Instant::now() + token.timeout();
        loop {
            if Arc::strong_count(&old) <= 1 {
                break;
            }
            let now = Instant::now();
            if now >= deadline {
                tracing::warn!(
                    name = %name,
                    strong_count = Arc::strong_count(&old),
                    timeout = ?token.timeout(),
                    "extension replace drain deadline exceeded; forcing swap"
                );
                break;
            }
            let sleep_for = (deadline - now).min(Duration::from_millis(10));
            tokio::time::sleep(sleep_for).await;
        }

        Ok(old)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::replace::{ReplaceError, ReplaceState};

    #[test]
    fn register_and_get_round_trip() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        ep.register("greeting", Arc::new("hello".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            ep.get("greeting").map(|s| (*s).clone()),
            Some("hello".to_string())
        );
    }

    #[test]
    fn register_rejects_duplicate_names() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        ep.register("a", Arc::new("first".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));
        let err = match ep.register("a", Arc::new("second".to_string())) {
            Ok(_) => panic!("expected Err, got Ok"),
            Err(e) => e,
        };
        assert!(matches!(err, ExtensionError::AlreadyRegistered { .. }));
    }

    #[test]
    fn register_or_replace_swallows_duplicates() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        let prev = ep.register_or_replace("a", Arc::new("first".to_string()));
        assert!(prev.is_none());
        let prev = ep.register_or_replace("a", Arc::new("second".to_string()));
        assert_eq!(prev.map(|s| (*s).clone()), Some("first".to_string()));
        assert_eq!(
            ep.get("a").map(|s| (*s).clone()),
            Some("second".to_string())
        );
    }

    #[test]
    fn replace_returns_previous_value() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        ep.register("a", Arc::new("v1".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));
        let prev = ep
            .replace("a", Arc::new("v2".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((*prev).clone(), "v1");
        assert_eq!(ep.get("a").map(|s| (*s).clone()), Some("v2".to_string()));
    }

    #[test]
    fn replace_missing_returns_not_found() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        let err = match ep.replace("missing", Arc::new("v".to_string())) {
            Ok(_) => panic!("expected Err, got Ok"),
            Err(e) => e,
        };
        assert!(matches!(err, ExtensionError::NotFound { .. }));
    }

    #[test]
    fn unregister_refuses_in_use_entry() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        ep.register("a", Arc::new("v".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));
        let _hold = match ep.get("a") {
            Some(v) => v,
            None => panic!("expected Some"),
        };
        let err = match ep.unregister("a") {
            Ok(_) => panic!("expected Err, got Ok"),
            Err(e) => e,
        };
        assert!(matches!(err, ExtensionError::InUse { .. }));
    }

    #[test]
    fn unregister_drops_when_only_registry_holds() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        ep.register("a", Arc::new("v".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));
        let removed = ep.unregister("a").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(removed.map(|s| (*s).clone()), Some("v".to_string()));
        assert!(ep.get("a").is_none());
    }

    #[test]
    fn force_unregister_drops_even_with_live_references() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        ep.register("a", Arc::new("v".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));
        let hold = match ep.get("a") {
            Some(v) => v,
            None => panic!("expected Some"),
        };
        let removed = ep.force_unregister("a");
        assert!(removed.is_some());
        // External holder still owns the Arc.
        assert_eq!((*hold).clone(), "v");
    }

    #[test]
    fn clone_shares_state() {
        let a: ExtensionPoint<String> = ExtensionPoint::new();
        let b = a.clone();
        a.register("shared", Arc::new("x".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(b.contains("shared"));
    }

    #[test]
    fn snapshot_is_sorted_and_complete() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        ep.register("b", Arc::new("B".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));
        ep.register("a", Arc::new("A".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));
        let snap = ep.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].0, "a");
        assert_eq!(snap[1].0, "b");
    }

    #[test]
    fn strong_count_reflects_holder_count() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        ep.register("a", Arc::new("v".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));
        // 1 = registry
        assert_eq!(ep.strong_count("a"), Some(1));
        let h1 = match ep.get("a") {
            Some(v) => v,
            None => panic!("expected Some"),
        };
        let h2 = match ep.get("a") {
            Some(v) => v,
            None => panic!("expected Some"),
        };
        assert_eq!(ep.strong_count("a"), Some(3));
        drop(h1);
        drop(h2);
        assert_eq!(ep.strong_count("a"), Some(1));
    }

    #[tokio::test]
    async fn replace_drains_in_flight_arcs() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        ep.register("a", Arc::new("v1".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));

        let inflight = ep.get("a").unwrap_or_else(|| panic!("expected Some"));
        assert_eq!(Arc::strong_count(&inflight), 2);

        let ep_arc: Arc<ExtensionPoint<String>> = Arc::new(ep);
        let ep_for_complete = ep_arc.clone();
        let token = ep_arc.begin_replace("a").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(token.state(), ReplaceState::Pending);

        let handle = tokio::spawn(async move {
            ep_for_complete
                .complete_replace("a", Arc::new("v2".to_string()), token)
                .await
        });

        for _ in 0..32 {
            tokio::task::yield_now().await;
        }
        assert!(
            !handle.is_finished(),
            "complete_replace should still be draining while in-flight Arc is held"
        );

        drop(inflight);

        let join_result = tokio::time::timeout(std::time::Duration::from_secs(2), handle)
            .await
            .unwrap_or_else(|_| {
                panic!("complete_replace should not hang after in-flight Arc is dropped")
            });
        let prev = match join_result {
            Ok(r) => r,
            Err(e) => panic!("task should not panic: {e}"),
        };
        let prev = match prev {
            Ok(v) => v,
            Err(e) => panic!("complete_replace should succeed: {e}"),
        };
        assert_eq!(*prev, "v1");
        assert_eq!(
            ep_arc.get("a").map(|s| (*s).clone()),
            Some("v2".to_string())
        );
    }

    #[tokio::test]
    async fn replace_force_swaps_after_deadline() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        ep.register("a", Arc::new("v1".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));

        let inflight = ep.get("a").unwrap_or_else(|| panic!("expected Some"));

        let ep_arc: Arc<ExtensionPoint<String>> = Arc::new(ep);
        let ep_for_complete = ep_arc.clone();
        let token = ep_arc
            .begin_replace_with_timeout("a", std::time::Duration::from_millis(100))
            .unwrap_or_else(|e| panic!("{e}"));

        let handle = tokio::spawn(async move {
            ep_for_complete
                .complete_replace("a", Arc::new("v2".to_string()), token)
                .await
        });

        let join_result = tokio::time::timeout(std::time::Duration::from_secs(2), handle)
            .await
            .unwrap_or_else(|_| {
                panic!("complete_replace should complete after deadline even with in-flight Arc")
            });
        let prev = match join_result {
            Ok(r) => r,
            Err(e) => panic!("task should not panic: {e}"),
        };
        let prev = match prev {
            Ok(v) => v,
            Err(e) => panic!("complete_replace should succeed: {e}"),
        };
        assert_eq!(*prev, "v1");
        assert_eq!(
            ep_arc.get("a").map(|s| (*s).clone()),
            Some("v2".to_string())
        );
        // The pre-replace in-flight Arc still observes v1 after the force swap.
        assert_eq!(*inflight, "v1");
        drop(inflight);
    }

    #[tokio::test]
    async fn begin_replace_rejects_missing_name() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        let err = match ep.begin_replace("missing") {
            Ok(_) => panic!("expected Err, got Ok"),
            Err(e) => e,
        };
        assert!(matches!(err, ExtensionError::NotFound { .. }));
    }

    #[tokio::test]
    async fn complete_replace_rejects_already_committed_token() {
        let ep: ExtensionPoint<String> = ExtensionPoint::new();
        ep.register("a", Arc::new("v1".to_string()))
            .unwrap_or_else(|e| panic!("{e}"));

        let ep_arc: Arc<ExtensionPoint<String>> = Arc::new(ep);
        let token = ep_arc.begin_replace("a").unwrap_or_else(|e| panic!("{e}"));

        // First call commits the token and writes the new value.
        let prev = ep_arc
            .clone()
            .complete_replace("a", Arc::new("v2".to_string()), token.clone())
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(*prev, "v1");
        assert_eq!(
            ep_arc.get("a").map(|s| (*s).clone()),
            Some("v2".to_string())
        );

        // Second call with the same (now Committed) token must be rejected.
        let err = match ep_arc
            .clone()
            .complete_replace("a", Arc::new("v3".to_string()), token)
            .await
        {
            Ok(_) => panic!("expected Err, got Ok"),
            Err(e) => e,
        };
        assert!(matches!(
            err,
            ExtensionError::Replace(ReplaceError::AlreadyCommitted)
        ));
        // And the entry was not overwritten.
        assert_eq!(
            ep_arc.get("a").map(|s| (*s).clone()),
            Some("v2".to_string())
        );
    }
}
