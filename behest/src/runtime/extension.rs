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
//! let ep: ExtensionPoint<String> = ExtensionPoint::new();
//! ep.register("greeting", Arc::new("hello".to_string())).unwrap();
//! assert_eq!(ep.get("greeting").map(|s| (*s).clone()), Some("hello".to_string()));
//! ```

#![allow(clippy::pedantic)]
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use thiserror::Error;

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
    /// (i.e. external callers still hold a reference).
    pub fn unregister(&self, name: &str) -> Result<Option<Arc<T>>, ExtensionError> {
        let map = self.read()?;
        if let Some(existing) = map.get(name)
            && Arc::strong_count(existing) > 1
        {
            return Err(ExtensionError::InUse {
                name: name.to_string(),
                strong_count: Arc::strong_count(existing),
            });
        }
        drop(map);
        Ok(self.write()?.remove(name))
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
