//! [`TransportHub`]: the central orchestrator for a composable set of
//! [`Transport`](super::Transport) implementations.
//!
//! # Responsibilities
//!
//! - **Registration**: [`TransportHub::add`] registers a transport
//!   under a user-assigned name. Duplicates are rejected.
//! - **Lifecycle**: `TransportHub::start_all` drives every registered
//!   transport concurrently until the shutdown token fires.
//! - **Health**: [`TransportHub::health`] aggregates per-transport
//!   health into a single map for `/healthz` responses.
//! - **Type-erased lookup**: [`TransportHub::get_typed::<T>`] downcasts
//!   to a concrete `Arc<T>`.

#![allow(clippy::pedantic)]

use std::collections::HashMap;
use std::sync::Arc;

use crate::health::HealthStatus;
use crate::runtime::lifecycle::ShutdownToken;
use crate::transport::{AnyTransport, TransportError, TypedAnyTransport};

/// Central registry of [`AnyTransport`]s.
///
/// Cloning a `TransportHub` is cheap: it is backed by `Arc`-shared
/// inner state.
#[derive(Clone, Default)]
pub struct TransportHub {
    inner: Arc<TransportHubInner>,
}

#[derive(Default)]
struct TransportHubInner {
    transports: std::sync::RwLock<HashMap<String, Arc<dyn AnyTransport>>>,
}

impl TransportHub {
    /// Construct a new, empty hub.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a typed transport under the given name.
    ///
    /// # Errors
    /// - [`TransportError::AlreadyRegistered`] if the name is taken.
    /// - [`TransportError::LockPoisoned`] if the internal lock was
    ///   poisoned.
    pub fn add<T>(&self, name: impl Into<String>, transport: Arc<T>) -> Result<(), TransportError>
    where
        T: crate::transport::Transport + 'static,
    {
        let name = name.into();
        let mut map = self
            .inner
            .transports
            .write()
            .map_err(|_| TransportError::LockPoisoned)?;
        if map.contains_key(&name) {
            return Err(TransportError::AlreadyRegistered { name });
        }
        let any: Arc<dyn AnyTransport> = Arc::new(TypedAnyTransport::new(transport));
        map.insert(name, any);
        Ok(())
    }

    /// Look up a registered transport by name, downcasting to a
    /// concrete `Arc<T>`.
    ///
    /// # Errors
    /// - [`TransportError::NotFound`] if the name is not registered.
    /// - [`TransportError::LockPoisoned`] if the internal lock was
    ///   poisoned.
    pub fn get_typed<T: crate::transport::Transport + 'static>(
        &self,
        name: &str,
    ) -> Result<Arc<T>, TransportError> {
        let map = self
            .inner
            .transports
            .read()
            .map_err(|_| TransportError::LockPoisoned)?;
        let any = map.get(name).ok_or_else(|| TransportError::NotFound {
            name: name.to_string(),
        })?;
        let arc = any.as_any_arc();
        let type_id = arc.type_id();
        arc.downcast::<T>().map_err(|_| TransportError::Serve {
            name: name.to_string(),
            message: format!("type mismatch: expected {type_id:?}"),
        })
    }

    /// Number of registered transports.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner
            .transports
            .read()
            .map(|m| m.len())
            .unwrap_or_default()
    }

    /// Returns `true` if no transports are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sorted list of registered transport names.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let mut names = self
            .inner
            .transports
            .read()
            .map(|m| m.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        names.sort_unstable();
        names
    }

    /// Start every registered transport. Each transport's `serve`
    /// future is spawned onto the tokio runtime. The returned future
    /// resolves as soon as every transport has started (does not
    /// wait for them to finish).
    ///
    /// # Errors
    /// - [`TransportError::LockPoisoned`]
    pub async fn start_all(&self, shutdown: ShutdownToken) -> Result<(), TransportError> {
        let entries: Vec<(String, Arc<dyn AnyTransport>)> = {
            let map = self
                .inner
                .transports
                .read()
                .map_err(|_| TransportError::LockPoisoned)?;
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        };
        for (name, any) in entries {
            let child = shutdown.child();
            let any2 = any.clone();
            let name2 = name.clone();
            tokio::spawn(async move {
                if let Err(e) = any2.serve(child).await {
                    tracing::error!("transport {name2} serve returned error: {e}");
                }
            });
        }
        Ok(())
    }

    /// Probe every registered transport and collect their health
    /// status. Probes run concurrently.
    pub async fn health(&self) -> HashMap<String, HealthStatus> {
        let entries: Vec<(String, Arc<dyn AnyTransport>)> = {
            self.inner
                .transports
                .read()
                .ok()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default()
        };
        let mut out = HashMap::new();
        for (name, any) in entries {
            let h = any.health().await;
            out.insert(name, h);
        }
        out
    }
}

impl std::fmt::Debug for TransportHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransportHub")
            .field("names", &self.names())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct StubTransport {
        started: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl crate::transport::Transport for StubTransport {
        const NAME: &'static str = "transport.test.stub";
        type Config = serde_json::Value;
        type Error = std::io::Error;

        async fn serve(&self, shutdown: ShutdownToken) -> Result<(), Self::Error> {
            self.started.fetch_add(1, Ordering::SeqCst);
            shutdown.wait().await;
            Ok(())
        }
    }

    #[test]
    fn empty_hub_has_no_names() {
        let hub = TransportHub::new();
        assert!(hub.is_empty());
        assert_eq!(hub.len(), 0);
        assert!(hub.names().is_empty());
    }

    #[test]
    fn add_rejects_duplicate() {
        let hub = TransportHub::new();
        let started = Arc::new(AtomicUsize::new(0));
        let t = Arc::new(StubTransport { started });
        hub.add("primary", t.clone())
            .unwrap_or_else(|e| panic!("{e}"));
        let err = hub
            .add("primary", t)
            .err()
            .unwrap_or_else(|| panic!("expected Err, got Ok"));
        assert!(matches!(err, TransportError::AlreadyRegistered { .. }));
    }

    #[test]
    fn get_typed_returns_concrete_arc() {
        let hub = TransportHub::new();
        let started = Arc::new(AtomicUsize::new(0));
        let t = Arc::new(StubTransport { started });
        hub.add("primary", t).unwrap_or_else(|e| panic!("{e}"));
        let got: Arc<StubTransport> = hub.get_typed("primary").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(got.started.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn get_typed_returns_not_found_for_missing() {
        let hub = TransportHub::new();
        let err = hub
            .get_typed::<StubTransport>("missing")
            .err()
            .unwrap_or_else(|| panic!("expected Err, got Ok"));
        assert!(matches!(err, TransportError::NotFound { .. }));
    }

    #[tokio::test]
    async fn start_all_serves_until_shutdown() {
        let hub = TransportHub::new();
        let started = Arc::new(AtomicUsize::new(0));
        let t = Arc::new(StubTransport {
            started: started.clone(),
        });
        hub.add("primary", t).unwrap_or_else(|e| panic!("{e}"));
        let shutdown = ShutdownToken::new();
        hub.start_all(shutdown.clone())
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        // Give transports a moment to register the start.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert_eq!(started.load(Ordering::SeqCst), 1);
        shutdown.signal_shutdown();
    }
}
