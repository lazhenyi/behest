//! [`ManagedRuntime`]: unified container orchestrating
//! [`AgentRuntime`], [`ComponentRegistry`], and
//! [`ShutdownToken`] into a single lifecycle.
//!
//! The managed runtime provides:
//!
//! - **Coordinated lifecycle**: `init_all → start_all → serve → stop_all`
//!   with a single root shutdown token.
//! - **Typed component access**: [`ManagedRuntime::component::<T>`]
//!   downcasts into the underlying [`ComponentRegistry`].
//! - **Aggregated health**: [`ManagedRuntime::health`] combines
//!   component-level and (when available) transport-level probes.
//! - **Hot-reload entry point**: [`ManagedRuntime::reload`] replaces
//!   a running component via the drain-aware protocol.

#![allow(clippy::pedantic)]

use std::sync::Arc;

use thiserror::Error;

use crate::health::HealthStatus;
use crate::runtime::agent::AgentRuntime;
use crate::runtime::component::{AnyComponent, Component};
use crate::runtime::extensions::Extensions;
use crate::runtime::lifecycle::ShutdownToken;
use crate::runtime::registry::{ComponentRegistry, RegistryError, TypedAnyComponent};

#[cfg(feature = "server")]
use crate::transport::hub::TransportHub;

/// Errors from [`ManagedRuntime`] operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ManagedError {
    /// A component was not found in the registry.
    #[error("component `{0}` not found")]
    ComponentNotFound(String),

    /// The component registry returned an error.
    #[error("registry error: {0}")]
    Registry(#[from] RegistryError),

    /// A reload operation failed.
    #[error("reload failed for component `{name}`: {message}")]
    Reload {
        /// The component that failed to reload.
        name: String,
        /// Human-readable error description.
        message: String,
    },

    /// A transport-level error.
    #[cfg(feature = "server")]
    #[error("transport error: {0}")]
    Transport(#[from] crate::transport::TransportError),
}

/// Unified container orchestrating [`AgentRuntime`],
/// [`ComponentRegistry`], an optional [`TransportHub`],
/// and a root [`ShutdownToken`].
///
/// Construct via
/// [`AgentConfigBuilder::build_managed`](crate::config::AgentConfigBuilder::build_managed)
/// or [`ManagedRuntime::new`].
///
/// # Lifecycle
///
/// ```text
///   new()  →  init_all  →  start_all  →  serve  →  signal_shutdown  →  stop_all
/// ```
///
/// # Hot-reload
///
/// [`ManagedRuntime::reload`] walks the drain-aware replace protocol:
/// old instance drains in-flight references, new instance is
/// constructed and started, then swapped in atomically.
pub struct ManagedRuntime {
    runtime: AgentRuntime,
    registry: ComponentRegistry,
    #[cfg(feature = "server")]
    hub: TransportHub,
    shutdown: ShutdownToken,
}

impl ManagedRuntime {
    /// Construct a new managed runtime from its constituent parts.
    ///
    /// The caller is responsible for ensuring that the `extensions`
    /// backing `runtime` and the `registry` are consistent (i.e. the
    /// registry's initialized components have already been applied to
    /// the extensions).
    #[must_use]
    pub fn new(
        runtime: AgentRuntime,
        registry: ComponentRegistry,
        #[cfg(feature = "server")] hub: TransportHub,
        shutdown: ShutdownToken,
    ) -> Self {
        Self {
            runtime,
            registry,
            #[cfg(feature = "server")]
            hub,
            shutdown,
        }
    }

    /// Borrow the underlying [`AgentRuntime`].
    #[must_use]
    pub fn runtime(&self) -> &AgentRuntime {
        &self.runtime
    }

    /// Borrow the [`ComponentRegistry`].
    #[must_use]
    pub fn registry(&self) -> &ComponentRegistry {
        &self.registry
    }

    /// Borrow the [`TransportHub`] (server feature only).
    #[cfg(feature = "server")]
    #[must_use]
    pub fn hub(&self) -> &TransportHub {
        &self.hub
    }

    /// Clone the root [`ShutdownToken`].
    #[must_use]
    pub fn shutdown_token(&self) -> ShutdownToken {
        self.shutdown.clone()
    }

    /// Clone the [`Extensions`] facade from the underlying runtime.
    #[must_use]
    pub fn extensions(&self) -> Arc<Extensions> {
        Arc::clone(self.runtime.extensions())
    }

    /// Look up an initialized component by name, downcasting to a
    /// concrete `Arc<T>`.
    ///
    /// # Errors
    /// - [`ManagedError::ComponentNotFound`] if the name is not
    ///   registered or not yet initialized.
    /// - [`ManagedError::Registry`] on type mismatch.
    pub fn component<T: Component>(&self, name: &str) -> Result<Arc<T>, ManagedError> {
        self.registry
            .get::<T>(name)
            .map_err(|_| ManagedError::ComponentNotFound(name.to_owned()))
    }

    /// Serve until the root shutdown token fires.
    ///
    /// When the `server` feature is enabled, this also starts all
    /// registered transports via [`TransportHub::start_all`].
    /// Transports are stopped via their child shutdown tokens when
    /// the root token fires; components are stopped in reverse
    /// dependency order afterward.
    ///
    /// For a blocking alternative that waits for all transports to
    /// complete, use [`TransportHub::serve_all`].
    ///
    /// # Errors
    /// Returns the first error from transport start or component
    /// stop.
    pub async fn serve(&self) -> Result<(), ManagedError> {
        #[cfg(feature = "server")]
        {
            self.hub.start_all(self.shutdown.child()).await?;
        }

        // Wait for shutdown signal.
        self.shutdown.wait().await;

        // Ordered shutdown: transports first (via child tokens),
        // then components in reverse dependency order.
        self.registry.stop_all().await?;
        Ok(())
    }

    /// Aggregate health of every initialized component.
    ///
    /// When the `server` feature is enabled, transport health is
    /// merged into the same map under the transport's name.
    #[must_use]
    pub async fn health(&self) -> std::collections::HashMap<String, HealthStatus> {
        let mut map = self.registry.health().await;

        #[cfg(feature = "server")]
        {
            let transport_health = self.hub.health().await;
            map.extend(transport_health);
        }

        map
    }

    /// Returns `true` if every component reports healthy.
    #[must_use]
    pub async fn is_healthy(&self) -> bool {
        let map = self.health().await;
        map.values().all(|s| s.is_healthy())
    }

    /// Aggregate all component and transport health into a single
    /// [`HealthStatus`] using worst-case semantics.
    ///
    /// Returns `Unhealthy` if any component is unhealthy, `Degraded`
    /// if any is degraded, `Healthy` otherwise (including empty).
    #[must_use]
    pub async fn overall_health(&self) -> HealthStatus {
        let map = self.health().await;
        HealthStatus::aggregate(&map)
    }

    /// Returns `true` if every component is at least operational
    /// (healthy or degraded). This is the readiness gate suitable
    /// for load-balancer probes.
    #[must_use]
    pub async fn is_ready(&self) -> bool {
        let map = self.health().await;
        map.values().all(|s| s.is_operational())
    }

    /// Build a JSON `/healthz` response body containing the overall
    /// status and per-component breakdown.
    #[must_use]
    pub async fn healthz_json(&self) -> serde_json::Value {
        let map = self.health().await;
        HealthStatus::healthz_response(&map)
    }

    /// Hot-reload a running component by replacing it with a new
    /// instance of the same type.
    ///
    /// The drain-aware protocol:
    ///
    /// 1. Calls `pre_replace_hook` on the old instance.
    /// 2. Starts the new instance. If this fails, the old instance
    ///    remains in place.
    /// 3. Atomically swaps the instance in the registry. Existing
    ///    `Arc<T>` clones held by other tasks keep the old instance
    ///    alive until dropped (natural drain).
    /// 4. Calls `post_replace_hook` on the old instance (best-effort).
    ///
    /// Returns the old `Arc<T>` so the caller can await explicit
    /// cleanup or hold it for drain purposes.
    ///
    /// # Errors
    /// - [`ManagedError::ComponentNotFound`] if the name is not
    ///   registered.
    /// - [`ManagedError::Reload`] if the component is not running,
    ///   or if any phase of the replace protocol fails.
    pub async fn reload<T: Component>(
        &self,
        name: &str,
        new_instance: T,
    ) -> Result<Arc<T>, ManagedError> {
        let boxed: Box<dyn AnyComponent> = Box::new(TypedAnyComponent::new(new_instance));
        let old_any = self
            .registry
            .replace_instance(name, boxed)
            .await
            .map_err(|e| match e {
                RegistryError::NotFound { name: n } => ManagedError::ComponentNotFound(n),
                RegistryError::Reload { name: n, message } => {
                    ManagedError::Reload { name: n, message }
                }
                other => ManagedError::Registry(other),
            })?;

        // Downcast the old instance back to Arc<T>.
        let any_arc = old_any.as_any_arc();
        any_arc.downcast::<T>().map_err(|_| ManagedError::Reload {
            name: name.to_string(),
            message: "old instance type mismatch after swap".to_string(),
        })
    }

    /// Low-level hot-reload using a type-erased replacement.
    ///
    /// This is the untyped counterpart of [`ManagedRuntime::reload`]:
    /// the caller supplies a fully constructed `Box<dyn AnyComponent>`
    /// instead of a typed `T`. Useful when the replacement was built
    /// through a factory or configuration-driven path.
    ///
    /// # Errors
    /// See [`ManagedRuntime::reload`].
    pub async fn reload_raw(
        &self,
        name: &str,
        new_instance: Box<dyn AnyComponent>,
    ) -> Result<Arc<dyn AnyComponent>, ManagedError> {
        self.registry
            .replace_instance(name, new_instance)
            .await
            .map_err(|e| match e {
                RegistryError::NotFound { name: n } => ManagedError::ComponentNotFound(n),
                RegistryError::Reload { name: n, message } => {
                    ManagedError::Reload { name: n, message }
                }
                other => ManagedError::Registry(other),
            })
    }
}

impl std::fmt::Debug for ManagedRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("ManagedRuntime");
        d.field("components", &self.registry.len());
        #[cfg(feature = "server")]
        d.field("transports", &self.hub.len());
        d.finish()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::runtime::component::ComponentContext;
    use crate::runtime::policy::RuntimePolicy;
    use async_trait::async_trait;
    use schemars::JsonSchema;
    use serde::Deserialize;
    use std::time::Duration;

    #[derive(Debug, Clone, Deserialize, JsonSchema)]
    struct TestConfig {
        label: String,
    }

    struct TestComp {
        label: String,
    }

    #[async_trait]
    impl Component for TestComp {
        const NAME: &'static str = "test.managed";
        type Config = TestConfig;
        type Error = std::io::Error;

        async fn init(cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
            Ok(Self {
                label: cfg.label.clone(),
            })
        }
    }

    fn test_runtime() -> ManagedRuntime {
        let exts = Arc::new(Extensions::default());
        let policy = RuntimePolicy::default();
        let runtime = AgentRuntime::new(exts, policy);
        let registry = ComponentRegistry::new();
        let shutdown = ShutdownToken::new();
        ManagedRuntime::new(
            runtime,
            registry,
            #[cfg(feature = "server")]
            TransportHub::new(),
            shutdown,
        )
    }

    #[tokio::test]
    async fn serve_returns_on_shutdown() {
        let managed = test_runtime();
        let token = managed.shutdown_token();
        let handle = tokio::spawn(async move {
            managed.serve().await.expect("serve should succeed");
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        token.signal_shutdown();
        handle.await.expect("task should complete");
    }

    #[tokio::test]
    async fn component_lookup_returns_not_found_for_empty() {
        let managed = test_runtime();
        let result = managed.component::<TestComp>("missing");
        assert!(result.is_err());
        assert!(matches!(result, Err(ManagedError::ComponentNotFound(_))));
    }

    #[tokio::test]
    async fn component_lookup_after_init() {
        let exts = Arc::new(Extensions::default());
        let policy = RuntimePolicy::default();
        let runtime = AgentRuntime::new(exts, policy);
        let registry = ComponentRegistry::new();
        registry
            .register_typed::<TestComp>("test", serde_json::json!({ "label": "hello" }))
            .expect("register should succeed");
        registry.init_all().await.expect("init should succeed");
        registry.start_all().await.expect("start should succeed");

        let managed = ManagedRuntime::new(
            runtime,
            registry,
            #[cfg(feature = "server")]
            TransportHub::new(),
            ShutdownToken::new(),
        );
        let comp: Arc<TestComp> = managed
            .component::<TestComp>("test")
            .expect("lookup should succeed");
        assert_eq!(comp.label, "hello");
    }

    #[tokio::test]
    async fn health_empty_is_healthy() {
        let managed = test_runtime();
        let map = managed.health().await;
        assert!(map.is_empty());
        assert!(managed.is_healthy().await);
    }

    #[tokio::test]
    async fn health_aggregates_registered_components() {
        let exts = Arc::new(Extensions::default());
        let policy = RuntimePolicy::default();
        let runtime = AgentRuntime::new(exts, policy);
        let registry = ComponentRegistry::new();
        registry
            .register_typed::<TestComp>("c1", serde_json::json!({ "label": "a" }))
            .expect("register should succeed");
        registry.init_all().await.expect("init should succeed");
        registry.start_all().await.expect("start should succeed");

        let managed = ManagedRuntime::new(
            runtime,
            registry,
            #[cfg(feature = "server")]
            TransportHub::new(),
            ShutdownToken::new(),
        );
        let map = managed.health().await;
        assert_eq!(map.len(), 1);
        assert!(map.get("c1").map(|s| s.is_healthy()).unwrap_or(false));
    }

    #[test]
    fn debug_format_shows_component_count() {
        let managed = test_runtime();
        let dbg = format!("{managed:?}");
        assert!(dbg.contains("ManagedRuntime"));
        assert!(dbg.contains("components"));
    }

    async fn running_runtime() -> ManagedRuntime {
        let exts = Arc::new(Extensions::default());
        let policy = RuntimePolicy::default();
        let runtime = AgentRuntime::new(exts, policy);
        let registry = ComponentRegistry::new();
        registry
            .register_typed::<TestComp>("c1", serde_json::json!({ "label": "old" }))
            .expect("register should succeed");
        registry.init_all().await.expect("init should succeed");
        registry.start_all().await.expect("start should succeed");
        ManagedRuntime::new(
            runtime,
            registry,
            #[cfg(feature = "server")]
            TransportHub::new(),
            ShutdownToken::new(),
        )
    }

    #[tokio::test]
    async fn reload_swaps_component_and_returns_old() {
        let managed = running_runtime().await;

        // Verify old component is in place.
        let old: Arc<TestComp> = managed
            .component::<TestComp>("c1")
            .expect("lookup should succeed");
        assert_eq!(old.label, "old");

        // Reload with a new instance.
        let returned_old = managed
            .reload::<TestComp>(
                "c1",
                TestComp {
                    label: "new".into(),
                },
            )
            .await
            .expect("reload should succeed");

        // The returned old instance should have the old label.
        assert_eq!(returned_old.label, "old");

        // The registry should now hold the new instance.
        let current: Arc<TestComp> = managed
            .component::<TestComp>("c1")
            .expect("lookup should succeed");
        assert_eq!(current.label, "new");
    }

    #[tokio::test]
    async fn reload_not_found_returns_error() {
        let managed = test_runtime();
        let result = managed
            .reload::<TestComp>("missing", TestComp { label: "x".into() })
            .await;
        assert!(matches!(result, Err(ManagedError::ComponentNotFound(_))));
    }

    #[tokio::test]
    async fn reload_raw_swaps_type_erased_instance() {
        let managed = running_runtime().await;

        let new_instance: Box<dyn AnyComponent> =
            Box::new(TypedAnyComponent::<TestComp>::new(TestComp {
                label: "raw-new".into(),
            }));

        managed
            .reload_raw("c1", new_instance)
            .await
            .expect("reload_raw should succeed");

        let current: Arc<TestComp> = managed
            .component::<TestComp>("c1")
            .expect("lookup should succeed");
        assert_eq!(current.label, "raw-new");
    }
}
