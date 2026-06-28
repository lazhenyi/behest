//! The `Component` trait: lifecycle contract for every pluggable runtime
//! building block.
//!
//! In `behest`, every pluggable element — chat providers, embedding
//! providers, tools, context adapters, stores, event publishers, snapshot
//! stores, RAG adapters, and the transport layer — implements [`Component`].
//! This gives the runtime a uniform shape for: declarative configuration,
//! ordered initialization, lifecycle management, and health aggregation.
//!
//! # Lifecycle
//!
//! ```text
//!    register factory ──► init ──► start ──► [serve] ──► stop
//!                            │
//!                            └──► on init error: instance is dropped
//! ```
//!
//! - [`Component::init`] is the only phase that takes a configuration value.
//!   It must be a pure constructor: no side effects, no network calls.
//! - [`Component::start`] opens connections, spawns background tasks, and
//!   registers with external systems. It must be idempotent: re-starting
//!   a component that is already started should succeed.
//! - [`Component::stop`] is the inverse of `start`. It must drain in-flight
//!   work, close connections, and persist pending state. Idempotent.
//! - [`Component::health`] is a non-mutating probe. It must be cheap
//!   (sub-millisecond) and safe to call from a hot path.
//!
//! # Errors
//!
//! Every phase returns `Result<_, Self::Error>`. The associated error type
//! must be a `std::error::Error` so that the registry can format it
//! uniformly. The trait is deliberately typed: implementations are free to
//! use the most precise error type (e.g. `reqwest::Error` for HTTP
//! providers) without forcing the registry to model every variant.
//!
//! # Hot-plugging
//!
//! Components are not required to be hot-swappable. The trait provides the
//! primitive contract; hot-swap semantics live in the
//! [`ExtensionPoint`](crate::runtime::extension::ExtensionPoint) layer,
//! which is responsible for atomic reference replacement and live-reference
//! detection.

#![allow(clippy::pedantic)]
use std::any::Any;
use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::health::HealthStatus;
use crate::runtime::lifecycle::ShutdownToken;

/// Context handed to [`Component::init`] and propagated through dependent
/// components.
#[derive(Clone)]
pub struct ComponentContext {
    shutdown: ShutdownToken,
}

impl ComponentContext {
    /// Construct a new context with the given shutdown token.
    #[must_use]
    pub fn new(shutdown: ShutdownToken) -> Self {
        Self { shutdown }
    }

    /// Borrow the shutdown token. Components should select a child token
    /// for their own background tasks so that a single component's
    /// shutdown does not implicitly trigger a registry-wide one.
    #[must_use]
    pub fn shutdown(&self) -> ShutdownToken {
        self.shutdown.clone()
    }

    /// Borrow a child shutdown token.
    #[must_use]
    pub fn child_shutdown(&self) -> ShutdownToken {
        self.shutdown.child()
    }
}

impl fmt::Debug for ComponentContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComponentContext").finish_non_exhaustive()
    }
}

/// The core contract every pluggable runtime element implements.
///
/// A `Component` is:
///
/// - **Self-describing**: [`Component::NAME`] is a stable identifier used
///   in config files, log spans, and metrics labels.
/// - **Schema-validated**: [`Component::Config`] implements
///   [`schemars::JsonSchema`], so the registry can emit a JSON schema
///   for IDEs and CLI tools.
/// - **Lifecycle-bounded**: the four-phase `init → start → [serve] → stop`
///   contract lets the registry run a coherent graph of components.
#[async_trait]
pub trait Component: Send + Sync + 'static {
    /// Stable identifier for the component kind (e.g. `"provider.openai"`,
    /// `"store.session.redis"`). Used in configuration and logging.
    const NAME: &'static str;

    /// Configuration shape. Must be deserializable from JSON/YAML/TOML and
    /// must produce a valid JSON Schema for documentation and validation.
    type Config: DeserializeOwned + JsonSchema + Send + Sync + 'static;

    /// Error type for lifecycle phases. Must implement [`std::error::Error`]
    /// so the registry can chain and format errors uniformly.
    type Error: StdError + Send + Sync + 'static;

    /// Construct a component instance from its validated configuration.
    ///
    /// Implementations must not perform IO here; defer network and disk
    /// access to [`Component::start`]. The `ctx` provides a shutdown
    /// token that the component can plumb into background tasks.
    async fn init(cfg: &Self::Config, ctx: &ComponentContext) -> Result<Self, Self::Error>
    where
        Self: Sized;

    /// Begin serving. Default is a no-op. Override to spawn workers, open
    /// connections, or warm caches.
    ///
    /// MUST be idempotent: calling `start` on an already-started component
    /// is a no-op and returns `Ok(())`.
    async fn start(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Stop serving. Default is a no-op. Override to drain queues, close
    /// connections, and persist state.
    ///
    /// MUST be idempotent. After `stop` returns, the component may be
    /// re-`start`-ed.
    async fn stop(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Non-mutating health probe. Default is [`HealthStatus::healthy`].
    /// Override to surface upstream connectivity, queue depth, retry
    /// pressure, etc.
    async fn health(&self) -> HealthStatus {
        HealthStatus::healthy()
    }

    /// Names of components this component depends on. Used by the registry
    /// to build a dependency graph and drive ordered initialization.
    fn depends_on() -> &'static [&'static str] {
        &[]
    }

    /// Called before this component is replaced by a new instance.
    ///
    /// Default is a no-op. Override to reject new traffic, flush
    /// buffers, or signal upstream systems that this instance is
    /// about to be swapped out.
    ///
    /// The component is still running when this hook fires; in-flight
    /// references held by other tasks remain valid.
    async fn pre_replace_hook(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Called after this component has been replaced by a new instance.
    ///
    /// Default is a no-op. Override to clean up resources that were
    /// not released during `stop`, or to notify upstream systems that
    /// the replacement is complete.
    ///
    /// At this point, new traffic is routed to the replacement
    /// instance. The old instance may still be held by tasks that
    /// obtained an `Arc` before the swap.
    async fn post_replace_hook(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Object-safe view of a [`Component`] instance, used by the registry to
/// store and drive components of heterogeneous types uniformly.
pub trait AnyComponent: Send + Sync + 'static {
    /// Stable name of the underlying [`Component`] kind.
    fn name(&self) -> &'static str;

    /// Underlying concrete type, used by `ComponentRegistry::get` to
    /// downcast back to a concrete `Arc<C>`. Returns a type-erased
    /// `Arc<dyn Any>` clone of the typed instance.
    fn as_any_arc(&self) -> Arc<dyn Any + Send + Sync>;

    /// Begin serving. See [`Component::start`].
    fn start(&self) -> futures_util::future::BoxFuture<'_, Result<(), AnyComponentError>>;

    /// Stop serving. See [`Component::stop`].
    fn stop(&self) -> futures_util::future::BoxFuture<'_, Result<(), AnyComponentError>>;

    /// Health probe. See [`Component::health`].
    fn health(&self) -> futures_util::future::BoxFuture<'_, HealthStatus>;

    /// Called before this component is replaced. See
    /// [`Component::pre_replace_hook`].
    fn pre_replace(&self) -> futures_util::future::BoxFuture<'_, Result<(), AnyComponentError>>;

    /// Called after this component has been replaced. See
    /// [`Component::post_replace_hook`].
    fn post_replace(&self) -> futures_util::future::BoxFuture<'_, Result<(), AnyComponentError>>;
}

/// Type-erased error from a boxed [`AnyComponent`]. The original typed
/// error is preserved as a string for log purposes; if callers need the
/// typed error they should downcast via [`AnyComponent::as_any_arc`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AnyComponentError {
    /// The underlying component's typed error, formatted as a string.
    #[error("component `{name}` failed: {message}")]
    Component {
        /// Name of the failing component.
        name: String,
        /// Human-readable message; the original typed error is preserved
        /// for diagnostic purposes only.
        message: String,
    },
    /// The component is registered but has not been initialized.
    #[error("component `{0}` is not initialized")]
    NotInitialized(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct DummyConfig {
        label: String,
    }

    struct DummyComponent {
        label: String,
    }

    #[async_trait]
    impl Component for DummyComponent {
        const NAME: &'static str = "test.dummy";
        type Config = DummyConfig;
        type Error = std::io::Error;

        async fn init(cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
            Ok(Self {
                label: cfg.label.clone(),
            })
        }
    }

    #[tokio::test]
    async fn component_init_constructs_with_config() {
        let shutdown = ShutdownToken::new();
        let ctx = ComponentContext::new(shutdown);
        let cfg = DummyConfig {
            label: "alpha".into(),
        };
        let c = DummyComponent::init(&cfg, &ctx).await.unwrap_or_else(|e| {
            panic!("init failed: {e}");
        });
        assert_eq!(c.label, "alpha");
    }

    #[tokio::test]
    async fn default_lifecycle_is_noop() {
        let shutdown = ShutdownToken::new();
        let ctx = ComponentContext::new(shutdown);
        let c = DummyComponent::init(&DummyConfig { label: "x".into() }, &ctx)
            .await
            .unwrap_or_else(|e| panic!("{e}"));
        // start / stop / health all use defaults; must not fail.
        let _ = c.start().await;
        let _ = c.stop().await;
        assert!(c.health().await.is_healthy());
    }
}
