//! Factory registry for runtime components.
//!
//! Maps `kind` strings (e.g. `"provider.openai"`, `"store.session.memory"`)
//! to factory invokers that deserialize a JSON config and produce a
//! [`Box<dyn AnyComponent>`](crate::runtime::AnyComponent).

use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

use crate::runtime::component::{AnyComponent, ComponentContext};

/// Errors from factory creation or registry operations.
#[derive(Debug, Error)]
pub enum FactoryError {
    /// No factory registered for the given kind.
    #[error("unknown component kind: {0}")]
    UnknownKind(String),

    /// Config deserialization failed for the matched factory.
    #[error("invalid config for kind {kind}: {source}")]
    InvalidConfig {
        /// The component kind that failed to deserialize.
        kind: String,
        /// The underlying deserialization error.
        source: serde_json::Error,
    },

    /// Factory implementation returned an error.
    #[error("factory for kind {0} failed: {1}")]
    FactoryFailed(String, String),
}

/// A factory invocable — given a JSON config and a [`ComponentContext`],
/// produces a [`Box<dyn AnyComponent>`].
pub trait FactoryInvoker: Send + Sync {
    /// Create a component from the given JSON config.
    ///
    /// # Errors
    ///
    /// Returns [`FactoryError::InvalidConfig`] when deserialization fails,
    /// or [`FactoryError::FactoryFailed`] when construction fails.
    fn invoke(
        &self,
        config: Value,
        ctx: ComponentContext,
    ) -> Result<Box<dyn AnyComponent>, FactoryError>;
}

/// Blanket implementation so that any matching closure can be used as a
/// [`FactoryInvoker`] directly.
impl<F> FactoryInvoker for F
where
    F: Fn(Value, ComponentContext) -> Result<Box<dyn AnyComponent>, FactoryError>
        + Send
        + Sync
        + 'static,
{
    fn invoke(
        &self,
        config: Value,
        ctx: ComponentContext,
    ) -> Result<Box<dyn AnyComponent>, FactoryError> {
        (self)(config, ctx)
    }
}

/// Thread-safe factory function pointer — an [`Arc`] around a boxed closure.
///
/// This type alias is provided so that downstream code (e.g. Task 7's
/// provider/store adapters) can store and pass factory functions easily.
pub type FactoryFn = Arc<
    dyn Fn(Value, ComponentContext) -> Result<Box<dyn AnyComponent>, FactoryError> + Send + Sync,
>;

/// A registry that maps `kind` strings (e.g. `"provider.openai"`) to
/// [`FactoryInvoker`] instances.
///
/// Invoking an unknown kind returns [`FactoryError::UnknownKind`]:
///
/// ```
/// # use behest::runtime::factory_registry::{FactoryError, FactoryRegistry};
/// # use behest::runtime::ComponentContext;
/// # use behest::runtime::lifecycle::ShutdownToken;
/// let reg = FactoryRegistry::new();
/// let ctx = ComponentContext::new(ShutdownToken::new());
/// let result = reg.invoke("nope", serde_json::json!({}), &ctx);
/// assert!(result.is_err());
/// assert!(matches!(result, Err(FactoryError::UnknownKind(_))));
/// ```
pub struct FactoryRegistry {
    by_kind: HashMap<&'static str, Arc<dyn FactoryInvoker>>,
}

impl Default for FactoryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FactoryRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            by_kind: HashMap::new(),
        }
    }

    /// Registers a factory invoker under the given `kind`.
    ///
    /// Returns `self` for chaining.
    #[must_use]
    pub fn register(mut self, kind: &'static str, invoker: impl FactoryInvoker + 'static) -> Self {
        self.by_kind.insert(kind, Arc::new(invoker));
        self
    }

    /// Returns `true` if a factory is registered for `kind`.
    #[must_use]
    pub fn contains(&self, kind: &str) -> bool {
        self.by_kind.contains_key(kind)
    }

    /// Iterates over all registered kind strings.
    pub fn kinds(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.by_kind.keys().copied()
    }

    /// Invokes the factory registered for `kind` with the given JSON config.
    ///
    /// # Errors
    ///
    /// Returns [`FactoryError::UnknownKind`] when no factory is registered,
    /// or the factory's own error on failure.
    pub fn invoke(
        &self,
        kind: &str,
        cfg: Value,
        ctx: &ComponentContext,
    ) -> Result<Box<dyn AnyComponent>, FactoryError> {
        let invoker = self
            .by_kind
            .get(kind)
            .ok_or_else(|| FactoryError::UnknownKind(kind.to_owned()))?;
        let ctx = ctx.clone();
        invoker.invoke(cfg, ctx)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, dead_code)]
    use super::*;
    use crate::runtime::component::ComponentContext;
    use crate::runtime::lifecycle::ShutdownToken;
    use crate::runtime::registry::TypedAnyComponent;
    use async_trait::async_trait;
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, JsonSchema)]
    struct DummyCfg {
        v: u32,
    }

    struct DummyComp {
        v: u32,
    }

    #[async_trait]
    impl crate::runtime::Component for DummyComp {
        const NAME: &'static str = "dummy";
        type Config = DummyCfg;
        type Error = std::io::Error;

        async fn init(cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
            Ok(Self { v: cfg.v })
        }

        async fn start(&self) -> Result<(), Self::Error> {
            Ok(())
        }

        async fn stop(&self) -> Result<(), Self::Error> {
            Ok(())
        }

        async fn health(&self) -> crate::health::HealthStatus {
            crate::health::HealthStatus::healthy()
        }
    }

    #[test]
    fn registry_invokes_known_kind() {
        let reg =
            FactoryRegistry::new().register("test.dummy", |cfg: Value, _ctx: ComponentContext| {
                let v: DummyCfg =
                    serde_json::from_value(cfg).map_err(|e| FactoryError::InvalidConfig {
                        kind: "test.dummy".to_owned(),
                        source: e,
                    })?;
                Ok(Box::new(TypedAnyComponent::new(DummyComp { v: v.v })) as Box<dyn AnyComponent>)
            });
        let ctx = ComponentContext::new(ShutdownToken::new());
        let comp = reg
            .invoke("test.dummy", serde_json::json!({ "v": 42 }), &ctx)
            .expect("invoke should succeed");
        assert_eq!(comp.name(), "dummy");
    }

    #[test]
    fn registry_rejects_unknown_kind() {
        let reg = FactoryRegistry::new();
        let ctx = ComponentContext::new(ShutdownToken::new());
        let result = reg.invoke("nope", serde_json::json!({}), &ctx);
        assert!(result.is_err());
        assert!(matches!(result, Err(FactoryError::UnknownKind(_))));
    }
}
