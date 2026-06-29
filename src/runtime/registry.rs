//! [`ComponentRegistry`]: ordered, dependency-aware, type-erased registry
//! for every [`Component`] in the runtime.
//!
//! # Responsibilities
//!
//! - **Registration**: [`ComponentRegistry::register_factory`] queues a
//!   component kind for initialization.
//! - **Dependency resolution**: factory metadata lists `depends_on`
//!   names; the registry topologically sorts the graph and refuses to
//!   proceed if a cycle is present.
//! - **Type erasure**: components are stored as `Arc<dyn AnyComponent>`
//!   so the registry can hold heterogeneous instances. Typed access uses
//!   [`ComponentRegistry::get`] to downcast back to a concrete
//!   `Arc<C>`.
//! - **Lifecycle orchestration**: [`ComponentRegistry::init_all`],
//!   [`ComponentRegistry::start_all`], [`ComponentRegistry::stop_all`]
//!   drive the four-phase component lifecycle in dependency order.
//! - **Health aggregation**: [`ComponentRegistry::health`] fans out
//!   `AnyComponent::health` calls concurrently and collects the results
//!   into a single map for `/healthz` responses.
//!
//! # Lifecycle ordering
//!
//! ```text
//!   register(...)        # queues factory
//!       ↓
//!   init_all()           # topo order, dep-first
//!       ↓
//!   start_all()          # topo order
//!   ... serve ...
//!       ↓
//!   stop_all()           # reverse topo order
//! ```
//!
//! On init error, the registry leaves the offending component in a
//! [`ComponentState::Failed`] state and continues to attempt
//! initialization of any components that do not depend on it. This
//! maximizes operator visibility: a single broken optional backend
//! should not prevent the rest of the system from coming up.

#![allow(clippy::pedantic)]
use std::any::{Any, TypeId};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use futures_util::future::BoxFuture;
use thiserror::Error;

use crate::health::HealthStatus;
use crate::runtime::component::{AnyComponent, AnyComponentError, Component, ComponentContext};
use crate::runtime::lifecycle::ShutdownToken;

/// Errors raised by [`ComponentRegistry`] operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RegistryError {
    /// Tried to register a name that is already in use.
    #[error("component `{name}` is already registered")]
    AlreadyRegistered {
        /// The conflicting name.
        name: String,
    },
    /// Tried to unregister a name that is not present.
    #[error("component `{name}` is not registered")]
    NotFound {
        /// The missing name.
        name: String,
    },
    /// A factory listed in `depends_on` was not registered.
    #[error("component `{name}` depends on missing component `{dep}`")]
    MissingDependency {
        /// The component declaring the dependency.
        name: String,
        /// The missing dependency name.
        dep: String,
    },
    /// The dependency graph contains a cycle. The first cycle found is
    /// reported as a list of node names.
    #[error("cycle detected in component dependencies: {cycle:?}")]
    Cycle {
        /// The cycle, in the order it was discovered.
        cycle: Vec<String>,
    },
    /// Init phase failed for a component.
    #[error("init failed for component `{name}`: {message}")]
    Init {
        /// Name of the failing component.
        name: String,
        /// Human-readable error.
        message: String,
    },
    /// Start phase failed for a component.
    #[error("start failed for component `{name}`: {message}")]
    Start {
        /// Name of the failing component.
        name: String,
        /// Human-readable error.
        message: String,
    },
    /// Stop phase failed for a component. Other components continue to
    /// stop regardless.
    #[error("stop failed for component `{name}`: {message}")]
    Stop {
        /// Name of the failing component.
        name: String,
        /// Human-readable error.
        message: String,
    },
    /// A reload (hot-swap) operation failed.
    #[error("reload failed for component `{name}`: {message}")]
    Reload {
        /// Name of the component being reloaded.
        name: String,
        /// Human-readable error.
        message: String,
    },
    /// Internal lock acquisition failed.
    #[error("component registry lock poisoned")]
    LockPoisoned,
    /// A typed downcast failed via [`ComponentRegistry::get`].
    #[error("component `{name}` exists but is not of type `{type_id:?}`")]
    TypeMismatch {
        /// The name that was looked up.
        name: String,
        /// The actual `TypeId` of the stored instance.
        type_id: TypeId,
    },
    /// Tried to `init_all` after a previous `init_all` already started
    /// or completed. Use explicit unregister to re-init.
    #[error("component `{name}` has already been initialized")]
    AlreadyInitialized {
        /// The already-initialized name.
        name: String,
    },
}

/// Initialization state of a registered component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ComponentState {
    /// Registered but not yet initialized.
    Pending,
    /// Currently running `init`.
    Initializing,
    /// Successfully initialized and ready to start.
    Initialized,
    /// `start` is in progress.
    Starting,
    /// `start` succeeded; component is serving.
    Running,
    /// `stop` is in progress.
    Stopping,
    /// `stop` completed; component is no longer serving but is still
    /// initialized and could be re-`start`-ed.
    Stopped,
    /// Init failed. See [`RegistryError::Init`].
    Failed,
}

/// Static descriptor of a component kind, used by the registry to plan
/// initialization order without holding the actual factory.
#[derive(Debug, Clone)]
pub struct ComponentDescriptor {
    /// The user-assigned instance name.
    pub name: String,
    /// Names of components this component depends on.
    pub depends_on: Vec<String>,
    /// Raw configuration value. The factory decides how to deserialize it.
    pub config: serde_json::Value,
}

/// Type-erased factory for a concrete [`Component`] type. The registry
/// holds factories as `Box<dyn ComponentFactory>` so it can drive
/// heterogeneous types uniformly.
#[async_trait]
pub trait ComponentFactory: Send + Sync {
    /// User-assigned instance name.
    fn name(&self) -> &str;
    /// Component kind identifier (e.g. `"provider.openai"`).
    fn kind(&self) -> &'static str;
    /// Names of components this component depends on.
    fn depends_on(&self) -> Vec<String>;
    /// Build an [`AnyComponent`] from the raw configuration value.
    async fn build(
        self: Box<Self>,
        config: serde_json::Value,
        ctx: &ComponentContext,
    ) -> Result<Box<dyn AnyComponent>, RegistryError>;
}

/// Convenience wrapper that adapts a concrete [`Component`] into a
/// [`ComponentFactory`].
pub struct TypedFactory<C: Component> {
    name: String,
    extra_deps: Vec<String>,
    _marker: std::marker::PhantomData<fn() -> C>,
}

impl<C: Component> TypedFactory<C> {
    /// Construct a typed factory descriptor.
    #[must_use]
    pub fn new(name: impl Into<String>, extra_deps: Vec<String>) -> Self {
        Self {
            name: name.into(),
            extra_deps,
            _marker: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<C: Component> ComponentFactory for TypedFactory<C> {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> &'static str {
        C::NAME
    }

    fn depends_on(&self) -> Vec<String> {
        let mut deps: Vec<String> = C::depends_on().iter().map(|s| (*s).to_string()).collect();
        for d in &self.extra_deps {
            if !deps.contains(d) {
                deps.push(d.clone());
            }
        }
        deps
    }

    async fn build(
        self: Box<Self>,
        config: serde_json::Value,
        ctx: &ComponentContext,
    ) -> Result<Box<dyn AnyComponent>, RegistryError> {
        let name = self.name.clone();
        let cfg: C::Config = serde_json::from_value(config).map_err(|e| RegistryError::Init {
            name: name.clone(),
            message: format!("config deserialize: {e}"),
        })?;
        let instance = C::init(&cfg, ctx).await.map_err(|e| RegistryError::Init {
            name: name.clone(),
            message: e.to_string(),
        })?;
        Ok(Box::new(TypedAnyComponent {
            name,
            kind: C::NAME,
            instance: Arc::new(instance),
        }))
    }
}

/// Adapter from a typed [`Component`] to the type-erased [`AnyComponent`]
/// trait. Created by [`TypedFactory::build`].
pub struct TypedAnyComponent<C: Component> {
    name: String,
    kind: &'static str,
    instance: Arc<C>,
}

impl<C: Component> TypedAnyComponent<C> {
    /// Wraps a component instance into a type-erased [`AnyComponent`].
    #[must_use]
    pub fn new(instance: C) -> Self {
        Self {
            name: C::NAME.to_owned(),
            kind: C::NAME,
            instance: Arc::new(instance),
        }
    }
}

#[async_trait]
impl<C: Component> AnyComponent for TypedAnyComponent<C> {
    fn name(&self) -> &'static str {
        self.kind
    }

    fn as_any_arc(&self) -> Arc<dyn Any + Send + Sync> {
        self.instance.clone()
    }

    fn start(&self) -> BoxFuture<'_, Result<(), AnyComponentError>> {
        let name = self.name.clone();
        let instance = self.instance.clone();
        Box::pin(async move {
            instance
                .start()
                .await
                .map_err(|e| AnyComponentError::Component {
                    name,
                    message: e.to_string(),
                })
        })
    }

    fn stop(&self) -> BoxFuture<'_, Result<(), AnyComponentError>> {
        let name = self.name.clone();
        let instance = self.instance.clone();
        Box::pin(async move {
            instance
                .stop()
                .await
                .map_err(|e| AnyComponentError::Component {
                    name,
                    message: e.to_string(),
                })
        })
    }

    fn health(&self) -> BoxFuture<'_, HealthStatus> {
        let instance = self.instance.clone();
        Box::pin(async move { instance.health().await })
    }

    fn pre_replace(&self) -> BoxFuture<'_, Result<(), AnyComponentError>> {
        let name = self.name.clone();
        let instance = self.instance.clone();
        Box::pin(async move {
            instance
                .pre_replace_hook()
                .await
                .map_err(|e| AnyComponentError::Component {
                    name,
                    message: e.to_string(),
                })
        })
    }

    fn post_replace(&self) -> BoxFuture<'_, Result<(), AnyComponentError>> {
        let name = self.name.clone();
        let instance = self.instance.clone();
        Box::pin(async move {
            instance
                .post_replace_hook()
                .await
                .map_err(|e| AnyComponentError::Component {
                    name,
                    message: e.to_string(),
                })
        })
    }
}

/// The central component registry.
///
/// Cloning a `ComponentRegistry` is cheap: it is backed by [`Arc`]-shared
/// inner state. All clones observe the same set of registered components.
pub struct ComponentRegistry {
    inner: Arc<RegistryInner>,
}

struct RegistryInner {
    /// Registered factories awaiting initialization. Cleared on init.
    factories: RwLock<HashMap<String, Box<dyn ComponentFactory>>>,
    /// Permanent descriptors; retained so that topo can be rebuilt
    /// after `init_all` clears the factory map.
    descriptors: RwLock<HashMap<String, ComponentDescriptor>>,
    /// Initialized components keyed by name.
    instances: RwLock<HashMap<String, Arc<dyn AnyComponent>>>,
    /// Current state of every registered component.
    states: RwLock<HashMap<String, ComponentState>>,
    /// Cached topological init order. `Some(vec)` once computed, `None`
    /// when invalid.
    topo: RwLock<Option<Vec<String>>>,
    /// Cooperative shutdown token handed to every component on init.
    shutdown: ShutdownToken,
    /// `true` once `init_all` has been called at least once. Used to
    /// detect double-init.
    init_started: AtomicBool,
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ComponentRegistry {
    /// Construct an empty registry with a fresh shutdown token.
    #[must_use]
    pub fn new() -> Self {
        Self::with_shutdown(ShutdownToken::new())
    }

    /// Construct a registry that hands `shutdown` to every component on
    /// init. Used by the future `ManagedRuntime` to wire a single root
    /// shutdown into every component.
    #[must_use]
    pub fn with_shutdown(shutdown: ShutdownToken) -> Self {
        Self {
            inner: Arc::new(RegistryInner {
                factories: RwLock::new(HashMap::new()),
                descriptors: RwLock::new(HashMap::new()),
                instances: RwLock::new(HashMap::new()),
                states: RwLock::new(HashMap::new()),
                topo: RwLock::new(None),
                shutdown,
                init_started: AtomicBool::new(false),
            }),
        }
    }

    /// Borrow the root shutdown token.
    #[must_use]
    pub fn shutdown(&self) -> ShutdownToken {
        self.inner.shutdown.clone()
    }

    /// Register a factory. Invalidates the cached topological order.
    ///
    /// # Errors
    /// - [`RegistryError::AlreadyRegistered`] if the name is taken.
    pub fn register_factory(
        &self,
        descriptor: ComponentDescriptor,
        factory: Box<dyn ComponentFactory>,
    ) -> Result<(), RegistryError> {
        let name = descriptor.name.clone();
        {
            let mut factories = self
                .inner
                .factories
                .write()
                .map_err(|_| RegistryError::LockPoisoned)?;
            if factories.contains_key(&name) {
                return Err(RegistryError::AlreadyRegistered { name });
            }
            factories.insert(name.clone(), factory);
        }
        {
            let mut descriptors = self
                .inner
                .descriptors
                .write()
                .map_err(|_| RegistryError::LockPoisoned)?;
            descriptors.insert(name.clone(), descriptor);
        }
        {
            let mut states = self
                .inner
                .states
                .write()
                .map_err(|_| RegistryError::LockPoisoned)?;
            states.insert(name, ComponentState::Pending);
        }
        self.invalidate_topo();
        Ok(())
    }

    /// Register a typed component using its [`Component::NAME`] constant
    /// and [`Component::depends_on`] metadata.
    pub fn register_typed<C: Component>(
        &self,
        instance_name: impl Into<String>,
        config: serde_json::Value,
    ) -> Result<(), RegistryError> {
        let name = instance_name.into();
        let factory = TypedFactory::<C>::new(name.clone(), Vec::new());
        let descriptor = ComponentDescriptor {
            name: name.clone(),
            depends_on: factory.depends_on(),
            config,
        };
        self.register_factory(descriptor, Box::new(factory))
    }

    /// Unregister a component. Returns the previous descriptor, or
    /// [`RegistryError::NotFound`] if the name was not registered.
    pub fn unregister(&self, name: &str) -> Result<ComponentDescriptor, RegistryError> {
        {
            let mut factories = self
                .inner
                .factories
                .write()
                .map_err(|_| RegistryError::LockPoisoned)?;
            factories.remove(name);
        }
        let descriptor = {
            let mut descriptors = self
                .inner
                .descriptors
                .write()
                .map_err(|_| RegistryError::LockPoisoned)?;
            descriptors
                .remove(name)
                .ok_or_else(|| RegistryError::NotFound {
                    name: name.to_string(),
                })?
        };
        {
            let mut states = self
                .inner
                .states
                .write()
                .map_err(|_| RegistryError::LockPoisoned)?;
            states.remove(name);
        }
        self.invalidate_topo();
        Ok(descriptor)
    }

    /// Number of registered components.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner
            .descriptors
            .read()
            .map(|m| m.len())
            .unwrap_or_default()
    }

    /// Returns `true` if no components are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the user-assigned names of all registered components,
    /// in topological init order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.recompute_topo().unwrap_or_default()
    }

    /// Look up an initialized component by name, downcasting to a
    /// concrete `Arc<C>`.
    ///
    /// # Errors
    /// - [`RegistryError::NotFound`] if the name is not registered.
    /// - [`RegistryError::TypeMismatch`] if the stored instance is not
    ///   of type `C`.
    /// - [`RegistryError::AlreadyInitialized`] is not raised here;
    ///   a registered component without an instance simply returns
    ///   `NotFound`.
    pub fn get<C: Component>(&self, name: &str) -> Result<Arc<C>, RegistryError> {
        let instances = self
            .inner
            .instances
            .read()
            .map_err(|_| RegistryError::LockPoisoned)?;
        let instance = instances.get(name).ok_or_else(|| RegistryError::NotFound {
            name: name.to_string(),
        })?;
        let any = instance.as_any_arc();
        let type_id = any.type_id();
        any.downcast::<C>()
            .map_err(|_| RegistryError::TypeMismatch {
                name: name.to_string(),
                type_id,
            })
    }

    /// Returns `true` if a component with the given name has been
    /// successfully initialized.
    #[must_use]
    pub fn is_initialized(&self, name: &str) -> bool {
        self.inner
            .states
            .read()
            .ok()
            .and_then(|m| m.get(name).copied())
            .is_some_and(|s| {
                matches!(
                    s,
                    ComponentState::Initialized
                        | ComponentState::Running
                        | ComponentState::Starting
                        | ComponentState::Stopping
                        | ComponentState::Stopped
                )
            })
    }

    /// Snapshot of the current state of a component, or `None` if the
    /// name is not registered.
    #[must_use]
    pub fn state_of(&self, name: &str) -> Option<ComponentState> {
        self.inner.states.read().ok()?.get(name).copied()
    }

    /// Initialize all registered components in topological order.
    ///
    /// On error from any single component, the registry records the
    /// failure and continues with components that do not depend on the
    /// failed one. The returned error is the first error encountered
    /// (the rest are recorded in [`ComponentState::Failed`]).
    ///
    /// After a successful `init_all`, subsequent `init_all` calls are
    /// no-ops for already-initialized components. To re-init, unregister
    /// the component and re-register it.
    pub async fn init_all(&self) -> Result<(), RegistryError> {
        self.inner.init_started.store(true, Ordering::SeqCst);
        let order = self.recompute_topo()?;
        let ctx = ComponentContext::new(self.inner.shutdown.child());
        let mut first_error: Option<RegistryError> = None;
        for name in order {
            let state = self.state_of(&name);
            if matches!(
                state,
                Some(
                    ComponentState::Initialized
                        | ComponentState::Running
                        | ComponentState::Starting
                        | ComponentState::Stopping
                        | ComponentState::Stopped,
                )
            ) {
                continue;
            }
            let factory_box = {
                let mut factories = self
                    .inner
                    .factories
                    .write()
                    .map_err(|_| RegistryError::LockPoisoned)?;
                factories.remove(&name)
            };
            let Some(factory_box) = factory_box else {
                continue;
            };
            let config = {
                let descriptors = self
                    .inner
                    .descriptors
                    .read()
                    .map_err(|_| RegistryError::LockPoisoned)?;
                descriptors
                    .get(&name)
                    .map(|d| d.config.clone())
                    .unwrap_or_default()
            };
            self.set_state(&name, ComponentState::Initializing);
            match factory_box.build(config, &ctx).await {
                Ok(any) => {
                    let arc: Arc<dyn AnyComponent> = any.into();
                    self.inner
                        .instances
                        .write()
                        .map_err(|_| RegistryError::LockPoisoned)?
                        .insert(name.clone(), arc);
                    self.set_state(&name, ComponentState::Initialized);
                }
                Err(e) => {
                    self.set_state(&name, ComponentState::Failed);
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Start all initialized components in topo order. Components that
    /// are not in [`ComponentState::Initialized`] (e.g. failed, already
    /// running) are skipped.
    pub async fn start_all(&self) -> Result<(), RegistryError> {
        let order = self.recompute_topo()?;
        let mut first_error: Option<RegistryError> = None;
        for name in order {
            let state = self.state_of(&name);
            if !matches!(state, Some(ComponentState::Initialized)) {
                continue;
            }
            self.set_state(&name, ComponentState::Starting);
            let instance = {
                let instances = self
                    .inner
                    .instances
                    .read()
                    .map_err(|_| RegistryError::LockPoisoned)?;
                instances.get(&name).cloned()
            };
            let Some(instance) = instance else {
                continue;
            };
            match instance.start().await {
                Ok(()) => {
                    self.set_state(&name, ComponentState::Running);
                }
                Err(e) => {
                    self.set_state(&name, ComponentState::Failed);
                    if first_error.is_none() {
                        first_error = Some(RegistryError::Start {
                            name: name.clone(),
                            message: e.to_string(),
                        });
                    }
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Stop all running components in reverse topo order. Continues
    /// even on per-component failure so a stuck component does not
    /// prevent the rest of the system from draining.
    pub async fn stop_all(&self) -> Result<(), RegistryError> {
        let order = self.recompute_topo()?;
        let mut first_error: Option<RegistryError> = None;
        for name in order.into_iter().rev() {
            let state = self.state_of(&name);
            if !matches!(
                state,
                Some(ComponentState::Running | ComponentState::Initialized)
            ) {
                continue;
            }
            self.set_state(&name, ComponentState::Stopping);
            let instance = {
                let instances = self
                    .inner
                    .instances
                    .read()
                    .map_err(|_| RegistryError::LockPoisoned)?;
                instances.get(&name).cloned()
            };
            let Some(instance) = instance else {
                continue;
            };
            if let Err(e) = instance.stop().await {
                if first_error.is_none() {
                    first_error = Some(RegistryError::Stop {
                        name: name.clone(),
                        message: e.to_string(),
                    });
                }
            }
            self.set_state(&name, ComponentState::Stopped);
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Atomically replace a running component with a new instance,
    /// following the drain-aware hot-swap protocol.
    ///
    /// The protocol proceeds in five steps:
    ///
    /// 1. Verify the named component is in [`ComponentState::Running`].
    /// 2. Call [`AnyComponent::pre_replace`] on the old instance, giving
    ///    it a chance to reject new traffic or flush buffers.
    /// 3. Call [`AnyComponent::start`] on the new instance. If this
    ///    fails, the old instance remains in place.
    /// 4. Swap the instance in the registry map. Existing `Arc` clones
    ///    held by other tasks remain valid and keep the old instance
    ///    alive until they are dropped (natural drain).
    /// 5. Call [`AnyComponent::post_replace`] on the old instance
    ///    (best-effort; errors are reported but do not roll back).
    ///
    /// Returns the old [`AnyComponent`] so the caller may await
    /// explicit drain or call `stop` when appropriate.
    ///
    /// # Errors
    /// - [`RegistryError::NotFound`] if no component with `name`
    ///   exists.
    /// - [`RegistryError::Reload`] if the component is not in
    ///   `Running` state, or if `pre_replace` / `start` fails.
    pub async fn replace_instance(
        &self,
        name: &str,
        new_instance: Box<dyn AnyComponent>,
    ) -> Result<Arc<dyn AnyComponent>, RegistryError> {
        let old_instance = {
            let instances = self
                .inner
                .instances
                .read()
                .map_err(|_| RegistryError::LockPoisoned)?;
            instances
                .get(name)
                .cloned()
                .ok_or_else(|| RegistryError::NotFound {
                    name: name.to_string(),
                })?
        };

        if !matches!(self.state_of(name), Some(ComponentState::Running)) {
            return Err(RegistryError::Reload {
                name: name.to_string(),
                message: "component is not in Running state".to_string(),
            });
        }

        if let Err(e) = old_instance.pre_replace().await {
            return Err(RegistryError::Reload {
                name: name.to_string(),
                message: format!("pre_replace hook failed: {e}"),
            });
        }

        if let Err(e) = new_instance.start().await {
            return Err(RegistryError::Reload {
                name: name.to_string(),
                message: format!("new instance start failed: {e}"),
            });
        }

        let new_arc: Arc<dyn AnyComponent> = new_instance.into();
        {
            let mut instances = self
                .inner
                .instances
                .write()
                .map_err(|_| RegistryError::LockPoisoned)?;
            instances.insert(name.to_string(), new_arc);
        }
        self.set_state(name, ComponentState::Running);

        if let Err(e) = old_instance.post_replace().await {
            tracing::warn!(
                component = name,
                error = %e,
                "post_replace hook failed (best-effort; continuing)"
            );
        }

        Ok(old_instance)
    }

    /// Aggregate health of every initialized component. Calls
    /// `AnyComponent::health` for each in topo order and collects the
    /// results.
    #[must_use]
    pub async fn health(&self) -> HashMap<String, HealthStatus> {
        let order = self.recompute_topo().unwrap_or_default();
        let mut out = HashMap::new();
        for name in order {
            let instance = self
                .inner
                .instances
                .read()
                .ok()
                .and_then(|m| m.get(&name).cloned());
            if let Some(instance) = instance {
                let h = instance.health().await;
                out.insert(name, h);
            } else {
                out.insert(name, HealthStatus::unhealthy("not initialized"));
            }
        }
        out
    }

    fn set_state(&self, name: &str, state: ComponentState) {
        if let Ok(mut states) = self.inner.states.write() {
            states.insert(name.to_string(), state);
        }
    }

    fn invalidate_topo(&self) {
        if let Ok(mut topo) = self.inner.topo.write() {
            *topo = None;
        }
    }

    /// Recompute the topological init order from the registered
    /// descriptors. Caches the result until the next registration.
    pub fn recompute_topo(&self) -> Result<Vec<String>, RegistryError> {
        if let Ok(topo) = self.inner.topo.read() {
            if let Some(order) = topo.as_ref() {
                return Ok(order.clone());
            }
        }
        let descriptors = self
            .inner
            .descriptors
            .read()
            .map_err(|_| RegistryError::LockPoisoned)?;
        if descriptors.is_empty() {
            if let Ok(mut cached) = self.inner.topo.write() {
                *cached = Some(Vec::new());
            }
            return Ok(Vec::new());
        }
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        for (name, desc) in descriptors.iter() {
            graph.entry(name.clone()).or_default();
            in_degree.entry(name.clone()).or_insert(0);
            for dep in &desc.depends_on {
                if !descriptors.contains_key(dep) {
                    return Err(RegistryError::MissingDependency {
                        name: name.clone(),
                        dep: dep.clone(),
                    });
                }
                graph.entry(dep.clone()).or_default().push(name.clone());
                *in_degree.entry(name.clone()).or_insert(0) += 1;
            }
        }

        // Kahn's algorithm with stable ordering for determinism.
        let mut queue: VecDeque<String> = {
            #[allow(clippy::filter_map_bool_then)]
            in_degree
                .iter()
                .filter_map(|(n, d)| (*d == 0).then(|| n.clone()))
                .collect()
        };
        let mut initial: Vec<String> = queue.drain(..).collect();
        initial.sort_unstable();
        for n in initial {
            queue.push_back(n);
        }

        let mut order: Vec<String> = Vec::with_capacity(in_degree.len());
        while let Some(n) = queue.pop_front() {
            order.push(n.clone());
            if let Some(deps) = graph.get(&n) {
                for m in deps {
                    if let Some(d) = in_degree.get_mut(m) {
                        *d -= 1;
                        if *d == 0 {
                            queue.push_back(m.clone());
                        }
                    }
                }
            }
        }

        if order.len() != in_degree.len() {
            let leftover: Vec<String> = {
                #[allow(clippy::filter_map_bool_then)]
                in_degree
                    .iter()
                    .filter_map(|(n, d)| (*d > 0).then(|| n.clone()))
                    .collect()
            };
            return Err(RegistryError::Cycle { cycle: leftover });
        }

        if let Ok(mut cached) = self.inner.topo.write() {
            *cached = Some(order.clone());
        }
        Ok(order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct EchoConfig {
        label: String,
    }

    struct EchoComponent {
        label: String,
    }

    #[async_trait]
    impl Component for EchoComponent {
        const NAME: &'static str = "test.echo";
        type Config = EchoConfig;
        type Error = std::io::Error;

        async fn init(cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
            Ok(Self {
                label: cfg.label.clone(),
            })
        }
    }

    fn cfg(label: &str) -> serde_json::Value {
        serde_json::json!({ "label": label })
    }

    #[tokio::test]
    async fn register_init_start_stop_lifecycle() {
        let reg = ComponentRegistry::new();
        reg.register_typed::<EchoComponent>("a", cfg("a-label"))
            .unwrap_or_else(|e| panic!("{e}"));
        reg.register_typed::<EchoComponent>("b", cfg("b-label"))
            .unwrap_or_else(|e| panic!("{e}"));

        reg.init_all().await.unwrap_or_else(|e| panic!("{e}"));
        assert!(reg.is_initialized("a"));
        assert!(reg.is_initialized("b"));

        reg.start_all().await.unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(reg.state_of("a"), Some(ComponentState::Running));

        reg.stop_all().await.unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(reg.state_of("a"), Some(ComponentState::Stopped));
    }

    #[tokio::test]
    async fn topo_respects_dependencies() {
        struct Dep;
        #[async_trait]
        impl Component for Dep {
            const NAME: &'static str = "test.dep";
            type Config = serde_json::Value;
            type Error = std::io::Error;
            async fn init(
                _cfg: &Self::Config,
                _ctx: &ComponentContext,
            ) -> Result<Self, Self::Error> {
                Ok(Dep)
            }
        }
        struct User;
        #[async_trait]
        impl Component for User {
            const NAME: &'static str = "test.user";
            type Config = serde_json::Value;
            type Error = std::io::Error;
            async fn init(
                _cfg: &Self::Config,
                _ctx: &ComponentContext,
            ) -> Result<Self, Self::Error> {
                Ok(User)
            }
        }
        let reg = ComponentRegistry::new();
        // Register `user` with an explicit extra-dep on the instance
        // name "dep". The registry resolves dependencies against user
        // assigned instance names, not against `Component::NAME`.
        reg.register_typed::<User>("user", serde_json::json!({}))
            .unwrap_or_else(|e| panic!("{e}"));
        reg.register_typed::<Dep>("dep", serde_json::json!({}))
            .unwrap_or_else(|e| panic!("{e}"));
        // Re-register `user` with the extra-dep, since the first
        // registration did not carry one.
        reg.unregister("user").unwrap_or_else(|e| panic!("{e}"));
        reg.register_factory(
            ComponentDescriptor {
                name: "user".into(),
                depends_on: vec!["dep".into()],
                config: serde_json::json!({}),
            },
            Box::new(TypedFactory::<User>::new("user", vec!["dep".into()])),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let order = reg.recompute_topo().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(order, vec!["dep".to_string(), "user".to_string()]);
    }

    #[tokio::test]
    async fn duplicate_registration_rejected() {
        let reg = ComponentRegistry::new();
        reg.register_typed::<EchoComponent>("a", cfg("a"))
            .unwrap_or_else(|e| panic!("{e}"));
        let err = match reg.register_typed::<EchoComponent>("a", cfg("b")) {
            Ok(_) => panic!("expected Err, got Ok"),
            Err(e) => e,
        };
        assert!(matches!(err, RegistryError::AlreadyRegistered { .. }));
    }

    #[tokio::test]
    async fn missing_dependency_detected() {
        let reg = ComponentRegistry::new();
        let factory = TypedFactory::<EchoComponent>::new("a", vec!["missing".to_string()]);
        reg.register_factory(
            ComponentDescriptor {
                name: "a".into(),
                depends_on: vec!["missing".into()],
                config: cfg("a"),
            },
            Box::new(factory),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let err = match reg.init_all().await {
            Ok(_) => panic!("expected Err, got Ok"),
            Err(e) => e,
        };
        assert!(matches!(err, RegistryError::MissingDependency { .. }));
    }

    #[tokio::test]
    async fn cycle_detected() {
        let reg = ComponentRegistry::new();
        let factory_a = TypedFactory::<EchoComponent>::new("a", vec!["b".to_string()]);
        let factory_b = TypedFactory::<EchoComponent>::new("b", vec!["a".to_string()]);
        reg.register_factory(
            ComponentDescriptor {
                name: "a".into(),
                depends_on: vec!["b".into()],
                config: cfg("a"),
            },
            Box::new(factory_a),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        reg.register_factory(
            ComponentDescriptor {
                name: "b".into(),
                depends_on: vec!["a".into()],
                config: cfg("b"),
            },
            Box::new(factory_b),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let err = match reg.init_all().await {
            Ok(_) => panic!("expected Err, got Ok"),
            Err(e) => e,
        };
        assert!(matches!(err, RegistryError::Cycle { .. }));
    }

    #[tokio::test]
    async fn health_aggregates_initialized_components() {
        let reg = ComponentRegistry::new();
        reg.register_typed::<EchoComponent>("a", cfg("a"))
            .unwrap_or_else(|e| panic!("{e}"));
        reg.init_all().await.unwrap_or_else(|e| panic!("{e}"));
        reg.start_all().await.unwrap_or_else(|e| panic!("{e}"));
        let h = reg.health().await;
        assert!(h.get("a").map(|s| s.is_healthy()).unwrap_or(false));
    }

    #[tokio::test]
    async fn get_downcasts_to_concrete_type() {
        let reg = ComponentRegistry::new();
        reg.register_typed::<EchoComponent>("a", cfg("hello"))
            .unwrap_or_else(|e| panic!("{e}"));
        reg.init_all().await.unwrap_or_else(|e| panic!("{e}"));
        let c: Arc<EchoComponent> = reg.get("a").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(c.label, "hello");
    }

    #[tokio::test]
    async fn get_returns_type_mismatch_on_wrong_type() {
        #[derive(Debug)]
        struct Other;
        #[async_trait]
        impl Component for Other {
            const NAME: &'static str = "test.other";
            type Config = serde_json::Value;
            type Error = std::io::Error;
            async fn init(
                _cfg: &Self::Config,
                _ctx: &ComponentContext,
            ) -> Result<Self, Self::Error> {
                Ok(Other)
            }
        }
        let reg = ComponentRegistry::new();
        reg.register_typed::<EchoComponent>("a", cfg("a"))
            .unwrap_or_else(|e| panic!("{e}"));
        reg.init_all().await.unwrap_or_else(|e| panic!("{e}"));
        let err = match reg.get::<Other>("a") {
            Ok(_) => panic!("expected Err, got Ok"),
            Err(e) => e,
        };
        assert!(matches!(err, RegistryError::TypeMismatch { .. }));
    }

    #[tokio::test]
    async fn unregister_removes_state() {
        let reg = ComponentRegistry::new();
        reg.register_typed::<EchoComponent>("a", cfg("a"))
            .unwrap_or_else(|e| panic!("{e}"));
        reg.unregister("a").unwrap_or_else(|e| panic!("{e}"));
        assert!(!reg.is_initialized("a"));
        assert_eq!(reg.state_of("a"), None);
    }

    #[tokio::test]
    async fn names_returns_topo_order() {
        let reg = ComponentRegistry::new();
        reg.register_typed::<EchoComponent>("zzz", cfg("z"))
            .unwrap_or_else(|e| panic!("{e}"));
        reg.register_typed::<EchoComponent>("aaa", cfg("a"))
            .unwrap_or_else(|e| panic!("{e}"));
        let names = reg.names();
        // With no dependencies, the topo sort returns names in
        // stable alphabetical order for determinism.
        assert_eq!(names.first().map(String::as_str), Some("aaa"));
        assert_eq!(names.last().map(String::as_str), Some("zzz"));
    }
}
