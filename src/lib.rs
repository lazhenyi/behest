#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![warn(rust_2018_idioms)]

//! Building blocks for Rust-native AI agent runtimes.
//!
//! # Architecture
//!
//! behest has two complementary surfaces:
//!
//! **The invocation surface** — a Socket.IO-inspired `emit` / `on` model
//! for interacting with an agent. Emit a user message, receive streaming
//! events back through typed handlers. This is the _caller-facing_ API,
//! built on [`RuntimeInvocation`].
//!
//! **The assembly surface** — pluggable components (providers, tools,
//! stores, transports) wired together via [`ComponentRegistry`]
//! and driven by a coordinated lifecycle. This is the _operator-facing_ API
//! for composing a runtime from parts, built on [`ManagedRuntime`].
//!
//! ```text
//!                   emit("question")       on(TextDelta, handler)
//!   Caller ──────────►  RuntimeInvocation  ──────────►  Caller
//!                          │
//!                          ▼
//!                    AgentRuntime  ◄──  ComponentRegistry
//!                          │              │
//!                          ▼              ▼
//!                 run_loop (FSM)      provider / tool / store / transport
//! ```
//!
//! The two surfaces are independent: you can use `RuntimeInvocation` with
//! a hand-assembled `AgentRuntime`, or drive the `ManagedRuntime` lifecycle
//! from your own invocation layer. They meet at `AgentRuntime` — the shared
//! kernel that both paths produce.
//!
//! # Quick start
//!
//! ```ignore
//! use behest::prelude::*;
//! use behest::runtime::RuntimeInvocation;
//!
//! // 1. Build config & runtime
//! let config = AgentConfigBuilder::default()
//!     .with_env("BEHEST")?
//!     .with_provider(ProviderId::new("openai"), ProviderConfig::new("https://api.openai.com/v1")
//!         .with_provider_type(ProviderType::OpenAi)
//!         .with_api_key("env:OPENAI_API_KEY")
//!         .with_model("gpt-4o"))
//!     .build()?;
//! let runtime = Arc::new(config.into_runtime().await?);
//!
//! // 2. Wrap in the invocation facade
//! let inv = RuntimeInvocation::new(runtime);
//!
//! // 3. Subscribe to text deltas
//! let handle = inv.on(EventKind::TextDelta, |envelope, _session, _control| async move {
//!     if let AgentEvent::TextDelta(td) = &envelope.event {
//!         print!("{}", td.delta);
//!     }
//! }).await?;
//!
//! // 4. Emit a request
//! let output = inv.emit(|_session, _control| async move {
//!     EmitRequest::new(provider_id, model, "Hello, world!")
//! }).await?;
//!
//! drop(handle);
//! ```
//!
//! # Modules
//!
//! - [`runtime`]: Runtime kernel, invocation facade, component system, policy
//! - [`provider`]: Provider traits, request/response types, and registry
//! - [`tool`]: Tool trait, registry, scoping
//! - [`context`]: Multi-adapter context factory for composing chat requests
//! - [`store`]: Persistence traits and backends for sessions, embeddings, artifacts
//! - [`config`]: Centralized configuration with file, env, and builder support
//! - [`adapt`]: Concrete provider adapters (OpenAI, Anthropic)
//! - [`error`]: Error types and result aliases
//! - [`rag`]: Retrieval-Augmented Generation context adapter (feature = `rag`)
//! - [`queue`]: External event publishing to message brokers (feature = `queue`)

pub mod adapt;
pub mod agent;
pub mod config;
pub mod context;
pub mod error;
pub mod health;
pub mod prelude;
pub mod provider;
pub mod runtime;
pub mod store;
pub mod token;
pub mod tool;
pub mod tool_output;
pub mod tool_scope;

#[cfg(feature = "rag")]
pub mod rag;

#[cfg(feature = "queue")]
pub mod queue;

pub use crate::error::{ContextError, Error, ProviderError, Result, StorageError, ToolError};
pub use crate::health::HealthStatus;
pub use crate::runtime::{
    AgentEvent, AgentRuntime, AnyComponent, AnyComponentError, CompactionResult, CompactionService,
    Component, ComponentContext, ComponentDescriptor, ComponentError, ComponentFactory,
    ComponentRegistry, ComponentState, ContextPipelineComponent, ContextPipelineConfig, Control,
    EmitRequest, EventKind, ExtensionError, ExtensionPoint, Extensions, FactoryError, FactoryFn,
    FactoryRegistry, FileSessionDataStore, FileSnapshotStore, InvocationError, InvocationEvent,
    InvocationHandle, InvocationSession, ManagedError, ManagedRuntime,
    MemoryArtifactStoreComponent, MemoryEmbeddingStoreComponent, MemoryExecutionStoreComponent,
    MemoryRunStoreComponent, MemorySessionDataStore, MemorySessionStoreComponent, ModelRouter,
    ProviderHttpComponentConfig, RegistryError, RunId, RunOutput, RunRequest, RunStatus,
    RuntimeError, RuntimeEventBridge, RuntimeEventBridgeError, RuntimeEventBridgeHandle,
    RuntimeEventEnvelope, RuntimeEventId, RuntimeEventStore, RuntimeEventStoreError,
    RuntimeInvocation, RuntimePolicy, RuntimeRoom, RuntimeStreamAdapter, RuntimeStreamError,
    RuntimeSubscription, RuntimeSubscriptionError, RuntimeSubscriptionHub, SessionDataError,
    SessionDataStore, ShutdownToken, Snapshot, SnapshotStore, TypedAnyComponent, TypedFactory,
    default_factory_registry, register_context_pipeline, register_memory_stores,
    register_providers,
};

#[cfg(feature = "openai")]
pub use crate::runtime::{OpenAiChatComponent, OpenAiEmbeddingComponent};

#[cfg(feature = "anthropic")]
pub use crate::runtime::AnthropicChatComponent;
pub use crate::tool_output::{ToolOutputConfig, TruncationResult};

#[cfg(feature = "redis")]
pub use crate::runtime::RedisSessionDataStore;
