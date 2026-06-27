#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![warn(rust_2018_idioms)]

//! Production-oriented building blocks for Rust-native AI agent runtimes.
//!
//! The crate currently focuses on provider-neutral chat, streaming, tool-calling,
//! and embedding contracts. Runtime integrations can implement the provider traits
//! and register them in [`provider::ProviderRegistry`].
//!
//! # Modules
//!
//! - [`config`]: Centralized configuration with file, env, and manual builder support
//! - [`runtime`]: Agent runtime kernel with streaming-first execution
//! - [`provider`]: Provider traits, request/response types, and registry
//! - [`tool`]: Runtime tool registry and execution
//! - [`context`]: Multi-adapter context factory for composing chat requests
//! - [`store`]: Persistence layer for sessions, embeddings, and artifacts
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

#[cfg(feature = "server")]
pub mod transport;

pub use crate::error::{ContextError, Error, ProviderError, Result, StorageError, ToolError};
pub use crate::health::HealthStatus;
pub use crate::runtime::{
    AgentEvent, AgentRuntime, AnyComponent, AnyComponentError, CompactionResult, CompactionService,
    Component, ComponentContext, ComponentDescriptor, ComponentFactory, ComponentRegistry,
    ComponentState, Control, EmitRequest, EventKind, ExtensionError, ExtensionPoint, Extensions,
    FileSessionDataStore, FileSnapshotStore, InvocationError, InvocationEvent, InvocationHandle,
    InvocationSession, MemorySessionDataStore, ModelRouter, RegistryError, RunId, RunOutput,
    RunRequest, RunStatus, RuntimeError, RuntimeEventBridge, RuntimeEventBridgeError,
    RuntimeEventBridgeHandle, RuntimeEventEnvelope, RuntimeEventId, RuntimeEventStore,
    RuntimeEventStoreError, RuntimeInvocation, RuntimePolicy, RuntimeRoom, RuntimeStreamAdapter,
    RuntimeStreamError, RuntimeSubscription, RuntimeSubscriptionError, RuntimeSubscriptionHub,
    SessionDataError, SessionDataStore, ShutdownToken, Snapshot, SnapshotStore, TypedAnyComponent,
    TypedFactory,
};
pub use crate::tool_output::{ToolOutputConfig, TruncationResult};

#[cfg(feature = "redis")]
pub use crate::runtime::RedisSessionDataStore;
