//! Agent runtime kernel.
//!
//! This module provides the core runtime engine for executing AI agents
//! with streaming-first execution, tool calling, session persistence,
//! event-sourced state management, snapshot recovery, compaction,
//! doom-loop detection, and background job processing.

pub mod accumulator;
pub mod agent;
pub mod compaction;
pub mod component;
pub mod component_factory;
pub mod components;
pub mod context;
pub mod doom_loop;
pub mod drain;
pub mod error;
pub mod event;
pub mod event_store;
pub mod extension;
pub mod extensions;
pub mod factory_registry;
pub mod input;
pub mod invocation;
pub mod job;
pub mod lifecycle;
pub mod managed;
pub mod memory;
pub mod policy;
pub mod registry;
pub mod replace;
pub mod router;
pub mod run;
mod run_loop;
pub mod session_gate;
pub mod snapshot;
pub mod state;
pub mod store;
pub mod stream;
pub mod stream_adapter;
pub mod subscription;
pub mod tool;
pub mod turn;

#[cfg(feature = "redis")]
pub mod session_data_store;

pub use agent::{AgentRuntime, RunOutput};
pub use compaction::{CompactionResult, CompactionService};
pub use component::{AnyComponent, AnyComponentError, Component, ComponentContext};
pub use component_factory::{
    ChatProviderAny, ChatProviderComponent, ChatProviderFactory, EmbeddingProviderAny,
    EmbeddingProviderComponent, EmbeddingProviderFactory, EmptyConfig, WrapperError,
};
pub use components::{
    AnthropicChatComponent, ComponentError, ContextPipelineComponent, ContextPipelineConfig,
    MemoryArtifactStoreComponent, MemoryEmbeddingStoreComponent, MemoryExecutionStoreComponent,
    MemoryRunStoreComponent, MemorySessionStoreComponent, OpenAiChatComponent,
    OpenAiEmbeddingComponent, ProviderHttpComponentConfig, default_factory_registry,
    register_context_pipeline, register_memory_stores, register_providers,
};
pub use context::ContextPipeline;
pub use doom_loop::{DoomLoopConfig, DoomLoopDetector, DoomLoopType, ToolCallFingerprint};
pub use drain::{DrainGuard, DrainResult};
pub use error::RuntimeError;
pub use event::AgentEvent;
pub use event_store::{
    DynRuntimeEventStore, FailingRuntimeEventStore, MemoryRuntimeEventStore, RuntimeEventStore,
    RuntimeEventStoreError,
};
pub use extension::{ExtensionError, ExtensionPoint};
pub use extensions::Extensions;
pub use factory_registry::{FactoryError, FactoryFn, FactoryRegistry};
pub use input::{
    InputAdmission, InputAdmissionConfig, InputEvent, InputId, InputRecord, InputState,
};
pub use invocation::{
    Control, EmitRequest, EventKind, FileSessionDataStore, InvocationError, InvocationEvent,
    InvocationHandle, InvocationSession, MemorySessionDataStore, RuntimeInvocation,
    SessionDataError, SessionDataStore,
};
pub use job::{BackgroundJob, BackgroundJobPool, JobConditions, JobPriority, JobType};
pub use lifecycle::ShutdownToken;
pub use managed::{ManagedError, ManagedRuntime};
pub use policy::{CompactionConfig, RuntimePolicy};
pub use registry::{
    ComponentDescriptor, ComponentFactory, ComponentRegistry, ComponentState, RegistryError,
    TypedAnyComponent, TypedFactory,
};
pub use replace::{DEFAULT_DRAIN_TIMEOUT, ReplaceError, ReplaceState, ReplaceToken};
pub use router::ModelRouter;
pub use run::{RunId, RunRequest, RunStatus};
pub use session_gate::{SessionGate, SessionGuard};
pub use snapshot::{FileSnapshotStore, Snapshot, SnapshotStore};
pub use state::RunState;
pub use store::RuntimeStore;
pub use stream::{
    BoxRuntimeEventStream, RuntimeEventEnvelope, RuntimeEventId, RuntimeRoom, RuntimeStreamError,
};
pub use stream_adapter::{
    DynRuntimeStreamAdapter, FailingRuntimeStreamAdapter, MemoryRuntimeStreamAdapter,
    RuntimeStreamAdapter,
};
pub use subscription::{
    RuntimeEventBridge, RuntimeEventBridgeError, RuntimeEventBridgeHandle, RuntimeSubscription,
    RuntimeSubscriptionError, RuntimeSubscriptionHub,
};
pub use tool::ToolRuntime;
pub use turn::{TurnState, TurnTransition};

#[cfg(feature = "redis")]
pub use event_store::redis::RedisRuntimeEventStore;

#[cfg(feature = "redis")]
pub use session_data_store::RedisSessionDataStore;

#[cfg(feature = "redis")]
pub use stream_adapter::redis::RedisRuntimeStreamAdapter;

#[cfg(feature = "sqlx-postgres")]
pub use event_store::postgres::PostgresRuntimeEventStore;

#[cfg(feature = "nats")]
pub use stream_adapter::nats_jetstream::NatsJetStreamStreamAdapter;
