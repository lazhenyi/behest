//! Agent runtime kernel.
//!
//! This module provides the core runtime engine for executing AI agents
//! with streaming-first execution, tool calling, session persistence,
//! and event-sourced state management.

pub mod accumulator;
pub mod agent;
pub mod compaction;
pub mod context;
pub mod doom_loop;
pub mod error;
pub mod event;
pub mod event_store;
pub mod input;
pub mod invocation;
pub mod job;
pub mod memory;
pub mod policy;
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

pub use agent::{AgentRuntime, RunOutput};
pub use compaction::{CompactionResult, CompactionService};
pub use context::ContextPipeline;
pub use doom_loop::{DoomLoopConfig, DoomLoopDetector, DoomLoopType, ToolCallFingerprint};
pub use error::RuntimeError;
pub use event::AgentEvent;
pub use event_store::{
    DynRuntimeEventStore, FailingRuntimeEventStore, MemoryRuntimeEventStore, RuntimeEventStore,
    RuntimeEventStoreError,
};
pub use input::{
    InputAdmission, InputAdmissionConfig, InputEvent, InputId, InputRecord, InputState,
};
pub use invocation::{
    Control, EmitRequest, EventKind, InvocationError, InvocationEvent, InvocationHandle,
    RuntimeInvocation, SessionContext,
};
pub use job::{BackgroundJob, BackgroundJobPool, JobConditions, JobPriority, JobType};
pub use policy::{CompactionConfig, RuntimePolicy};
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
