//! Agent runtime kernel.
//!
//! This module provides the core runtime engine for executing AI agents
//! with streaming-first execution, tool calling, session persistence,
//! and event-sourced state management.

pub mod accumulator;
pub mod agent;
pub mod context;
pub mod error;
pub mod event;
pub mod memory;
pub mod policy;
pub mod router;
pub mod run;
pub mod store;
pub mod tool;

pub use agent::{AgentRuntime, RunOutput};
pub use context::ContextPipeline;
pub use error::RuntimeError;
pub use event::AgentEvent;
pub use policy::RuntimePolicy;
pub use router::ModelRouter;
pub use run::{RunId, RunRequest, RunStatus};
pub use store::RuntimeStore;
pub use tool::ToolRuntime;
