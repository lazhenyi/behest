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
//! - [`runtime`]: Agent runtime kernel with streaming-first execution
//! - [`provider`]: Provider traits, request/response types, and registry
//! - [`tool`]: Runtime tool registry and execution
//! - [`context`]: Multi-adapter context factory for composing chat requests
//! - [`store`]: Persistence layer for sessions, embeddings, and artifacts
//! - [`adapt`]: Concrete provider adapters (OpenAI, Anthropic)
//! - [`error`]: Error types and result aliases

pub mod adapt;
pub mod context;
pub mod error;
pub mod prelude;
pub mod provider;
pub mod runtime;
pub mod store;
pub mod tool;

pub use crate::error::{ContextError, Error, ProviderError, Result, StorageError, ToolError};
pub use crate::runtime::{
    AgentEvent, AgentRuntime, ModelRouter, RunId, RunOutput, RunRequest, RunStatus, RuntimeError,
    RuntimePolicy,
};
