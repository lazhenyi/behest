//! Async runtime kernel for the behest agent runtime.
//!
//! During migration from `behest::runtime`, some doc and pub lint
//! warnings are temporarily relaxed. They will be re-enabled once
//! the migration is complete.

#![forbid(unsafe_code)]
// #![deny(missing_docs)]
// #![deny(unreachable_pub)]

pub mod accumulator;
pub mod agent;
pub mod cache_stats;
pub mod compaction;
pub mod component;
pub mod component_factory;
pub mod components;
pub mod context;
pub mod doom_loop;
pub mod error;
pub mod event;
pub mod event_publisher;
pub mod event_store;
pub mod extension;
pub mod extensions;
pub mod factory_registry;
pub mod input;
pub mod invocation;
pub mod lifecycle;
pub mod managed;
pub mod memory;
pub mod policy;
pub mod registry;
pub mod router;
pub mod run;
pub mod run_loop;
#[cfg(feature = "redis")]
pub mod session_data_store;
pub mod session_gate;
pub mod snapshot;
pub mod state;
pub mod store;
pub mod stream;
pub mod stream_adapter;
pub mod subscription;
pub mod token;
pub mod tool_output;
pub mod tool_runtime;
pub mod tool_scope;
pub mod turn;

pub use error::RuntimeError;
