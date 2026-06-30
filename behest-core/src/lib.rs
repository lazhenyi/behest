//! Core types for the behest agent runtime.
//!
//! This crate provides foundational types used across the behest ecosystem:
//! - Strongly-typed identifiers ([`id::ProviderId`], [`id::ModelName`])
//! - Error types ([`error::Error`], [`error::ProviderError`], [`error::ToolError`], etc.)
//! - Provider-neutral message types ([`message::Message`], [`message::ChatRequest`], [`message::ChatResponse`])
//! - Tool and embedding types
//! - Sans-IO run state machine ([`run::RunState`], [`run::RunInput`], [`run::RunAction`])

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]

pub mod cache;
pub mod capabilities;
pub mod embedding;
pub mod error;
pub mod events;
pub mod health;
pub mod id;
pub mod message;
pub mod run;
pub mod token;
pub mod tool_types;
