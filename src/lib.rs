#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![warn(rust_2018_idioms)]
#![allow(unused)]

//! Production-oriented building blocks for Rust-native AI agent runtimes.
//!
//! The crate currently focuses on provider-neutral chat, streaming, tool-calling,
//! and embedding contracts. Runtime integrations can implement the provider traits
//! and register them in [`provider::ProviderRegistry`].

pub mod error;
pub mod prelude;
pub mod provider;
pub mod adapt;

pub use crate::error::{Error, ProviderError, Result};
