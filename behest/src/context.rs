//! Re-export of the high-level context factory API, which now lives
//! in [`behest_context`]. This module is preserved for facade
//! compatibility; prefer `use behest_context::*` directly in new code.

pub use behest_context::{
    ContextAdapter, ContextFactory, ContextInput, ContextOutput, ContextResult, FunctionAdapter,
    StaticAdapter,
};
