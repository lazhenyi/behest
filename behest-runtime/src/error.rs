//! Error types for the behest runtime.

use behest_core::error::ProviderError;
use thiserror::Error;

/// Result type for runtime operations.
pub type RuntimeResult<T> = std::result::Result<T, RuntimeError>;

/// Errors produced by the runtime layer.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// A provider returned an unrecoverable error.
    #[error("provider error: {0}")]
    Provider(#[from] ProviderError),

    /// The run was cancelled.
    #[error("run was cancelled")]
    Cancelled,

    /// Context building failed.
    #[error("context error: {0}")]
    Context(String),

    /// Tool execution failed (aggregated).
    #[error("tool execution error: {0}")]
    Tool(String),

    /// Memory operation failed.
    #[error("memory error: {0}")]
    Memory(String),
}
