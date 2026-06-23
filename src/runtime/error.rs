//! Runtime error types.

use thiserror::Error;

use super::run::RunId;

/// Errors that can occur during runtime execution.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Provider not found in registry.
    #[error("provider not found: {0}")]
    ProviderNotFound(String),

    /// Session not found.
    #[error("session not found: {0}")]
    SessionNotFound(uuid::Uuid),

    /// Run not found.
    #[error("run not found: {0}")]
    RunNotFound(RunId),

    /// Run is in invalid state for operation.
    #[error("invalid run state: expected {expected}, got {actual}")]
    InvalidRunState {
        /// Expected state.
        expected: String,
        /// Actual state.
        actual: String,
    },

    /// Maximum iteration limit exceeded.
    #[error("iteration limit exceeded: {0}")]
    IterationLimitExceeded(usize),

    /// Token budget exceeded.
    #[error("token budget exceeded: {used} > {limit}")]
    TokenBudgetExceeded {
        /// Tokens used.
        used: usize,
        /// Token limit.
        limit: usize,
    },

    /// Context length exceeded the model's maximum.
    #[error("context length exceeded: model context {context}, estimated {estimated}")]
    ContextOverflow {
        /// Model context window size.
        context: u32,
        /// Estimated token usage.
        estimated: usize,
    },

    /// Session is already being processed by another run.
    #[error("session busy: {0}")]
    SessionBusy(uuid::Uuid),

    /// Tool execution timeout.
    #[error("tool execution timeout: {tool}")]
    ToolTimeout {
        /// Tool name.
        tool: String,
    },

    /// Provider error.
    #[error(transparent)]
    Provider(#[from] crate::error::ProviderError),

    /// Context error.
    #[error(transparent)]
    Context(#[from] crate::error::ContextError),

    /// Storage error.
    #[error(transparent)]
    Storage(#[from] crate::error::StorageError),

    /// Tool error.
    #[error(transparent)]
    Tool(#[from] crate::error::ToolError),

    /// Snapshot or recovery failed.
    #[error("recovery error: {0}")]
    RecoveryFailed(String),

    /// Doom loop detected — agent is stuck in repetitive tool call pattern.
    #[error("doom loop detected: {description}")]
    DoomLoopDetected {
        /// Human-readable description of the detected pattern.
        description: String,
    },

    /// Input was rejected by the admission pipeline.
    #[error("input rejected: {input_id} — {reason}")]
    InputRejected {
        /// Input identifier.
        input_id: crate::runtime::input::InputId,
        /// Rejection reason.
        reason: String,
    },

    /// Input admission internal error.
    #[error("input admission error: {0}")]
    InputAdmissionFailed(String),
}

/// Result type for runtime operations.
pub type RuntimeResult<T> = Result<T, RuntimeError>;
