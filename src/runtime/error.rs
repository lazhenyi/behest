//! Runtime error types.

use thiserror::Error;

use super::run::RunId;

/// Errors that can occur during runtime execution.
///
/// Covers provider resolution, session management, run lifecycle, policy
/// enforcement, tool execution, storage, snapshot recovery, doom-loop
/// detection, and input admission failures.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// The requested provider was not found in the registry.
    #[error("provider not found: {0}")]
    ProviderNotFound(String),

    /// The requested session does not exist.
    #[error("session not found: {0}")]
    SessionNotFound(uuid::Uuid),

    /// The requested run was not found.
    #[error("run not found: {0}")]
    RunNotFound(RunId),

    /// Operation cannot proceed because the run is in an unexpected state.
    #[error("invalid run state: expected {expected}, got {actual}")]
    InvalidRunState {
        /// Expected state descriptor.
        expected: String,
        /// Actual state descriptor.
        actual: String,
    },

    /// The run exceeded the maximum allowed model call iterations.
    #[error("iteration limit exceeded: {0}")]
    IterationLimitExceeded(usize),

    /// Accumulated token usage exceeded the configured budget.
    #[error("token budget exceeded: {used} > {limit}")]
    TokenBudgetExceeded {
        /// Tokens consumed so far.
        used: usize,
        /// Maximum allowed tokens.
        limit: usize,
    },

    /// Estimated context size exceeds the model's maximum context window.
    #[error("context length exceeded: model context {context}, estimated {estimated}")]
    ContextOverflow {
        /// Model's maximum context window size.
        context: u32,
        /// Estimated token usage for the full context.
        estimated: usize,
    },

    /// Session is locked by another concurrent run.
    #[error("session busy: {0}")]
    SessionBusy(uuid::Uuid),

    /// A tool call exceeded its configured execution timeout.
    #[error("tool execution timeout: {tool}")]
    ToolTimeout {
        /// Name of the timed-out tool.
        tool: String,
    },

    /// An error propagated from the provider layer.
    #[error(transparent)]
    Provider(#[from] crate::error::ProviderError),

    /// An error propagated from the context layer.
    #[error(transparent)]
    Context(#[from] crate::error::ContextError),

    /// An error propagated from the storage layer.
    #[error(transparent)]
    Storage(#[from] crate::error::StorageError),

    /// An error propagated from the tool layer.
    #[error(transparent)]
    Tool(#[from] crate::error::ToolError),

    /// Snapshot persistence or recovery failed.
    #[error("recovery error: {0}")]
    RecoveryFailed(String),

    /// Doom loop detected — agent is stuck in a repetitive tool call pattern.
    #[error("doom loop detected: {description}")]
    DoomLoopDetected {
        /// Human-readable description of the detected pattern.
        description: String,
    },

    /// Input was rejected by the admission pipeline.
    #[error("input rejected: {input_id} — {reason}")]
    InputRejected {
        /// Identifier of the rejected input.
        input_id: crate::runtime::input::InputId,
        /// Reason for rejection.
        reason: String,
    },

    /// Internal error during input admission processing.
    #[error("input admission error: {0}")]
    InputAdmissionFailed(String),
}

/// Result type for runtime operations.
pub type RuntimeResult<T> = Result<T, RuntimeError>;
