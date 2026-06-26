//! Error types for the reasoning graph runtime.

use thiserror::Error;

/// Result type alias for reasoning operations.
pub type ReasoningResult<T> = std::result::Result<T, ReasoningError>;

/// Errors produced during reasoning graph construction and execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReasoningError {
    /// An operator failed during execution.
    #[error("operator `{operator}` failed: {message}")]
    OperatorFailed {
        /// Name of the operator that failed.
        operator: String,
        /// Human-readable failure description.
        message: String,
    },

    /// Invalid state transition attempted.
    #[error("invalid state transition: {message}")]
    InvalidTransition {
        /// Description of the invalid transition.
        message: String,
    },

    /// Strategy compilation into a reasoning graph failed.
    #[error("strategy compilation failed: {message}")]
    CompilationFailed {
        /// Description of the compilation failure.
        message: String,
    },

    /// Graph execution exceeded the maximum allowed iterations.
    #[error("graph execution exceeded max iterations: {max}")]
    MaxIterationsExceeded {
        /// The maximum iteration limit that was breached.
        max: usize,
    },

    /// Verification step determined the current state is unsatisfactory.
    #[error("verification failed: {message}")]
    VerificationFailed {
        /// Description of what verification detected.
        message: String,
    },

    /// Graph structure validation failed.
    #[error("graph validation failed: {message}")]
    GraphValidation {
        /// Description of the structural problem.
        message: String,
    },

    /// Topological sort detected a cycle in the graph.
    #[error("graph contains a cycle")]
    CycleDetected,
}
