//! Error types shared across the public API.

use std::time::Duration;

use thiserror::Error;

use crate::provider::ProviderId;

/// Crate-wide result type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Top-level crate error.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Error returned by a model or embedding provider.
    #[error(transparent)]
    Provider(#[from] ProviderError),

    /// Error from tool execution.
    #[error(transparent)]
    Tool(#[from] ToolError),

    /// Error from context construction.
    #[error(transparent)]
    Context(#[from] ContextError),

    /// Error from a storage backend.
    #[error(transparent)]
    Storage(#[from] StorageError),
}

/// Errors produced by provider implementations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProviderError {
    /// Provider credentials were rejected.
    #[error("authentication failed for provider `{provider}`")]
    Authentication {
        /// Provider that rejected authentication.
        provider: ProviderId,
    },

    /// Request payload was invalid for the target provider.
    #[error("provider `{provider}` rejected request: {message}")]
    BadRequest {
        /// Provider that rejected the request.
        provider: ProviderId,
        /// Provider-facing validation message.
        message: String,
    },

    /// Provider asked the caller to slow down.
    #[error("provider `{provider}` rate limited request")]
    RateLimited {
        /// Provider that rate limited the request.
        provider: ProviderId,
        /// Suggested delay before retrying.
        retry_after: Option<Duration>,
    },

    /// Provider call exceeded the configured timeout.
    #[error("provider `{provider}` timed out")]
    Timeout {
        /// Provider that timed out.
        provider: ProviderId,
    },

    /// Provider is temporarily overloaded.
    #[error("provider `{provider}` is overloaded")]
    Overloaded {
        /// Provider that reported overload.
        provider: ProviderId,
    },

    /// Requested provider feature is unavailable.
    #[error("provider `{provider}` does not support feature `{feature}`")]
    Unsupported {
        /// Provider that cannot serve the feature.
        provider: ProviderId,
        /// Feature name, such as `chat_stream` or `embedding`.
        feature: String,
    },

    /// Transport layer failed before a provider response was decoded.
    #[error("transport error for provider `{provider}`: {source}")]
    Transport {
        /// Provider reached by the transport.
        provider: ProviderId,
        /// Lower-level HTTP client error.
        #[source]
        source: reqwest::Error,
    },

    /// Provider response could not be decoded into the neutral schema.
    #[error("failed to decode response from provider `{provider}`: {message}")]
    Decode {
        /// Provider that returned an invalid response.
        provider: ProviderId,
        /// Decode failure summary.
        message: String,
    },

    /// Provider returned a structured error response.
    #[error("provider `{provider}` returned error: {message}")]
    Provider {
        /// Provider that returned the error.
        provider: ProviderId,
        /// Optional HTTP status or provider-specific status code.
        status: Option<u16>,
        /// Human-readable provider error message.
        message: String,
    },
}

impl ProviderError {
    /// Returns `true` when retrying the same request may succeed.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimited { .. }
                | Self::Timeout { .. }
                | Self::Overloaded { .. }
                | Self::Transport { .. }
        )
    }
}

/// Errors produced during tool execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ToolError {
    /// Requested tool is not registered.
    #[error("tool `{name}` is not registered")]
    NotFound {
        /// Name of the missing tool.
        name: String,
    },

    /// Tool execution failed with an application error.
    #[error("tool `{name}` failed: {message}")]
    Execution {
        /// Name of the tool that failed.
        name: String,
        /// Human-readable error message.
        message: String,
    },

    /// Tool arguments could not be parsed or validated.
    #[error("tool `{name}` received invalid arguments: {message}")]
    InvalidArguments {
        /// Name of the tool that rejected arguments.
        name: String,
        /// Validation or parsing failure description.
        message: String,
    },

    /// Tool is defined but not yet implemented.
    #[error("tool `{name}` is not implemented")]
    NotImplemented {
        /// Name of the unimplemented tool.
        name: String,
    },
}

/// Errors produced during context construction.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ContextError {
    /// Context adapter failed to produce its fragment.
    #[error("context adapter `{adapter}` failed: {message}")]
    AdapterFailed {
        /// Name of the adapter that failed.
        adapter: String,
        /// Human-readable error message.
        message: String,
    },

    /// Context input was invalid or incomplete.
    #[error("invalid context input: {message}")]
    InvalidInput {
        /// Description of the validation failure.
        message: String,
    },

    /// Required context adapter is not registered.
    #[error("context adapter `{adapter}` is not registered")]
    AdapterNotFound {
        /// Name of the missing adapter.
        adapter: String,
    },
}

/// Errors produced by storage backends.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StorageError {
    /// Requested entity was not found.
    #[error("storage entity not found: `{id}`")]
    NotFound {
        /// Identifier of the missing entity.
        id: String,
    },

    /// Failed to connect to the storage backend.
    #[error("storage connection failed ({backend}): {message}")]
    ConnectionFailed {
        /// Backend identifier (e.g., "postgres", "redis").
        backend: String,
        /// Human-readable failure description.
        message: String,
        /// Lower-level error, if available.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Serialization or deserialization failed.
    #[error("storage serialization failed: {message}")]
    SerializationFailed {
        /// Description of the serialization failure.
        message: String,
        /// Lower-level error, if available.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Backend-specific operational error.
    #[error("storage backend error ({backend}): {message}")]
    BackendError {
        /// Backend identifier.
        backend: String,
        /// Human-readable error description.
        message: String,
        /// Lower-level error, if available.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Schema migration failed.
    #[error("storage migration failed ({backend}): {message}")]
    MigrationFailed {
        /// Backend identifier.
        backend: String,
        /// Human-readable failure description.
        message: String,
        /// Lower-level error, if available.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Stored data is malformed or corrupted and cannot be interpreted.
    ///
    /// This is distinct from [`SerializationFailed`](StorageError::SerializationFailed),
    /// which indicates an operational error during encode/decode.
    /// `DataCorruption` means bytes were successfully read from storage
    /// but do not form valid data — an integrity error.
    #[error("data corruption in `{field}`: {message}")]
    DataCorruption {
        /// Name of the corrupted field (e.g., "session.id", "message.metadata").
        field: String,
        /// Human-readable description of the corruption.
        message: String,
        /// Underlying parse or deserialization error, if available.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}
