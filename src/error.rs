//! Error types shared across the public API.

use std::time::Duration;

use thiserror::Error;

use crate::provider::ProviderId;

/// Crate-wide result type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Top-level crate error.
#[derive(Debug, Error)]
pub enum Error {
    /// Error returned by a model or embedding provider.
    #[error(transparent)]
    Provider(#[from] ProviderError),
}

/// Errors produced by provider implementations.
#[derive(Debug, Error)]
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
