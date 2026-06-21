//! Shared configuration for HTTP-backed provider clients.

use std::time::Duration;

use secrecy::SecretString;

use crate::provider::ProviderId;

/// Default request timeout for provider calls.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Default TCP connection timeout for provider calls.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// HTTP connection settings common to model providers.
#[derive(Debug, Clone)]
pub struct ProviderHttpConfig {
    /// Logical provider identifier used by registries and errors.
    pub id: ProviderId,
    /// Provider API base URL.
    pub base_url: String,
    /// Optional API key or bearer token.
    pub api_key: Option<SecretString>,
    /// Optional organization, workspace, or tenant identifier.
    pub organization: Option<String>,
    /// End-to-end request timeout.
    pub timeout: Duration,
    /// TCP connection timeout.
    pub connect_timeout: Duration,
    /// Maximum retry attempts for retryable failures.
    pub max_retries: usize,
}

impl ProviderHttpConfig {
    /// Creates a configuration with production-safe timeout defaults.
    #[must_use]
    pub fn new(id: ProviderId, base_url: impl Into<String>) -> Self {
        Self {
            id,
            base_url: base_url.into(),
            api_key: None,
            organization: None,
            timeout: DEFAULT_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            max_retries: 2,
        }
    }

    /// Sets the provider API key.
    #[must_use]
    pub fn with_api_key(mut self, api_key: SecretString) -> Self {
        self.api_key = Some(api_key);
        self
    }

    /// Sets the provider organization or tenant identifier.
    #[must_use]
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Sets request and connection timeouts.
    #[must_use]
    pub fn with_timeouts(mut self, timeout: Duration, connect_timeout: Duration) -> Self {
        self.timeout = timeout;
        self.connect_timeout = connect_timeout;
        self
    }

    /// Sets the maximum retry count for retryable failures.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }
}
