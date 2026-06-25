//! Provider connection configuration.

use std::time::Duration;

use secrecy::SecretString;
use serde::{Deserialize, Serialize};

use crate::provider::{ModelName, ProviderId};

/// Identifies which concrete adapter implementation to use for a provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// OpenAI-compatible provider (chat + embeddings).
    #[cfg(feature = "openai")]
    OpenAi,

    /// Anthropic-compatible provider (chat only).
    #[cfg(feature = "anthropic")]
    Anthropic,
}

/// Configuration for a single model provider.
///
/// Supports `env:VAR_NAME` syntax for `api_key` to reference
/// environment variables instead of embedding secrets directly.
///
/// Builder methods (`with_*`) are available for ergonomic construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Which adapter implementation to use.
    #[serde(default)]
    pub provider_type: Option<ProviderType>,

    /// Provider API base URL.
    pub base_url: String,

    /// Default model name for this provider.
    #[serde(default)]
    pub model: Option<ModelName>,

    /// Known model names for this provider (model catalog).
    #[serde(default)]
    pub models: Vec<ModelName>,

    /// Model to use for compaction. Falls back to `model` when `None`.
    #[serde(default)]
    pub compaction_model: Option<ModelName>,

    /// API key or bearer token.
    ///
    /// Use `"env:MY_VAR"` to read from the environment variable `MY_VAR`.
    #[serde(default)]
    pub api_key: Option<String>,

    /// Organization, workspace, or tenant identifier.
    #[serde(default)]
    pub organization: Option<String>,

    /// End-to-end request timeout in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// TCP connection timeout in seconds.
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,

    /// Maximum retry attempts for retryable failures.
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
}

impl ProviderConfig {
    /// Creates a minimal provider config with only the base URL.
    ///
    /// All other fields (`provider_type`, `model`, `api_key`, etc.) are set
    /// to their defaults. Use the `with_*` builder methods to customize.
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            provider_type: None,
            base_url: base_url.into(),
            model: None,
            models: Vec::new(),
            compaction_model: None,
            api_key: None,
            organization: None,
            timeout_secs: default_timeout_secs(),
            connect_timeout_secs: default_connect_timeout_secs(),
            max_retries: default_max_retries(),
        }
    }

    /// Sets the provider adapter type (e.g. `ProviderType::OpenAi`).
    #[must_use]
    pub fn with_provider_type(mut self, provider_type: ProviderType) -> Self {
        self.provider_type = Some(provider_type);
        self
    }

    /// Sets the default model name for completion requests.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<ModelName>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the API key, optionally using `"env:VAR_NAME"` to defer to an
    /// environment variable at resolve time.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Resolves the API key, expanding `env:VAR_NAME` references.
    ///
    /// Returns `None` when no key is configured.
    #[must_use]
    pub fn resolve_api_key(&self) -> Option<SecretString> {
        self.api_key.as_ref().and_then(|key| {
            if let Some(var_name) = key.strip_prefix("env:") {
                std::env::var(var_name)
                    .ok()
                    .map(|s| SecretString::new(s.into_boxed_str()))
            } else {
                Some(SecretString::new(key.clone().into_boxed_str()))
            }
        })
    }

    /// Converts to the provider HTTP config used by adapters.
    #[must_use]
    pub fn to_http_config(&self, id: ProviderId) -> crate::provider::ProviderHttpConfig {
        crate::provider::ProviderHttpConfig {
            id,
            base_url: self.base_url.clone(),
            api_key: self.resolve_api_key(),
            organization: self.organization.clone(),
            timeout: Duration::from_secs(self.timeout_secs),
            connect_timeout: Duration::from_secs(self.connect_timeout_secs),
            max_retries: self.max_retries,
        }
    }
}

const fn default_timeout_secs() -> u64 {
    60
}

const fn default_connect_timeout_secs() -> u64 {
    10
}

const fn default_max_retries() -> usize {
    2
}
