//! Shared HTTP client construction and request helpers.

use reqwest::{Client, ClientBuilder, RequestBuilder, StatusCode};
use secrecy::ExposeSecret;

use crate::error::ProviderError;
use crate::provider::ProviderHttpConfig;
use crate::provider::ProviderId;

/// Builds a configured [`reqwest::Client`] from provider HTTP settings.
///
/// # Errors
///
/// Returns [`ProviderError::Transport`] when the underlying client builder fails.
pub(crate) fn build_client(config: &ProviderHttpConfig) -> Result<Client, ProviderError> {
    ClientBuilder::new()
        .timeout(config.timeout)
        .connect_timeout(config.connect_timeout)
        .build()
        .map_err(|source| ProviderError::Transport {
            provider: config.id.clone(),
            source,
        })
}

/// Applies a bearer token authorization header when an API key is configured.
pub(crate) fn with_bearer_auth(builder: RequestBuilder, config: &ProviderHttpConfig) -> RequestBuilder {
    match &config.api_key {
        Some(key) => {
            let header_value = format!("Bearer {}", key.expose_secret());
            builder.header("Authorization", header_value)
        }
        None => builder,
    }
}

/// Maps an HTTP response status code to a [`ProviderError`].
pub(crate) fn status_to_error(
    provider: &ProviderId,
    status: StatusCode,
    body_text: &str,
) -> ProviderError {
    match status.as_u16() {
        401 | 403 => ProviderError::Authentication {
            provider: provider.clone(),
        },
        400 => ProviderError::BadRequest {
            provider: provider.clone(),
            message: truncate_body(body_text),
        },
        429 => ProviderError::RateLimited {
            provider: provider.clone(),
            retry_after: None,
        },
        500..=599 => ProviderError::Overloaded {
            provider: provider.clone(),
        },
        _ => ProviderError::Provider {
            provider: provider.clone(),
            status: Some(status.as_u16()),
            message: truncate_body(body_text),
        },
    }
}

fn truncate_body(body: &str) -> String {
    if body.len() > 512 {
        format!("{}...", &body[..512])
    } else {
        body.to_owned()
    }
}
