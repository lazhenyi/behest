//! Shared HTTP client construction and request helpers.

use std::time::Duration;

use reqwest::{Client, ClientBuilder, RequestBuilder, StatusCode};
use secrecy::ExposeSecret;

use crate::error::ProviderError;
use crate::provider::ProviderHttpConfig;
use crate::provider::ProviderId;

/// Builds a configured [`reqwest::Client`] from provider HTTP settings.
///
/// Applies the timeout and connect timeout from the config to the client.
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

/// Applies a `Bearer` authorization header from the provider API key, if set.
///
/// Returns the builder unchanged when no API key is configured.
pub(crate) fn with_bearer_auth(
    builder: RequestBuilder,
    config: &ProviderHttpConfig,
) -> RequestBuilder {
    match &config.api_key {
        Some(key) => {
            let header_value = format!("Bearer {}", key.expose_secret());
            builder.header("Authorization", header_value)
        }
        None => builder,
    }
}

/// Maps an HTTP response status code to a [`ProviderError`].
///
/// The `retry_after_secs` parameter is extracted from the `Retry-After` response
/// header by the caller and passed explicitly because this function operates on
/// body text only.
///
/// The body text is truncated to 512 characters to avoid including large
/// payloads in error messages.
pub(crate) fn status_to_error(
    provider: &ProviderId,
    status: StatusCode,
    body_text: &str,
    retry_after_secs: Option<u64>,
) -> ProviderError {
    let retry_after = retry_after_secs.map(Duration::from_secs);
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
            retry_after,
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

/// Extracts the `Retry-After` header value as seconds, if present.
///
/// Supports the integer form (e.g. `120`). The HTTP-date form is not
/// yet parsed and returns `None`.
pub(crate) fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
}

/// Truncates a string to at most 512 characters, preserving UTF-8 boundaries.
///
/// Appends `"..."` when the input exceeds the limit.
fn truncate_body(body: &str) -> String {
    if body.len() <= 512 {
        return body.to_owned();
    }
    let mut end = 512;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &body[..end])
}
