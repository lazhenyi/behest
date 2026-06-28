//! OpenAI embedding provider adapter implementing [`EmbeddingProvider`].

use async_trait::async_trait;
use reqwest::Client;

use crate::adapt::http::{build_client, parse_retry_after, status_to_error, with_bearer_auth};
use crate::error::ProviderError;
use crate::provider::{
    Embedding, EmbeddingInput, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, ModelName,
    ProviderCapabilities, ProviderHttpConfig, ProviderId, ProviderResult, TokenUsage,
};

use super::types::{OpenAiEmbeddingRequest, OpenAiEmbeddingResponse};

/// OpenAI-compatible embedding adapter.
///
/// Implements [`EmbeddingProvider`] for OpenAI's `/v1/embeddings` endpoint.
/// Supports multiple input texts and optional dimension reduction.
/// Works with OpenAI, Azure OpenAI, and any OpenAI-compatible API endpoint.
///
/// # Authentication
///
/// The API key is sent via the `Authorization: Bearer` header. Configure it
/// through the [`ProviderHttpConfig`] passed to [`new`](Self::new).
pub struct OpenAiEmbeddingAdapter {
    id: ProviderId,
    client: Client,
    config: ProviderHttpConfig,
}

impl OpenAiEmbeddingAdapter {
    /// Creates an OpenAI embedding adapter with a new HTTP client.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError::Transport`] when the HTTP client cannot be built.
    pub fn new(config: ProviderHttpConfig) -> Result<Self, ProviderError> {
        let client = build_client(&config)?;
        Ok(Self {
            id: config.id.clone(),
            client,
            config,
        })
    }

    /// Creates an OpenAI embedding adapter reusing an existing HTTP client.
    ///
    /// Useful when multiple adapters share the same connection pool or custom
    /// TLS configuration.
    ///
    /// # Parameters
    ///
    /// * `config` — Provider HTTP configuration including API key and base URL.
    /// * `client` — A pre-built [`reqwest::Client`] to use for all requests.
    #[must_use]
    pub fn with_client(config: ProviderHttpConfig, client: Client) -> Self {
        Self {
            id: config.id.clone(),
            client,
            config,
        }
    }

    fn url(&self) -> String {
        format!("{}/embeddings", self.config.base_url)
    }

    fn wrap_transport(&self, source: reqwest::Error) -> ProviderError {
        crate::adapt::http::wrap_transport(&self.id, source)
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbeddingAdapter {
    fn id(&self) -> ProviderId {
        self.id.clone()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::embeddings()
    }

    async fn embed(&self, request: EmbeddingRequest) -> ProviderResult<EmbeddingResponse> {
        let input_texts = extract_texts(&request.input);
        let body = OpenAiEmbeddingRequest {
            model: request.model.as_str().to_owned(),
            input: input_texts,
            dimensions: request.dimensions,
        };

        let builder = self.client.post(self.url()).json(&body);
        let builder = with_bearer_auth(builder, &self.config);
        let response = builder.send().await.map_err(|e| self.wrap_transport(e))?;

        if !response.status().is_success() {
            let status = response.status();
            let retry_after = parse_retry_after(response.headers());
            let text = response
                .text()
                .await
                .unwrap_or_else(|e| format!("<failed to read error body: {e}>"));
            return Err(status_to_error(&self.id, status, &text, retry_after));
        }

        let parsed: OpenAiEmbeddingResponse =
            response.json().await.map_err(|e| ProviderError::Decode {
                provider: self.id.clone(),
                message: e.to_string(),
            })?;

        Ok(from_response(&self.id, &request.model, &parsed))
    }
}

/// Extracts plain text from [`EmbeddingInput`] items.
///
/// Token-based inputs are serialized as space-separated token strings.
fn extract_texts(inputs: &[EmbeddingInput]) -> Vec<String> {
    inputs
        .iter()
        .map(|input| match input {
            EmbeddingInput::Text { text } => text.clone(),
            EmbeddingInput::Tokens { tokens } => tokens
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" "),
        })
        .collect()
}

/// Converts an [`OpenAiEmbeddingResponse`] to a neutral [`EmbeddingResponse`].
///
/// Maps each embedding data item by index and sets output tokens to 0 since
/// embedding requests only consume input tokens.
fn from_response(
    provider: &ProviderId,
    model: &ModelName,
    response: &OpenAiEmbeddingResponse,
) -> EmbeddingResponse {
    let embeddings = response
        .data
        .iter()
        .map(|d| Embedding::new(d.index, d.embedding.clone()))
        .collect();

    // Embeddings only consume input tokens; output_tokens is always 0.
    let usage = response
        .usage
        .as_ref()
        .map(|u| TokenUsage::new(u.prompt_tokens, 0));

    EmbeddingResponse {
        provider: provider.clone(),
        model: model.clone(),
        embeddings,
        usage,
        raw: None,
    }
}
