//! OpenAI chat provider adapter implementing [`ChatProvider`].

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt as _;
use reqwest::Client;
use tokio::sync::Mutex;

use crate::adapt::http::{build_client, parse_retry_after, status_to_error, with_bearer_auth};
use crate::adapt::sse::{SseEvent, SseStream};
use crate::error::ProviderError;
use crate::provider::{
    ChatProvider, ChatRequest, ChatResponse, ChatStream, ChatStreamEvent, FinishReason,
    ProviderCapabilities, ProviderHttpConfig, ProviderId, ProviderResult, TokenUsage,
};

use super::convert::{from_openai_response, to_openai_request};
use super::types::{OpenAiChatResponse, OpenAiStreamChunk, OpenAiToolCall};

/// OpenAI-compatible chat completion adapter.
///
/// Implements [`ChatProvider`] for OpenAI's `/v1/chat/completions` endpoint.
/// Supports streaming, tool calling (including parallel), structured output
/// (JSON schema), and vision. Works with OpenAI, Azure OpenAI, and any
/// OpenAI-compatible API endpoint.
///
/// # Authentication
///
/// The API key is sent via the `Authorization: Bearer` header. Configure it
/// through the [`ProviderHttpConfig`] passed to [`new`](Self::new).
pub struct OpenAiChatAdapter {
    id: ProviderId,
    client: Client,
    config: ProviderHttpConfig,
}

impl OpenAiChatAdapter {
    /// Creates an OpenAI chat adapter with a new HTTP client.
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

    /// Creates an OpenAI chat adapter reusing an existing HTTP client.
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
        format!("{}/chat/completions", self.config.base_url)
    }

    fn build_request(&self, body: &impl serde::Serialize) -> reqwest::RequestBuilder {
        let builder = self.client.post(self.url()).json(body);
        with_bearer_auth(builder, &self.config)
    }

    fn wrap_transport(&self, source: reqwest::Error) -> ProviderError {
        crate::adapt::http::wrap_transport(&self.id, source)
    }
}

#[async_trait]
impl ChatProvider for OpenAiChatAdapter {
    fn id(&self) -> ProviderId {
        self.id.clone()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            chat: true,
            chat_stream: true,
            tool_calling: true,
            parallel_tool_calls: true,
            json_schema_output: true,
            vision: true,
            ..ProviderCapabilities::empty()
        }
    }

    async fn complete(&self, request: ChatRequest) -> ProviderResult<ChatResponse> {
        let body = to_openai_request(&request, false);
        let response = self
            .build_request(&body)
            .send()
            .await
            .map_err(|e| self.wrap_transport(e))?;

        if !response.status().is_success() {
            let status = response.status();
            let retry_after = parse_retry_after(response.headers());
            let text = response
                .text()
                .await
                .unwrap_or_else(|e| format!("<failed to read error body: {e}>"));
            return Err(status_to_error(&self.id, status, &text, retry_after));
        }

        let parsed: OpenAiChatResponse = response
            .json()
            .await
            .map_err(|e| decode_error(&self.id, &e))?;

        from_openai_response(&self.id, &parsed).ok_or_else(|| ProviderError::Decode {
            provider: self.id.clone(),
            message: "empty choices in response".to_owned(),
        })
    }

    async fn stream(&self, request: ChatRequest) -> ProviderResult<ChatStream> {
        let body = to_openai_request(&request, true);
        let response = self
            .build_request(&body)
            .send()
            .await
            .map_err(|e| self.wrap_transport(e))?;

        if !response.status().is_success() {
            let status = response.status();
            let retry_after = parse_retry_after(response.headers());
            let text = response
                .text()
                .await
                .unwrap_or_else(|e| format!("<failed to read error body: {e}>"));
            return Err(status_to_error(&self.id, status, &text, retry_after));
        }

        let byte_stream = response.bytes_stream();
        let sse_stream = SseStream::new(byte_stream, self.id.clone());

        let model = request.model.clone();
        let provider_id = self.id.clone();
        let started = ChatStreamEvent::Started {
            provider: provider_id.clone(),
            model: model.clone(),
        };

        let state = Arc::new(Mutex::new(OpenAiStreamState::new(provider_id.clone())));
        let mapped = sse_stream.filter_map(move |event| {
            let state = Arc::clone(&state);
            async move {
                let mut st = state.lock().await;
                map_sse_event(&mut st, event)
            }
        });

        let combined = futures_util::stream::once(async { Ok(started) }).chain(mapped);
        Ok(Box::pin(combined))
    }
}

/// Accumulated state for mapping OpenAI SSE deltas to [`ChatStreamEvent`].
///
/// Tracks tool call identifiers, names, and partial JSON argument buffers
/// indexed by the stream choice index.
struct OpenAiStreamState {
    provider: ProviderId,
    index_to_id: HashMap<usize, String>,
    index_to_name: HashMap<usize, String>,
    index_to_args: HashMap<usize, String>,
}

impl OpenAiStreamState {
    fn new(provider: ProviderId) -> Self {
        Self {
            provider,
            index_to_id: HashMap::new(),
            index_to_name: HashMap::new(),
            index_to_args: HashMap::new(),
        }
    }
}

/// Maps one raw SSE event from an OpenAI stream to a [`ChatStreamEvent`].
///
/// Filters out `[DONE]` terminal events and delegates chunk parsing to
/// [`parse_chunk_event`].
fn map_sse_event(
    state: &mut OpenAiStreamState,
    event: Result<SseEvent, ProviderError>,
) -> Option<Result<ChatStreamEvent, ProviderError>> {
    match event {
        Err(e) => Some(Err(e)),
        Ok(sse) if sse.is_openai_done() => None,
        Ok(sse) => parse_chunk_event(state, &sse.data),
    }
}

/// Parses one SSE data line as an [`OpenAiStreamChunk`] and emits the
/// corresponding [`ChatStreamEvent`].
///
/// Handles text deltas, tool call deltas (start, arguments, completion),
/// and finish signals.
fn parse_chunk_event(
    state: &mut OpenAiStreamState,
    data: &str,
) -> Option<Result<ChatStreamEvent, ProviderError>> {
    let chunk: OpenAiStreamChunk = match serde_json::from_str(data) {
        Ok(c) => c,
        Err(e) => {
            return Some(Err(ProviderError::Decode {
                provider: state.provider.clone(),
                message: e.to_string(),
            }));
        }
    };

    let choice = chunk.choices.first()?;

    if let Some(reason) = &choice.finish_reason {
        let finished = ChatStreamEvent::Finished {
            finish_reason: convert_stream_finish(reason),
            usage: chunk
                .usage
                .as_ref()
                .map(|u| TokenUsage::new(u.prompt_tokens, u.completion_tokens)),
        };
        return Some(Ok(finished));
    }

    if let Some(text) = &choice.delta.content
        && !text.is_empty()
    {
        return Some(Ok(ChatStreamEvent::TextDelta {
            delta: text.clone(),
        }));
    }

    if let Some(calls) = &choice.delta.tool_calls {
        return convert_tool_call_deltas(state, calls);
    }

    None
}

/// Converts an OpenAI stream finish reason string to the neutral [`FinishReason`].
fn convert_stream_finish(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "tool_calls" => FinishReason::ToolCalls,
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Unknown(other.to_owned()),
    }
}

/// Converts OpenAI tool call stream deltas into [`ChatStreamEvent`]s.
///
/// Tracks tool call identity (id, name) and argument accumulation across
/// multiple chunks. Emits [`ChatStreamEvent::ToolCallStarted`] when both
/// id and name are available, and [`ChatStreamEvent::ToolCallArgumentsDelta`]
/// for each argument fragment.
fn convert_tool_call_deltas(
    state: &mut OpenAiStreamState,
    calls: &[OpenAiToolCall],
) -> Option<Result<ChatStreamEvent, ProviderError>> {
    let call = calls.first()?;
    let index = call.index.unwrap_or(0);

    if let Some(id) = &call.id {
        state.index_to_id.insert(index, id.clone());
    }
    if let Some(name) = &call.function.name {
        state.index_to_name.insert(index, name.clone());
    }

    if let Some(id) = call.id.as_deref()
        && let Some(name) = call.function.name.as_deref()
        && !id.is_empty()
        && !name.is_empty()
    {
        return Some(Ok(ChatStreamEvent::ToolCallStarted {
            id: id.to_owned(),
            name: name.to_owned(),
        }));
    }

    if let Some(args) = &call.function.arguments {
        state.index_to_args.entry(index).or_default().push_str(args);

        let call_id = state
            .index_to_id
            .get(&index)
            .cloned()
            .unwrap_or_else(|| format!("call_{index}"));

        return Some(Ok(ChatStreamEvent::ToolCallArgumentsDelta {
            id: call_id,
            delta: args.clone(),
        }));
    }

    None
}

/// Wraps a [`reqwest::Error`] from JSON deserialization into [`ProviderError::Decode`].
fn decode_error(provider_id: &ProviderId, error: &reqwest::Error) -> ProviderError {
    ProviderError::Decode {
        provider: provider_id.clone(),
        message: error.to_string(),
    }
}
