//! Anthropic chat provider adapter implementing [`ChatProvider`].

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt as _;
use reqwest::Client;
use secrecy::ExposeSecret;
use tokio::sync::Mutex;

use crate::adapt::http::{build_client, parse_retry_after, status_to_error};
use crate::adapt::sse::SseStream;
use crate::error::ProviderError;
use crate::provider::{
    ChatProvider, ChatRequest, ChatResponse, ChatStream, ChatStreamEvent, FinishReason,
    ProviderCapabilities, ProviderHttpConfig, ProviderId, ProviderResult, TokenUsage, ToolCall,
};

use super::API_VERSION;
use super::convert::{from_anthropic_response, to_anthropic_request};
use super::types::{
    AnthropicContentBlock, AnthropicDelta, AnthropicResponse, AnthropicStreamEvent,
};

/// Anthropic Claude chat completion adapter.
///
/// Implements [`ChatProvider`] for Anthropic's `/v1/messages` endpoint.
/// Supports streaming, tool calling, and vision. Embeddings are not supported
/// by the Anthropic API.
///
/// # Authentication
///
/// The API key is sent via the `x-api-key` header. Configure it through the
/// [`ProviderHttpConfig`] passed to [`new`](Self::new).
pub struct AnthropicChatAdapter {
    id: ProviderId,
    client: Client,
    config: ProviderHttpConfig,
}

impl AnthropicChatAdapter {
    /// Creates an Anthropic chat adapter with a new HTTP client.
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

    /// Creates an Anthropic chat adapter reusing an existing HTTP client.
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
        format!("{}/messages", self.config.base_url)
    }

    fn build_request(&self, body: &impl serde::Serialize) -> reqwest::RequestBuilder {
        let mut builder = self
            .client
            .post(self.url())
            .header("anthropic-version", API_VERSION)
            .json(body);

        if let Some(key) = &self.config.api_key {
            builder = builder.header("x-api-key", key.expose_secret());
        }

        builder
    }

    fn wrap_transport(&self, source: reqwest::Error) -> ProviderError {
        crate::adapt::http::wrap_transport(&self.id, source)
    }
}

#[async_trait]
impl ChatProvider for AnthropicChatAdapter {
    fn id(&self) -> ProviderId {
        self.id.clone()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            chat: true,
            chat_stream: true,
            tool_calling: true,
            parallel_tool_calls: false,
            vision: true,
            ..ProviderCapabilities::empty()
        }
    }

    async fn complete(&self, request: ChatRequest) -> ProviderResult<ChatResponse> {
        let body = to_anthropic_request(&request, false);
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

        let parsed: AnthropicResponse =
            response.json().await.map_err(|e| ProviderError::Decode {
                provider: self.id.clone(),
                message: e.to_string(),
            })?;

        Ok(from_anthropic_response(&self.id, &parsed))
    }

    async fn stream(&self, request: ChatRequest) -> ProviderResult<ChatStream> {
        let body = to_anthropic_request(&request, true);
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

        let state = Arc::new(Mutex::new(StreamState::new(provider_id.clone())));
        let mapped = sse_stream.filter_map(move |event| {
            let state = Arc::clone(&state);
            async move {
                let mut st = state.lock().await;
                map_anthropic_event(&mut st, event)
            }
        });

        let combined = futures_util::stream::once(async { Ok(started) }).chain(mapped);
        Ok(Box::pin(combined))
    }
}

/// Accumulated state for mapping Anthropic SSE events to [`ChatStreamEvent`].
///
/// Tracks the model name, in-progress tool calls, and their partial JSON
/// argument buffers across multiple stream chunks.
struct StreamState {
    provider: ProviderId,
    model: Option<String>,
    tool_calls: Vec<ToolCallState>,
}

/// Accumulator for one in-progress tool call during streaming.
struct ToolCallState {
    id: String,
    name: String,
    arguments: String,
}

impl StreamState {
    fn new(provider: ProviderId) -> Self {
        Self {
            provider,
            model: None,
            tool_calls: Vec::new(),
        }
    }
}

/// Maps one Anthropic SSE event to a [`ChatStreamEvent`], updating stream state.
///
/// Returns `None` for intermediate events that do not produce user-facing
/// deltas (e.g. `message_start`, `content_block_stop` without tool calls).
fn map_anthropic_event(
    state: &mut StreamState,
    event: Result<crate::adapt::sse::SseEvent, ProviderError>,
) -> Option<Result<ChatStreamEvent, ProviderError>> {
    let sse = match event {
        Err(e) => return Some(Err(e)),
        Ok(e) => e,
    };

    let parsed: AnthropicStreamEvent = match serde_json::from_str(&sse.data) {
        Ok(p) => p,
        Err(e) => {
            return Some(Err(ProviderError::Decode {
                provider: state.provider.clone(),
                message: e.to_string(),
            }));
        }
    };

    match parsed {
        AnthropicStreamEvent::MessageStart { message } => {
            state.model = Some(message.model);
            None
        }
        AnthropicStreamEvent::ContentBlockStart {
            index,
            content_block,
        } => handle_block_start(state, index, content_block),
        AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
            handle_block_delta(state, index, delta)
        }
        AnthropicStreamEvent::ContentBlockStop { index } => handle_block_stop(state, index),
        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            let reason = delta.stop_reason.as_deref();
            let finish = convert_stream_stop_reason(reason);
            Some(Ok(ChatStreamEvent::Finished {
                finish_reason: finish,
                usage: usage.map(|u| TokenUsage::new(u.input_tokens, u.output_tokens)),
            }))
        }
        AnthropicStreamEvent::MessageStop | AnthropicStreamEvent::Other => None,
    }
}

/// Handles a `content_block_start` event, emitting [`ChatStreamEvent::ToolCallStarted`]
/// when the new block is a tool use invocation.
fn handle_block_start(
    state: &mut StreamState,
    _index: usize,
    block: AnthropicContentBlock,
) -> Option<Result<ChatStreamEvent, ProviderError>> {
    match block {
        AnthropicContentBlock::ToolUse { id, name, .. } => {
            state.tool_calls.push(ToolCallState {
                id: id.clone(),
                name: name.clone(),
                arguments: String::new(),
            });
            Some(Ok(ChatStreamEvent::ToolCallStarted { id, name }))
        }
        _ => None,
    }
}

/// Handles a `content_block_delta` event, emitting text deltas or tool call
/// argument deltas as appropriate.
fn handle_block_delta(
    state: &mut StreamState,
    index: usize,
    delta: AnthropicDelta,
) -> Option<Result<ChatStreamEvent, ProviderError>> {
    match delta {
        AnthropicDelta::TextDelta { text } => {
            if text.is_empty() {
                None
            } else {
                Some(Ok(ChatStreamEvent::TextDelta { delta: text }))
            }
        }
        AnthropicDelta::InputJsonDelta { partial_json } => {
            if let Some(tc) = state.tool_calls.get_mut(index) {
                tc.arguments.push_str(&partial_json);
            }
            let call_id = state
                .tool_calls
                .get(index)
                .map_or_else(|| format!("call_{index}"), |tc| tc.id.clone());
            Some(Ok(ChatStreamEvent::ToolCallArgumentsDelta {
                id: call_id,
                delta: partial_json,
            }))
        }
        AnthropicDelta::Other => None,
    }
}

/// Handles a `content_block_stop` event, emitting [`ChatStreamEvent::ToolCallCompleted`]
/// when a tool use block completes.
///
/// Attempts to parse the accumulated JSON arguments. Falls back to `null` on
/// parse failure.
fn handle_block_stop(
    state: &mut StreamState,
    index: usize,
) -> Option<Result<ChatStreamEvent, ProviderError>> {
    let tc = state.tool_calls.get(index)?;
    let arguments = serde_json::from_str(&tc.arguments).unwrap_or_else(|e| {
        tracing::warn!(
            tool_name = %tc.name,
            error = %e,
            "failed to parse tool call arguments, falling back to null"
        );
        serde_json::Value::Null
    });
    Some(Ok(ChatStreamEvent::ToolCallCompleted {
        call: ToolCall::new(tc.id.clone(), tc.name.clone(), arguments),
    }))
}

/// Converts an Anthropic stop reason string to the neutral [`FinishReason`].
///
/// Maps `"end_turn"` / `"stop_sequence"` → [`FinishReason::Stop`],
/// `"tool_use"` → [`FinishReason::ToolCalls`],
/// `"max_tokens"` → [`FinishReason::Length`],
/// and anything else → [`FinishReason::Unknown`].
fn convert_stream_stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn" | "stop_sequence") => FinishReason::Stop,
        Some("tool_use") => FinishReason::ToolCalls,
        Some("max_tokens") => FinishReason::Length,
        Some(other) => FinishReason::Unknown(other.to_owned()),
        None => FinishReason::Unknown("null".to_owned()),
    }
}
