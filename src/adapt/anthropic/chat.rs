//! Anthropic chat provider adapter implementing [`ChatProvider`].

use async_trait::async_trait;
use futures_util::StreamExt as _;
use reqwest::Client;
use secrecy::ExposeSecret;

use crate::adapt::http::{build_client, status_to_error};
use crate::adapt::sse::SseStream;
use crate::error::ProviderError;
use crate::provider::{
    ChatProvider, ChatRequest, ChatResponse, ChatStream, ChatStreamEvent, FinishReason,
    ProviderCapabilities, ProviderHttpConfig, ProviderId, ProviderResult, TokenUsage, ToolCall,
};

use super::convert::{from_anthropic_response, to_anthropic_request};
use super::types::{AnthropicContentBlock, AnthropicDelta, AnthropicResponse, AnthropicStreamEvent};
use super::API_VERSION;

/// Anthropic Claude chat completion adapter.
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
        let response = self.build_request(&body).send().await.map_err(wrap_transport)?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(status_to_error(&self.id, status, &text));
        }

        let parsed: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Decode {
                provider: self.id.clone(),
                message: e.to_string(),
            })?;

        Ok(from_anthropic_response(&self.id, &parsed))
    }

    async fn stream(&self, request: ChatRequest) -> ProviderResult<ChatStream> {
        let body = to_anthropic_request(&request, true);
        let response = self.build_request(&body).send().await.map_err(wrap_transport)?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(status_to_error(&self.id, status, &text));
        }

        let byte_stream = response.bytes_stream();
        let sse_stream = SseStream::new(byte_stream, self.id.clone());

        let model = request.model.clone();
        let provider_id = self.id.clone();
        let started = ChatStreamEvent::Started {
            provider: provider_id.clone(),
            model: model.clone(),
        };

        let state = StreamState::new(provider_id.clone());
        let mapped = sse_stream.filter_map(move |event| {
            let mut st = state.clone();
            async move { map_anthropic_event(&mut st, event) }
        });

        let combined = futures_util::stream::once(async { Ok(started) }).chain(mapped);
        Ok(Box::pin(combined))
    }
}

#[derive(Clone)]
struct StreamState {
    provider: ProviderId,
    model: Option<String>,
    tool_calls: Vec<ToolCallState>,
}

#[derive(Clone)]
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
        AnthropicStreamEvent::MessageStop => None,
        AnthropicStreamEvent::Other => None,
    }
}

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
                .map(|tc| tc.id.clone())
                .unwrap_or_else(|| format!("call_{index}"));
            Some(Ok(ChatStreamEvent::ToolCallArgumentsDelta {
                id: call_id,
                delta: partial_json,
            }))
        }
        AnthropicDelta::Other => None,
    }
}

fn handle_block_stop(
    state: &mut StreamState,
    index: usize,
) -> Option<Result<ChatStreamEvent, ProviderError>> {
    let tc = state.tool_calls.get(index)?;
    let arguments = serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);
    Some(Ok(ChatStreamEvent::ToolCallCompleted {
        call: ToolCall::new(tc.id.clone(), tc.name.clone(), arguments),
    }))
}

fn convert_stream_stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn") => FinishReason::Stop,
        Some("tool_use") => FinishReason::ToolCalls,
        Some("max_tokens") => FinishReason::Length,
        Some("stop_sequence") => FinishReason::Stop,
        Some(other) => FinishReason::Unknown(other.to_owned()),
        None => FinishReason::Unknown("null".to_owned()),
    }
}

fn wrap_transport(source: reqwest::Error) -> ProviderError {
    if source.is_timeout() {
        ProviderError::Timeout {
            provider: ProviderId::new("anthropic"),
        }
    } else {
        ProviderError::Transport {
            provider: ProviderId::new("anthropic"),
            source,
        }
    }
}
