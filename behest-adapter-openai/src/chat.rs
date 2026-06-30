//! OpenAI chat provider adapter implementing [`ChatProvider`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt as _;
use reqwest::Client;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::http::{build_client, parse_retry_after, status_to_error, with_bearer_auth};
use crate::sse::{SseEvent, SseStream};
use behest_core::error::ProviderError;
use behest_provider::{
    ChatProvider, ChatRequest, ChatResponse, ChatStream, ChatStreamEvent, FinishReason,
    ProviderCapabilities, ProviderHttpConfig, ProviderId, ProviderResult, TokenUsage, ToolCall,
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
        crate::http::wrap_transport(&self.id, source)
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
        let mapped = sse_stream
            .then(move |event| {
                let state = Arc::clone(&state);
                async move {
                    let mut st = state.lock().await;
                    map_sse_event(&mut st, event)
                }
            })
            .flat_map(futures_util::stream::iter);

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
    started_indices: HashSet<usize>,
}

impl OpenAiStreamState {
    fn new(provider: ProviderId) -> Self {
        Self {
            provider,
            index_to_id: HashMap::new(),
            index_to_name: HashMap::new(),
            index_to_args: HashMap::new(),
            started_indices: HashSet::new(),
        }
    }

    fn completed_tool_calls(&mut self) -> Vec<ToolCall> {
        let mut indices: Vec<usize> = self
            .index_to_id
            .keys()
            .chain(self.index_to_name.keys())
            .chain(self.index_to_args.keys())
            .copied()
            .collect();
        indices.sort_unstable();
        indices.dedup();

        let mut calls = Vec::new();
        for index in indices {
            let Some(name) = self.index_to_name.get(&index).cloned() else {
                continue;
            };
            let id = self
                .index_to_id
                .get(&index)
                .cloned()
                .unwrap_or_else(|| format!("call_{index}"));
            let arguments = self
                .index_to_args
                .get(&index)
                .map_or(Value::Null, |raw| parse_tool_arguments(&name, raw));
            calls.push(ToolCall::new(id, name, arguments));
        }

        self.index_to_id.clear();
        self.index_to_name.clear();
        self.index_to_args.clear();
        self.started_indices.clear();

        calls
    }
}

/// Maps one raw SSE event from an OpenAI stream to a [`ChatStreamEvent`].
///
/// Filters out `[DONE]` terminal events and delegates chunk parsing to
/// [`parse_chunk_event`].
fn map_sse_event(
    state: &mut OpenAiStreamState,
    event: Result<SseEvent, ProviderError>,
) -> Vec<Result<ChatStreamEvent, ProviderError>> {
    match event {
        Err(e) => vec![Err(e)],
        Ok(sse) if sse.is_openai_done() => Vec::new(),
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
) -> Vec<Result<ChatStreamEvent, ProviderError>> {
    let chunk: OpenAiStreamChunk = match serde_json::from_str(data) {
        Ok(c) => c,
        Err(e) => {
            return vec![Err(ProviderError::Decode {
                provider: state.provider.clone(),
                message: e.to_string(),
            })];
        }
    };

    let Some(choice) = chunk.choices.first() else {
        return Vec::new();
    };

    let mut events = Vec::new();

    if let Some(reason) = &choice.finish_reason {
        let finish_reason = convert_stream_finish(reason);
        if matches!(finish_reason, FinishReason::ToolCalls) {
            events.extend(state.completed_tool_calls().into_iter().map(|call| {
                Ok(ChatStreamEvent::ToolCallCompleted {
                    call,
                })
            }));
        }
        events.push(Ok(ChatStreamEvent::Finished {
            finish_reason,
            usage: chunk
                .usage
                .as_ref()
                .map(|u| TokenUsage::new(u.prompt_tokens, u.completion_tokens)),
        }));
        return events;
    }

    if let Some(text) = &choice.delta.content
        && !text.is_empty()
    {
        events.push(Ok(ChatStreamEvent::TextDelta {
            delta: text.clone(),
        }));
    }

    if let Some(calls) = &choice.delta.tool_calls {
        events.extend(convert_tool_call_deltas(state, calls));
    }

    events
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
) -> Vec<Result<ChatStreamEvent, ProviderError>> {
    let mut events = Vec::new();

    for call in calls {
        let index = call.index.unwrap_or(0);

        if let Some(id) = &call.id {
            state.index_to_id.insert(index, id.clone());
        }
        if let Some(name) = &call.function.name {
            state.index_to_name.insert(index, name.clone());
        }

        if let (Some(id), Some(name)) = (
            state.index_to_id.get(&index),
            state.index_to_name.get(&index),
        ) && !id.is_empty()
            && !name.is_empty()
            && state.started_indices.insert(index)
        {
            events.push(Ok(ChatStreamEvent::ToolCallStarted {
                id: id.clone(),
                name: name.clone(),
            }));
        }

        if let Some(args) = &call.function.arguments {
            state.index_to_args.entry(index).or_default().push_str(args);

            let call_id = state
                .index_to_id
                .get(&index)
                .cloned()
                .unwrap_or_else(|| format!("call_{index}"));

            events.push(Ok(ChatStreamEvent::ToolCallArgumentsDelta {
                id: call_id,
                delta: args.clone(),
            }));
        }
    }

    events
}

fn parse_tool_arguments(tool_name: &str, raw: &str) -> Value {
    match serde_json::from_str(raw) {
        Ok(value) => value,
        Err(e) => {
            tracing::warn!(
                tool_name = %tool_name,
                error = %e,
                "failed to parse tool call arguments, falling back to null"
            );
            Value::Null
        }
    }
}

/// Wraps a [`reqwest::Error`] from JSON deserialization into [`ProviderError::Decode`].
fn decode_error(provider_id: &ProviderId, error: &reqwest::Error) -> ProviderError {
    ProviderError::Decode {
        provider: provider_id.clone(),
        message: error.to_string(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chunk_events(state: &mut OpenAiStreamState, data: &str) -> Vec<ChatStreamEvent> {
        parse_chunk_event(state, data)
            .into_iter()
            .map(Result::unwrap)
            .collect()
    }

    #[test]
    fn stream_chunk_emits_tool_start_and_arguments_from_same_delta() {
        let mut state = OpenAiStreamState::new(ProviderId::new("openai"));

        let events = chunk_events(
            &mut state,
            r#"{
                "id": "chunk_1",
                "model": "gpt-test",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": "{\"city\":\"London\"}"
                            }
                        }]
                    },
                    "finish_reason": null
                }],
                "usage": null
            }"#,
        );

        assert_eq!(
            events,
            vec![
                ChatStreamEvent::ToolCallStarted {
                    id: "call_1".to_owned(),
                    name: "weather".to_owned(),
                },
                ChatStreamEvent::ToolCallArgumentsDelta {
                    id: "call_1".to_owned(),
                    delta: "{\"city\":\"London\"}".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn stream_finish_tool_calls_emits_completed_before_finished() {
        let mut state = OpenAiStreamState::new(ProviderId::new("openai"));
        let _ = chunk_events(
            &mut state,
            r#"{
                "id": "chunk_1",
                "model": "gpt-test",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": "{\"city\":\"London\"}"
                            }
                        }]
                    },
                    "finish_reason": null
                }],
                "usage": null
            }"#,
        );

        let events = chunk_events(
            &mut state,
            r#"{
                "id": "chunk_2",
                "model": "gpt-test",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls"
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3
                }
            }"#,
        );

        assert_eq!(
            events,
            vec![
                ChatStreamEvent::ToolCallCompleted {
                    call: ToolCall::new("call_1", "weather", json!({"city": "London"})),
                },
                ChatStreamEvent::Finished {
                    finish_reason: FinishReason::ToolCalls,
                    usage: Some(TokenUsage::new(1, 2)),
                },
            ]
        );
    }
}
