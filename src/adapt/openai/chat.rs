//! OpenAI chat provider adapter implementing [`ChatProvider`].

use async_trait::async_trait;
use futures_util::StreamExt as _;
use reqwest::Client;

use crate::adapt::http::{build_client, status_to_error, with_bearer_auth};
use crate::adapt::sse::{SseEvent, SseStream};
use crate::error::ProviderError;
use crate::provider::{
    ChatProvider, ChatRequest, ChatResponse, ChatStream, ChatStreamEvent, FinishReason,
    ProviderCapabilities, ProviderHttpConfig, ProviderId, ProviderResult, TokenUsage,
};

use super::convert::{from_openai_response, to_openai_request};
use super::types::{OpenAiChatResponse, OpenAiStreamChunk, OpenAiToolCall};

/// OpenAI-compatible chat completion adapter.
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
        let response = self.build_request(&body).send().await.map_err(wrap_transport)?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(status_to_error(&self.id, status, &text));
        }

        let parsed: OpenAiChatResponse = response
            .json()
            .await
            .map_err(|e| decode_error(&self.id, e))?;

        from_openai_response(&self.id, &parsed)
            .ok_or_else(|| ProviderError::Decode {
                provider: self.id.clone(),
                message: "empty choices in response".to_owned(),
            })
    }

    async fn stream(&self, request: ChatRequest) -> ProviderResult<ChatStream> {
        let body = to_openai_request(&request, true);
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

        let mapped = sse_stream.filter_map(move |event| {
            let pid = provider_id.clone();
            async move { map_sse_event(&pid, event) }
        });

        let combined = futures_util::stream::once(async { Ok(started) }).chain(mapped);
        Ok(Box::pin(combined))
    }
}

fn map_sse_event(
    provider_id: &ProviderId,
    event: Result<SseEvent, ProviderError>,
) -> Option<Result<ChatStreamEvent, ProviderError>> {
    match event {
        Err(e) => Some(Err(e)),
        Ok(sse) if sse.is_openai_done() => None,
        Ok(sse) => parse_chunk_event(provider_id, &sse.data),
    }
}

fn parse_chunk_event(
    provider_id: &ProviderId,
    data: &str,
) -> Option<Result<ChatStreamEvent, ProviderError>> {
    let chunk: OpenAiStreamChunk = match serde_json::from_str(data) {
        Ok(c) => c,
        Err(e) => {
            return Some(Err(ProviderError::Decode {
                provider: provider_id.clone(),
                message: e.to_string(),
            }));
        }
    };

    let choice = chunk.choices.first()?;

    if let Some(reason) = &choice.finish_reason {
        return Some(Ok(ChatStreamEvent::Finished {
            finish_reason: convert_stream_finish(reason),
            usage: chunk.usage.as_ref().map(|u| TokenUsage::new(u.prompt_tokens, u.completion_tokens)),
        }));
    }

    if let Some(text) = &choice.delta.content {
        if !text.is_empty() {
            return Some(Ok(ChatStreamEvent::TextDelta {
                delta: text.clone(),
            }));
        }
    }

    if let Some(calls) = &choice.delta.tool_calls {
        return convert_tool_call_deltas(calls);
    }

    None
}

fn convert_stream_finish(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "tool_calls" => FinishReason::ToolCalls,
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Unknown(other.to_owned()),
    }
}

fn convert_tool_call_deltas(
    calls: &[OpenAiToolCall],
) -> Option<Result<ChatStreamEvent, ProviderError>> {
    let call = calls.first()?;
    let index = call.index.unwrap_or(0);
    let call_id = call.id.as_deref().unwrap_or("");
    let name = call.function.name.as_deref().unwrap_or("");

    if !call_id.is_empty() && !name.is_empty() {
        return Some(Ok(ChatStreamEvent::ToolCallStarted {
            id: call_id.to_owned(),
            name: name.to_owned(),
        }));
    }

    if let Some(args) = &call.function.arguments {
        return Some(Ok(ChatStreamEvent::ToolCallArgumentsDelta {
            id: format!("delta_{index}"),
            delta: args.clone(),
        }));
    }

    None
}

fn wrap_transport(source: reqwest::Error) -> ProviderError {
    if source.is_timeout() {
        ProviderError::Timeout {
            provider: ProviderId::new("openai"),
        }
    } else {
        ProviderError::Transport {
            provider: ProviderId::new("openai"),
            source,
        }
    }
}

fn decode_error(provider_id: &ProviderId, error: reqwest::Error) -> ProviderError {
    ProviderError::Decode {
        provider: provider_id.clone(),
        message: error.to_string(),
    }
}
