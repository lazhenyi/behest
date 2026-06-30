//! Type conversions between neutral API types and OpenAI wire types.

use serde_json::{Value, json};

use crate::provider::{
    ChatRequest, ChatResponse, ContentPart, FinishReason, Message, ModelName, ProviderId,
    ResponseFormat, TokenUsage, ToolCall, ToolChoice, ToolSpec,
};

use super::types::{
    OpenAiChatRequest, OpenAiFunctionCall, OpenAiFunctionDef, OpenAiMessage, OpenAiToolCall,
    OpenAiToolDef,
};

/// Converts a neutral [`ChatRequest`] into an [`OpenAiChatRequest`].
///
/// Maps messages, tool definitions, tool choice, response format, and sampling
/// parameters to the OpenAI wire format. When `stream` is `true` the resulting
/// request uses SSE streaming.
///
/// # Parameters
///
/// * `request` — The neutral chat request to convert.
/// * `stream` — Whether to enable SSE streaming.
pub fn to_openai_request(request: &ChatRequest, stream: bool) -> OpenAiChatRequest {
    OpenAiChatRequest {
        model: request.model.as_str().to_owned(),
        messages: request.messages.iter().map(convert_message).collect(),
        tools: request.tools.iter().map(convert_tool_spec).collect(),
        tool_choice: convert_tool_choice(&request.tool_choice),
        response_format: request
            .response_format
            .as_ref()
            .map(convert_response_format),
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_output_tokens,
        stop: request.stop.clone(),
        stream,
    }
}

/// Converts one neutral [`Message`] to an [`OpenAiMessage`].
///
/// Handles `System`, `User`, `Assistant` (with tool calls), and `Tool` roles.
fn convert_message(message: &Message) -> OpenAiMessage {
    match message {
        Message::System { content } => OpenAiMessage {
            role: "system".to_owned(),
            content: Some(serialize_content(content)),
            tool_calls: None,
            tool_call_id: None,
        },
        Message::User { content } => OpenAiMessage {
            role: "user".to_owned(),
            content: Some(serialize_content(content)),
            tool_calls: None,
            tool_call_id: None,
        },
        Message::Assistant {
            content,
            tool_calls,
        } => OpenAiMessage {
            role: "assistant".to_owned(),
            content: Some(serialize_content(content)),
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls.iter().map(convert_tool_call).collect())
            },
            tool_call_id: None,
        },
        Message::Tool {
            tool_call_id,
            name: _,
            content,
        } => OpenAiMessage {
            role: "tool".to_owned(),
            content: Some(serialize_content(content)),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.clone()),
        },
    }
}

/// Serializes content parts to the OpenAI `content` field format.
///
/// Single text parts are serialized as a plain string; multiple parts or
/// non-text content are serialized as a JSON array of content blocks.
fn serialize_content(parts: &[ContentPart]) -> Value {
    if parts.len() == 1
        && let ContentPart::Text { text } = &parts[0]
    {
        return Value::String(text.clone());
    }

    let items: Vec<Value> = parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => {
                json!({"type": "text", "text": text})
            }
            ContentPart::Json { value } => value.clone(),
            ContentPart::ImageUrl { url, mime_type } => {
                let mut obj = json!({"type": "image_url", "image_url": {"url": url}});
                if let Some(mime) = mime_type {
                    obj["image_url"]["detail"] = Value::String(mime.clone());
                }
                obj
            }
        })
        .collect();

    Value::Array(items)
}

/// Converts a neutral [`ToolSpec`] to an [`OpenAiToolDef`] with `type: "function"`.
fn convert_tool_spec(spec: &ToolSpec) -> OpenAiToolDef {
    OpenAiToolDef {
        kind: "function".to_owned(),
        function: OpenAiFunctionDef {
            name: spec.name.clone(),
            description: spec.description.clone(),
            parameters: spec.parameters_schema.clone(),
        },
    }
}

/// Converts a neutral [`ToolCall`] to an [`OpenAiToolCall`] for assistant messages.
fn convert_tool_call(call: &ToolCall) -> OpenAiToolCall {
    OpenAiToolCall {
        id: Some(call.id.clone()),
        index: None,
        kind: Some("function".to_owned()),
        function: OpenAiFunctionCall {
            name: Some(call.name.clone()),
            arguments: Some(
                call.arguments
                    .as_str()
                    .map_or_else(|| call.arguments.to_string(), str::to_owned),
            ),
        },
    }
}

/// Converts a neutral [`ToolChoice`] to an OpenAI tool_choice JSON value.
///
/// Unlike Anthropic, OpenAI always sends a tool_choice value (defaulting to
/// `"auto"`), never `None`.
fn convert_tool_choice(choice: &ToolChoice) -> Option<Value> {
    match choice {
        ToolChoice::Auto => Some(json!("auto")),
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Required => Some(json!("required")),
        ToolChoice::Tool { name } => Some(json!({"type": "function", "function": {"name": name}})),
    }
}

/// Converts a neutral [`ResponseFormat`] to an OpenAI response_format JSON value.
///
/// Supports `text`, `json_object`, and `json_schema` (with strict mode) formats.
fn convert_response_format(format: &ResponseFormat) -> Value {
    match format {
        ResponseFormat::Text => json!({"type": "text"}),
        ResponseFormat::JsonObject => json!({"type": "json_object"}),
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => json!({
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema,
                "strict": strict,
            }
        }),
    }
}

/// Converts an [`super::types::OpenAiChatResponse`] into a neutral [`ChatResponse`].
///
/// Extracts the first choice's message, tool calls, finish reason, and usage
/// tokens. Returns `None` when the response contains zero choices.
///
/// # Parameters
///
/// * `provider` — The provider identifier to attach to the response.
/// * `response` — The raw OpenAI API response.
pub fn from_openai_response(
    provider: &ProviderId,
    response: &super::types::OpenAiChatResponse,
) -> Option<ChatResponse> {
    let choice = response.choices.first()?;
    let message = convert_response_message(&choice.message);

    Some(ChatResponse {
        provider: provider.clone(),
        model: ModelName::new(&response.model),
        message,
        finish_reason: convert_finish_reason(choice.finish_reason.as_deref()),
        usage: response.usage.as_ref().map(convert_usage),
        raw: None,
    })
}

/// Converts an [`OpenAiMessage`] response to a neutral [`Message::Assistant`].
///
/// Parses text content and tool calls from the response message.
fn convert_response_message(message: &OpenAiMessage) -> Message {
    let content = parse_content_value(message.content.as_ref());
    let tool_calls = message
        .tool_calls
        .as_deref()
        .map(convert_response_tool_calls)
        .unwrap_or_default();

    Message::Assistant {
        content,
        tool_calls,
    }
}

/// Parses the OpenAI `content` field (string, array, or object) into [`ContentPart`]s.
fn parse_content_value(content: Option<&Value>) -> Vec<ContentPart> {
    match content {
        None => Vec::new(),
        Some(Value::String(text)) => vec![ContentPart::text(text.clone())],
        Some(Value::Array(items)) => items.iter().filter_map(parse_content_item).collect(),
        Some(other) => vec![ContentPart::Json {
            value: other.clone(),
        }],
    }
}

/// Parses one content array item into a [`ContentPart`].
///
/// Supports `text` and `image_url` typed items; unknown types are preserved
/// as [`ContentPart::Json`].
fn parse_content_item(item: &Value) -> Option<ContentPart> {
    let kind = item.get("type")?.as_str()?;
    match kind {
        "text" => {
            let text = item.get("text")?.as_str()?;
            Some(ContentPart::text(text))
        }
        "image_url" => {
            let url = item.get("image_url")?.get("url")?.as_str()?;
            Some(ContentPart::image_url(url, None))
        }
        _ => Some(ContentPart::Json {
            value: item.clone(),
        }),
    }
}

/// Converts OpenAI response tool calls to neutral [`ToolCall`]s.
///
/// Filters out calls with missing id or name. Falls back to `Value::Null`
/// for unparseable JSON arguments.
fn convert_response_tool_calls(calls: &[OpenAiToolCall]) -> Vec<ToolCall> {
    calls
        .iter()
        .filter_map(|call| {
            let id = call.id.clone()?;
            let name = call.function.name.clone()?;
            let arguments_str = call.function.arguments.as_deref().unwrap_or("{}");
            let arguments = serde_json::from_str(arguments_str).unwrap_or_else(|e| {
                tracing::warn!(
                    tool_name = %name,
                    error = %e,
                    "failed to parse tool call arguments, falling back to null"
                );
                Value::Null
            });
            Some(ToolCall::new(id, name, arguments))
        })
        .collect()
}

/// Converts an OpenAI finish reason string to the neutral [`FinishReason`].
fn convert_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("length") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        Some(other) => FinishReason::Unknown(other.to_owned()),
        None => FinishReason::Unknown("null".to_owned()),
    }
}

/// Converts [`OpenAiUsage`] to neutral [`TokenUsage`].
fn convert_usage(usage: &super::types::OpenAiUsage) -> TokenUsage {
    TokenUsage::new(usage.prompt_tokens, usage.completion_tokens)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_request() -> ChatRequest {
        ChatRequest::new(ModelName::new("gpt-4")).with_user_text("Hello")
    }

    #[test]
    fn to_openai_request_should_set_model_and_stream() {
        let req = sample_request();
        let openai = to_openai_request(&req, true);

        assert_eq!(openai.model, "gpt-4");
        assert!(openai.stream);
        assert_eq!(openai.messages.len(), 1);
        assert_eq!(openai.messages[0].role, "user");
    }

    #[test]
    fn to_openai_request_should_convert_system_message() {
        let req = ChatRequest::new(ModelName::new("gpt-4"))
            .with_message(Message::system_text("You are helpful"))
            .with_user_text("Hi");

        let openai = to_openai_request(&req, false);
        assert_eq!(openai.messages[0].role, "system");
        assert_eq!(openai.messages[1].role, "user");
    }

    #[test]
    fn to_openai_request_should_convert_tool_choice_auto() {
        let req = sample_request();
        let openai = to_openai_request(&req, false);
        assert_eq!(openai.tool_choice, Some(json!("auto")));
    }

    #[test]
    fn to_openai_request_should_convert_tool_choice_none() {
        let mut req = sample_request();
        req.tool_choice = ToolChoice::None;
        let openai = to_openai_request(&req, false);
        assert_eq!(openai.tool_choice, Some(json!("none")));
    }

    #[test]
    fn serialize_content_should_return_string_for_single_text() {
        let parts = vec![ContentPart::text("hello")];
        let result = serialize_content(&parts);
        assert_eq!(result, Value::String("hello".to_owned()));
    }

    #[test]
    fn serialize_content_should_return_array_for_multiple_parts() {
        let parts = vec![ContentPart::text("hello"), ContentPart::text("world")];
        let result = serialize_content(&parts);
        assert!(result.is_array());
        assert_eq!(result.as_array().unwrap().len(), 2);
    }

    #[test]
    fn convert_finish_reason_should_map_known_reasons() {
        assert_eq!(convert_finish_reason(Some("stop")), FinishReason::Stop);
        assert_eq!(
            convert_finish_reason(Some("tool_calls")),
            FinishReason::ToolCalls
        );
        assert_eq!(convert_finish_reason(Some("length")), FinishReason::Length);
        assert_eq!(
            convert_finish_reason(Some("content_filter")),
            FinishReason::ContentFilter
        );
    }

    #[test]
    fn convert_finish_reason_should_handle_unknown_and_none() {
        assert_eq!(
            convert_finish_reason(Some("custom")),
            FinishReason::Unknown("custom".to_owned())
        );
        assert_eq!(
            convert_finish_reason(None),
            FinishReason::Unknown("null".to_owned())
        );
    }

    #[test]
    fn from_openai_response_should_return_none_for_empty_choices() {
        let provider = ProviderId::new("openai");
        let response = super::super::types::OpenAiChatResponse {
            id: "1".to_owned(),
            model: "gpt-4".to_owned(),
            choices: vec![],
            usage: None,
        };

        assert!(from_openai_response(&provider, &response).is_none());
    }

    #[test]
    fn from_openai_response_should_convert_text_response() {
        let provider = ProviderId::new("openai");
        let response = super::super::types::OpenAiChatResponse {
            id: "1".to_owned(),
            model: "gpt-4".to_owned(),
            choices: vec![super::super::types::OpenAiChatChoice {
                index: 0,
                message: super::super::types::OpenAiMessage {
                    role: "assistant".to_owned(),
                    content: Some(Value::String("Hello!".to_owned())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".to_owned()),
            }],
            usage: Some(super::super::types::OpenAiUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        };

        let result = from_openai_response(&provider, &response).unwrap();
        assert_eq!(result.model.as_str(), "gpt-4");
        assert_eq!(result.finish_reason, FinishReason::Stop);
        assert!(matches!(result.message, Message::Assistant { .. }));
        let usage = result.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn convert_response_tool_calls_should_parse_valid_arguments() {
        let calls = vec![super::super::types::OpenAiToolCall {
            id: Some("call_1".to_owned()),
            index: None,
            kind: Some("function".to_owned()),
            function: super::super::types::OpenAiFunctionCall {
                name: Some("get_weather".to_owned()),
                arguments: Some(r#"{"city":"London"}"#.to_owned()),
            },
        }];

        let result = convert_response_tool_calls(&calls);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "call_1");
        assert_eq!(result[0].name, "get_weather");
        assert_eq!(result[0].arguments, json!({"city": "London"}));
    }

    #[test]
    fn convert_response_tool_calls_should_skip_missing_id_or_name() {
        let calls = vec![super::super::types::OpenAiToolCall {
            id: None,
            index: None,
            kind: None,
            function: super::super::types::OpenAiFunctionCall {
                name: Some("test".to_owned()),
                arguments: None,
            },
        }];

        let result = convert_response_tool_calls(&calls);
        assert!(result.is_empty());
    }
}
