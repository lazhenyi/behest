//! Type conversions between neutral API types and OpenAI wire types.

use serde_json::{Value, json};

use crate::provider::{
    ChatRequest, ChatResponse, ContentPart, FinishReason, Message, ModelName, ProviderId,
    ResponseFormat, TokenUsage, ToolCall, ToolChoice, ToolSpec,
};

use super::types::{
    OpenAiChatRequest, OpenAiDelta, OpenAiFunctionCall, OpenAiFunctionDef, OpenAiMessage,
    OpenAiToolCall, OpenAiToolDef,
};

/// Converts a neutral chat request into an OpenAI chat request.
pub fn to_openai_request(request: &ChatRequest, stream: bool) -> OpenAiChatRequest {
    OpenAiChatRequest {
        model: request.model.as_str().to_owned(),
        messages: request
            .messages
            .iter()
            .map(convert_message)
            .collect(),
        tools: request.tools.iter().map(convert_tool_spec).collect(),
        tool_choice: convert_tool_choice(&request.tool_choice),
        response_format: request.response_format.as_ref().map(convert_response_format),
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_output_tokens,
        stop: request.stop.clone(),
        stream,
    }
}

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

fn serialize_content(parts: &[ContentPart]) -> Value {
    if parts.len() == 1 {
        if let ContentPart::Text { text } = &parts[0] {
            return Value::String(text.clone());
        }
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
                    .map(str::to_owned)
                    .unwrap_or_else(|| call.arguments.to_string()),
            ),
        },
    }
}

fn convert_tool_choice(choice: &ToolChoice) -> Option<Value> {
    match choice {
        ToolChoice::Auto => Some(json!("auto")),
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Required => Some(json!("required")),
        ToolChoice::Tool { name } => {
            Some(json!({"type": "function", "function": {"name": name}}))
        }
    }
}

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

/// Converts an OpenAI chat response into a neutral chat response.
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

fn parse_content_value(content: Option<&Value>) -> Vec<ContentPart> {
    match content {
        None => Vec::new(),
        Some(Value::String(text)) => vec![ContentPart::text(text.clone())],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(parse_content_item)
            .collect(),
        Some(other) => vec![ContentPart::Json {
            value: other.clone(),
        }],
    }
}

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
        _ => Some(ContentPart::Json { value: item.clone() }),
    }
}

fn convert_response_tool_calls(calls: &[OpenAiToolCall]) -> Vec<ToolCall> {
    calls
        .iter()
        .filter_map(|call| {
            let id = call.id.clone()?;
            let name = call.function.name.clone()?;
            let arguments_str = call.function.arguments.as_deref().unwrap_or("{}");
            let arguments = serde_json::from_str(arguments_str).unwrap_or(Value::Null);
            Some(ToolCall::new(id, name, arguments))
        })
        .collect()
}

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

fn convert_usage(usage: &super::types::OpenAiUsage) -> TokenUsage {
    TokenUsage::new(usage.prompt_tokens, usage.completion_tokens)
}

/// Extracts the model name from a streaming delta.
pub fn stream_delta_model(delta: &OpenAiDelta) -> Option<String> {
    delta.role.clone()
}
