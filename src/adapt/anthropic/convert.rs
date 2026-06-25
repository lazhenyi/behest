//! Type conversions between neutral API types and Anthropic wire types.

use serde_json::{Value, json};

use crate::provider::{
    ChatRequest, ChatResponse, ContentPart, FinishReason, Message, ModelName, ProviderId,
    TokenUsage, ToolCall, ToolChoice, ToolSpec,
};

use super::types::{
    AnthropicContentBlock, AnthropicMessage, AnthropicRequest, AnthropicResponse, AnthropicToolDef,
    AnthropicToolResultContent,
};

const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Converts a neutral [`ChatRequest`] into an [`AnthropicRequest`].
///
/// Extracts system messages into the top-level `system` field, converts
/// remaining messages to Anthropic wire format, and maps tool definitions.
/// When `stream` is `true` the resulting request uses SSE streaming.
///
/// # Parameters
///
/// * `request` — The neutral chat request to convert.
/// * `stream` — Whether to enable SSE streaming for the Anthropic API.
pub fn to_anthropic_request(request: &ChatRequest, stream: bool) -> AnthropicRequest {
    let system = extract_system_text(&request.messages);
    let messages = convert_messages(&request.messages);

    AnthropicRequest {
        model: request.model.as_str().to_owned(),
        max_tokens: request.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        system,
        messages,
        tools: request.tools.iter().map(convert_tool_spec).collect(),
        tool_choice: convert_tool_choice(&request.tool_choice, !request.tools.is_empty()),
        temperature: request.temperature,
        top_p: request.top_p,
        stop_sequences: request.stop.clone(),
        stream,
    }
}

/// Extracts and joins all system message text content.
///
/// Concatenates system text parts with double-newline separators.
/// Returns `None` when there are no system messages.
fn extract_system_text(messages: &[Message]) -> Option<String> {
    let system_parts: Vec<&str> = messages
        .iter()
        .filter_map(|m| match m {
            Message::System { content } => Some(content),
            _ => None,
        })
        .flat_map(|parts| {
            parts.iter().filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
        })
        .collect();

    if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    }
}

/// Filters out system messages and converts the rest to Anthropic wire format.
fn convert_messages(messages: &[Message]) -> Vec<AnthropicMessage> {
    messages
        .iter()
        .filter(|m| !matches!(m, Message::System { .. }))
        .map(convert_single_message)
        .collect()
}

/// Converts one neutral [`Message`] to an [`AnthropicMessage`].
///
/// * `User` → role `"user"` with content blocks.
/// * `Assistant` → role `"assistant"` with content blocks and optional tool calls.
/// * `Tool` → role `"user"` with a `tool_result` content block.
/// * `System` → panics (callers must filter via [`convert_messages`]).
fn convert_single_message(message: &Message) -> AnthropicMessage {
    match message {
        Message::User { content } => AnthropicMessage {
            role: "user".to_owned(),
            content: convert_user_content(content),
        },
        Message::Assistant {
            content,
            tool_calls,
        } => {
            let mut blocks = convert_assistant_content(content);
            for call in tool_calls {
                blocks.push(AnthropicContentBlock::ToolUse {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    input: call.arguments.clone(),
                });
            }
            AnthropicMessage {
                role: "assistant".to_owned(),
                content: blocks,
            }
        }
        Message::Tool {
            tool_call_id,
            name: _,
            content,
        } => AnthropicMessage {
            role: "user".to_owned(),
            content: vec![AnthropicContentBlock::ToolResult {
                tool_use_id: tool_call_id.clone(),
                content: convert_tool_result_content(content),
            }],
        },
        // System messages are filtered by `convert_messages` before reaching here.
        Message::System { .. } => unreachable!("system messages are filtered by convert_messages"),
    }
}

/// Converts user content parts (text, JSON, images) to Anthropic content blocks.
fn convert_user_content(parts: &[ContentPart]) -> Vec<AnthropicContentBlock> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => AnthropicContentBlock::Text { text: text.clone() },
            ContentPart::Json { value } => AnthropicContentBlock::Text {
                text: value.to_string(),
            },
            ContentPart::ImageUrl { url, mime_type } => AnthropicContentBlock::Image {
                source: super::types::AnthropicImageSource {
                    source_type: "url".to_owned(),
                    url: url.clone(),
                    media_type: mime_type.clone(),
                },
            },
        })
        .collect()
}

/// Converts tool result content parts to Anthropic tool result content blocks.
///
/// Only text and image parts are valid in tool results; JSON parts are
/// serialized as text.
fn convert_tool_result_content(parts: &[ContentPart]) -> Vec<AnthropicToolResultContent> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => AnthropicToolResultContent::Text { text: text.clone() },
            ContentPart::Json { value } => AnthropicToolResultContent::Text {
                text: value.to_string(),
            },
            ContentPart::ImageUrl { url, mime_type } => AnthropicToolResultContent::Image {
                source: super::types::AnthropicImageSource {
                    source_type: "url".to_owned(),
                    url: url.clone(),
                    media_type: mime_type.clone(),
                },
            },
        })
        .collect()
}

/// Converts assistant content parts to Anthropic content blocks.
///
/// Image parts are replaced with a `[image]` placeholder text since the
/// Anthropic API does not accept images in assistant messages.
fn convert_assistant_content(parts: &[ContentPart]) -> Vec<AnthropicContentBlock> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => AnthropicContentBlock::Text { text: text.clone() },
            ContentPart::Json { value } => AnthropicContentBlock::Text {
                text: value.to_string(),
            },
            ContentPart::ImageUrl { .. } => AnthropicContentBlock::Text {
                text: "[image]".to_owned(),
            },
        })
        .collect()
}

/// Converts a neutral [`ToolSpec`] to an [`AnthropicToolDef`].
fn convert_tool_spec(spec: &ToolSpec) -> AnthropicToolDef {
    AnthropicToolDef {
        name: spec.name.clone(),
        description: spec.description.clone(),
        input_schema: spec.parameters_schema.clone(),
    }
}

/// Converts a neutral [`ToolChoice`] to an Anthropic tool_choice JSON value.
///
/// Returns `None` when no tools are available or the choice is `ToolChoice::None`.
fn convert_tool_choice(choice: &ToolChoice, has_tools: bool) -> Option<Value> {
    if !has_tools {
        return None;
    }
    match choice {
        ToolChoice::Auto => Some(json!({"type": "auto"})),
        ToolChoice::None => None,
        ToolChoice::Required => Some(json!({"type": "any"})),
        ToolChoice::Tool { name } => Some(json!({"type": "tool", "name": name})),
    }
}

/// Converts an [`AnthropicResponse`] into a neutral [`ChatResponse`].
///
/// Extracts text content and tool calls from the response content blocks and
/// maps the stop reason and usage tokens.
///
/// # Parameters
///
/// * `provider` — The provider identifier to attach to the response.
/// * `response` — The raw Anthropic API response.
pub fn from_anthropic_response(
    provider: &ProviderId,
    response: &AnthropicResponse,
) -> ChatResponse {
    let (content, tool_calls) = parse_content_blocks(&response.content);

    ChatResponse {
        provider: provider.clone(),
        model: ModelName::new(&response.model),
        message: Message::Assistant {
            content,
            tool_calls,
        },
        finish_reason: convert_stop_reason(response.stop_reason.as_deref()),
        usage: response.usage.as_ref().map(convert_usage),
        raw: None,
    }
}

/// Splits Anthropic content blocks into text parts and tool calls.
///
/// Image and tool result blocks are silently skipped as they do not appear
/// in assistant responses from the non-streaming endpoint.
fn parse_content_blocks(blocks: &[AnthropicContentBlock]) -> (Vec<ContentPart>, Vec<ToolCall>) {
    let mut content_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text } => {
                content_parts.push(ContentPart::text(text.clone()));
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall::new(id.clone(), name.clone(), input.clone()));
            }
            AnthropicContentBlock::Image { .. } | AnthropicContentBlock::ToolResult { .. } => {}
        }
    }

    (content_parts, tool_calls)
}

/// Converts an Anthropic stop reason string to the neutral [`FinishReason`].
fn convert_stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn" | "stop_sequence") => FinishReason::Stop,
        Some("tool_use") => FinishReason::ToolCalls,
        Some("max_tokens") => FinishReason::Length,
        Some(other) => FinishReason::Unknown(other.to_owned()),
        None => FinishReason::Unknown("null".to_owned()),
    }
}

/// Converts [`AnthropicUsage`] to neutral [`TokenUsage`].
fn convert_usage(usage: &super::types::AnthropicUsage) -> TokenUsage {
    TokenUsage::new(usage.input_tokens, usage.output_tokens)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_request() -> ChatRequest {
        ChatRequest::new(ModelName::new("claude-3-sonnet")).with_user_text("Hello")
    }

    #[test]
    fn to_anthropic_request_should_set_model_and_max_tokens() {
        let req = sample_request();
        let anthropic = to_anthropic_request(&req, false);

        assert_eq!(anthropic.model, "claude-3-sonnet");
        assert_eq!(anthropic.max_tokens, DEFAULT_MAX_TOKENS);
        assert!(!anthropic.stream);
    }

    #[test]
    fn to_anthropic_request_should_use_custom_max_tokens() {
        let mut req = sample_request();
        req.max_output_tokens = Some(1024);
        let anthropic = to_anthropic_request(&req, false);
        assert_eq!(anthropic.max_tokens, 1024);
    }

    #[test]
    fn extract_system_text_should_join_system_messages() {
        let messages = vec![
            Message::system_text("You are helpful."),
            Message::user_text("Hi"),
            Message::system_text("Be concise."),
        ];

        let system = extract_system_text(&messages);
        assert_eq!(system, Some("You are helpful.\n\nBe concise.".to_owned()));
    }

    #[test]
    fn extract_system_text_should_return_none_when_no_system() {
        let messages = vec![Message::user_text("Hi")];
        assert_eq!(extract_system_text(&messages), None);
    }

    #[test]
    fn convert_messages_should_filter_system_messages() {
        let messages = vec![
            Message::system_text("System"),
            Message::user_text("User"),
            Message::assistant_text("Assistant"),
        ];

        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].role, "user");
        assert_eq!(converted[1].role, "assistant");
    }

    #[test]
    fn convert_single_message_should_convert_tool_message_to_user_role() {
        let msg = Message::tool_text("call_1", "weather", "Sunny");
        let converted = convert_single_message(&msg);

        assert_eq!(converted.role, "user");
        assert_eq!(converted.content.len(), 1);
        assert!(matches!(
            &converted.content[0],
            AnthropicContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "call_1"
        ));
    }

    #[test]
    fn convert_single_message_should_include_tool_calls_in_assistant() {
        let msg = Message::Assistant {
            content: vec![ContentPart::text("Let me check")],
            tool_calls: vec![ToolCall::new(
                "call_1",
                "weather",
                json!({"city": "London"}),
            )],
        };

        let converted = convert_single_message(&msg);
        assert_eq!(converted.role, "assistant");
        assert_eq!(converted.content.len(), 2);
        assert!(matches!(
            &converted.content[1],
            AnthropicContentBlock::ToolUse { name, .. } if name == "weather"
        ));
    }

    #[test]
    fn convert_tool_choice_should_return_none_when_no_tools() {
        assert_eq!(convert_tool_choice(&ToolChoice::Auto, false), None);
    }

    #[test]
    fn convert_tool_choice_should_map_auto() {
        let result = convert_tool_choice(&ToolChoice::Auto, true);
        assert_eq!(result, Some(json!({"type": "auto"})));
    }

    #[test]
    fn convert_tool_choice_should_return_none_for_none_choice() {
        assert_eq!(convert_tool_choice(&ToolChoice::None, true), None);
    }

    #[test]
    fn from_anthropic_response_should_convert_text_response() {
        let provider = ProviderId::new("anthropic");
        let response = super::super::types::AnthropicResponse {
            id: "msg_1".to_owned(),
            model: "claude-3-sonnet".to_owned(),
            content: vec![AnthropicContentBlock::Text {
                text: "Hello!".to_owned(),
            }],
            stop_reason: Some("end_turn".to_owned()),
            usage: Some(super::super::types::AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
            }),
        };

        let result = from_anthropic_response(&provider, &response);
        assert_eq!(result.model.as_str(), "claude-3-sonnet");
        assert_eq!(result.finish_reason, FinishReason::Stop);
        assert!(matches!(result.message, Message::Assistant { .. }));
        let usage = result.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn from_anthropic_response_should_extract_tool_calls() {
        let provider = ProviderId::new("anthropic");
        let response = super::super::types::AnthropicResponse {
            id: "msg_2".to_owned(),
            model: "claude-3-sonnet".to_owned(),
            content: vec![AnthropicContentBlock::ToolUse {
                id: "toolu_1".to_owned(),
                name: "get_weather".to_owned(),
                input: json!({"city": "London"}),
            }],
            stop_reason: Some("tool_use".to_owned()),
            usage: None,
        };

        let result = from_anthropic_response(&provider, &response);
        assert_eq!(result.finish_reason, FinishReason::ToolCalls);
        if let Message::Assistant { tool_calls, .. } = &result.message {
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].name, "get_weather");
        } else {
            panic!("expected assistant message with tool calls");
        }
    }

    #[test]
    fn convert_stop_reason_should_map_all_known_reasons() {
        assert_eq!(convert_stop_reason(Some("end_turn")), FinishReason::Stop);
        assert_eq!(
            convert_stop_reason(Some("tool_use")),
            FinishReason::ToolCalls
        );
        assert_eq!(
            convert_stop_reason(Some("max_tokens")),
            FinishReason::Length
        );
        assert_eq!(
            convert_stop_reason(Some("stop_sequence")),
            FinishReason::Stop
        );
    }
}
