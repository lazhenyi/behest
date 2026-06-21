//! Type conversions between neutral API types and Anthropic wire types.

use serde_json::{Value, json};

use crate::provider::{
    ChatRequest, ChatResponse, ContentPart, FinishReason, Message, ModelName, ProviderId,
    TokenUsage, ToolCall, ToolChoice, ToolSpec,
};

use super::types::{
    AnthropicContentBlock, AnthropicMessage, AnthropicRequest, AnthropicResponse, AnthropicToolDef,
};

const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Converts a neutral chat request into an Anthropic request.
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

fn convert_messages(messages: &[Message]) -> Vec<AnthropicMessage> {
    messages
        .iter()
        .filter(|m| !matches!(m, Message::System { .. }))
        .map(convert_single_message)
        .collect()
}

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
                content: convert_user_content(content),
            }],
        },
        Message::System { .. } => AnthropicMessage {
            role: "user".to_owned(),
            content: vec![AnthropicContentBlock::Text {
                text: String::new(),
            }],
        },
    }
}

fn convert_user_content(parts: &[ContentPart]) -> Vec<AnthropicContentBlock> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => AnthropicContentBlock::Text { text: text.clone() },
            ContentPart::Json { value } => AnthropicContentBlock::Text {
                text: value.to_string(),
            },
            ContentPart::ImageUrl { url, mime_type } => {
                AnthropicContentBlock::Image {
                    source: super::types::AnthropicImageSource {
                        source_type: "url".to_owned(),
                        url: url.clone(),
                        media_type: mime_type.clone(),
                    },
                }
            }
        })
        .collect()
}

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

fn convert_tool_spec(spec: &ToolSpec) -> AnthropicToolDef {
    AnthropicToolDef {
        name: spec.name.clone(),
        description: spec.description.clone(),
        input_schema: spec.parameters_schema.clone(),
    }
}

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

/// Converts an Anthropic response into a neutral chat response.
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

fn convert_stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn") => FinishReason::Stop,
        Some("tool_use") => FinishReason::ToolCalls,
        Some("max_tokens") => FinishReason::Length,
        Some("stop_sequence") => FinishReason::Stop,
        Some(other) => FinishReason::Unknown(other.to_owned()),
        None => FinishReason::Unknown("null".to_owned()),
    }
}

fn convert_usage(usage: &super::types::AnthropicUsage) -> TokenUsage {
    TokenUsage::new(usage.input_tokens, usage.output_tokens)
}
