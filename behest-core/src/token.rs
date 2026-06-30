//! Token estimation using character-based heuristics.
//!
//! Uses the `chars / 4` rule-of-thumb to estimate token counts without
//! requiring a tokenizer. The 20,000-token buffer in compaction absorbs
//! estimation error, and the heuristic is consistent with industry practice.

use crate::message::{ContentPart, Message};
use crate::tool_types::ToolCall;

const CHARS_PER_TOKEN: usize = 4;

/// Estimates the number of tokens for a string.
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(CHARS_PER_TOKEN)
}

/// Estimates the token count for a content part.
#[must_use]
pub fn estimate_content_part_tokens(part: &ContentPart) -> usize {
    match part {
        ContentPart::Text { text } => estimate_tokens(text),
        ContentPart::Json { value } => estimate_tokens(&value.to_string()),
        ContentPart::ImageUrl { url, .. } => estimate_tokens(url),
    }
}

/// Estimates the token count for a tool call.
#[must_use]
pub fn estimate_tool_call_tokens(call: &ToolCall) -> usize {
    let name_tokens = estimate_tokens(&call.name);
    let args_tokens = estimate_tokens(&call.arguments.to_string());
    name_tokens + args_tokens + 20
}

/// Estimates the total token count for a provider [`Message`].
#[must_use]
pub fn estimate_message_tokens(message: &Message) -> usize {
    match message {
        Message::System { content } | Message::User { content } => {
            content
                .iter()
                .map(estimate_content_part_tokens)
                .sum::<usize>()
                + 8
        }
        Message::Assistant {
            content,
            tool_calls,
        } => {
            let content_tokens: usize = content.iter().map(estimate_content_part_tokens).sum();
            let tool_tokens: usize = tool_calls.iter().map(estimate_tool_call_tokens).sum();
            content_tokens + tool_tokens + 8
        }
        Message::Tool {
            tool_call_id,
            name,
            content,
        } => {
            let id_tokens = estimate_tokens(tool_call_id);
            let name_tokens = estimate_tokens(name);
            let content_tokens: usize = content.iter().map(estimate_content_part_tokens).sum();
            id_tokens + name_tokens + content_tokens + 10
        }
    }
}

/// Estimates the total token count for a slice of provider [`Message`]s.
#[must_use]
pub fn estimate_messages_tokens(messages: &[Message]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn estimate_should_round_up_for_fractional_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1);
        assert_eq!(estimate_tokens("ab"), 1);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens("12345678"), 2);
    }

    #[test]
    fn estimate_content_part_text() {
        let part = ContentPart::text("Hello, world!");
        assert_eq!(estimate_content_part_tokens(&part), 4);
    }

    #[test]
    fn estimate_content_part_json() {
        let part = ContentPart::json(json!({"key": "value"}));
        assert_eq!(estimate_content_part_tokens(&part), 4);
    }

    #[test]
    fn estimate_message_system() {
        let msg = Message::system_text("You are helpful.");
        assert_eq!(estimate_message_tokens(&msg), 12);
    }

    #[test]
    fn estimate_message_user() {
        let msg = Message::user_text("Hello");
        assert_eq!(estimate_message_tokens(&msg), 10);
    }

    #[test]
    fn estimate_messages_slice() {
        let messages = vec![
            Message::system_text("System"),
            Message::user_text("User"),
            Message::assistant_text("Assistant"),
        ];
        let total = estimate_messages_tokens(&messages);
        assert!(total > 0);
    }

    #[test]
    fn estimate_tool_call_includes_overhead() {
        let call = ToolCall::new("call_1", "echo", json!({}));
        let tokens = estimate_tool_call_tokens(&call);
        assert_eq!(tokens, 22);
    }
}
