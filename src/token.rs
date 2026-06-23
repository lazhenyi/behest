//! Token estimation using character-based heuristics.
//!
//! Uses the `chars / 4` rule-of-thumb to estimate token counts without
//! requiring a tokenizer. This is intentionally simple: the 20,000-token
//! buffer in compaction absorbs estimation error, and the heuristic is
//! consistent with what OpenCode V1/V2 uses.
//!
//! Ported from OpenCode V2: `packages/core/src/util/token.ts`.

use crate::provider::{ContentPart, Message, ToolCall};
use crate::store::MessageRecord;

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
        ContentPart::ImageUrl { url, .. } => {
            // Image URLs contribute minimal tokens (the URL itself)
            estimate_tokens(url)
        }
    }
}

/// Estimates the token count for a tool call.
#[must_use]
pub fn estimate_tool_call_tokens(call: &ToolCall) -> usize {
    // Tool calls rendered as JSON: name + arguments
    let name_tokens = estimate_tokens(&call.name);
    let args_tokens = estimate_tokens(&call.arguments.to_string());
    // Overhead for function call structure (~20 tokens)
    name_tokens + args_tokens + 20
}

/// Estimates the total token count for a provider [`Message`].
#[must_use]
pub fn estimate_message_tokens(message: &Message) -> usize {
    match message {
        Message::System { content } | Message::User { content } => {
            content.iter().map(estimate_content_part_tokens).sum::<usize>()
                // Role overhead for user messages
                + 8
        }
        Message::Assistant {
            content,
            tool_calls,
        } => {
            let content_tokens: usize = content.iter().map(estimate_content_part_tokens).sum();
            let tool_tokens: usize = tool_calls.iter().map(estimate_tool_call_tokens).sum();
            // Role overhead
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
            // Role + tool metadata overhead
            id_tokens + name_tokens + content_tokens + 10
        }
    }
}

/// Estimates the total token count for a slice of provider [`Message`]s.
#[must_use]
pub fn estimate_messages_tokens(messages: &[Message]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

/// Estimates the token count for a store [`MessageRecord`].
#[must_use]
pub fn estimate_record_tokens(record: &MessageRecord) -> usize {
    let content_tokens: usize = record
        .content
        .iter()
        .map(estimate_content_part_tokens)
        .sum();

    let tool_call_tokens: usize = record
        .tool_calls
        .iter()
        .map(estimate_tool_call_tokens)
        .sum();

    let tool_meta_tokens = match (&record.tool_call_id, &record.tool_name) {
        (Some(id), Some(name)) => estimate_tokens(id) + estimate_tokens(name),
        _ => 0,
    };

    // Role overhead
    let role_overhead = match record.role {
        crate::store::MessageRole::System
        | crate::store::MessageRole::User
        | crate::store::MessageRole::Assistant => 8,
        crate::store::MessageRole::Tool => 10,
    };

    content_tokens + tool_call_tokens + tool_meta_tokens + role_overhead
}

/// Estimates the total token count for a slice of [`MessageRecord`]s.
#[must_use]
pub fn estimate_records_tokens(records: &[MessageRecord]) -> usize {
    records.iter().map(estimate_record_tokens).sum()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::provider::{ContentPart, Message, ToolCall};
    use crate::store::MessageRecord;
    use serde_json::json;
    use uuid::Uuid;

    #[test]
    fn estimate_should_round_up_for_fractional_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1); // len 1 / 4 -> 1
        assert_eq!(estimate_tokens("ab"), 1); // len 2 / 4 -> 1
        assert_eq!(estimate_tokens("abcd"), 1); // len 4 / 4 -> 1
        assert_eq!(estimate_tokens("abcde"), 2); // len 5 / 4 -> 2
        assert_eq!(estimate_tokens("12345678"), 2); // len 8 / 4 -> 2
    }

    #[test]
    fn estimate_content_part_text() {
        let part = ContentPart::text("Hello, world!"); // len 13
        assert_eq!(estimate_content_part_tokens(&part), 4); // 13/4 -> 4
    }

    #[test]
    fn estimate_content_part_json() {
        let part = ContentPart::json(json!({"key": "value"}));
        // len of '{"key":"value"}' = 15
        assert_eq!(estimate_content_part_tokens(&part), 4); // 15/4 -> 4
    }

    #[test]
    fn estimate_message_system() {
        let msg = Message::system_text("You are helpful.");
        // "You are helpful." → 4 tokens (16/4) + role overhead 8 tokens = 12
        assert_eq!(estimate_message_tokens(&msg), 12);
    }

    #[test]
    fn estimate_message_user() {
        let msg = Message::user_text("Hello");
        // "Hello" → 2 tokens (5/4) + overhead 8 tokens = 10
        assert_eq!(estimate_message_tokens(&msg), 10);
    }

    #[test]
    fn estimate_message_assistant_with_tools() {
        let msg = Message::Assistant {
            content: vec![ContentPart::text("Using tool...")],
            tool_calls: vec![ToolCall::new("call_1", "echo", json!({"message": "test"}))],
        };
        // content: "Using tool..." len 14 -> 4 tokens
        // tool call: name "echo" len 4 -> 1 + args '{"message":"test"}' len 20 -> 5 + overhead 20 = 26
        // role overhead: 8
        // total: 4 + 26 + 8 = 38 -> 38/4 = 10
        let tokens = estimate_message_tokens(&msg);
        assert!(tokens > 5, "should account for tool calls: {tokens}");
    }

    #[test]
    fn estimate_message_tool() {
        let msg = Message::tool_text("call_1", "echo", r#"{"result":"ok"}"#);
        // id "call_1" → 2, name "echo" → 1, content '{"result":"ok"}' → 4, overhead 10 → all in tokens
        // 2+1+4+10 = 17
        assert_eq!(estimate_message_tokens(&msg), 17);
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
    fn estimate_record() {
        let record = MessageRecord::new(
            Uuid::now_v7(),
            crate::store::MessageRole::User,
            vec![ContentPart::text("Hello")],
        );
        let tokens = estimate_record_tokens(&record);
        // "Hello" → 2 tokens + overhead 8 = 10
        assert_eq!(tokens, 10);
    }

    #[test]
    fn estimate_records_slice() {
        let records = vec![
            MessageRecord::new(
                Uuid::now_v7(),
                crate::store::MessageRole::System,
                vec![ContentPart::text("System")],
            ),
            MessageRecord::new(
                Uuid::now_v7(),
                crate::store::MessageRole::User,
                vec![ContentPart::text("User")],
            ),
        ];
        let total = estimate_records_tokens(&records);
        assert!(total > 0);
    }

    #[test]
    fn estimate_tool_call_includes_overhead() {
        let call = ToolCall::new("call_1", "echo", json!({}));
        let tokens = estimate_tool_call_tokens(&call);
        // name "echo" → 1, args "{}" → 1, overhead 20 = 22 tokens
        assert_eq!(tokens, 22);
    }
}
