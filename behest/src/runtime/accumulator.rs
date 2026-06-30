//! Streaming accumulator for text and tool calls.
//!
//! Maintains state for accumulating streaming deltas from provider
//! into complete assistant messages and tool calls.

use std::collections::HashMap;

use crate::provider::{ContentPart, Message, ToolCall};

/// Accumulates streaming deltas from a provider into a complete assistant
/// message and zero or more tool calls. Tracks both text content and
/// partial tool call arguments until all deltas are received.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    text: String,
    tool_calls: HashMap<String, ToolCallAccumulator>,
}

impl StreamAccumulator {
    /// Creates a new accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a text delta chunk to the internal buffer.
    pub fn append_text(&mut self, delta: &str) {
        self.text.push_str(delta);
    }

    /// Starts a tool call.
    pub fn start_tool_call(&mut self, id: String, name: String) {
        self.tool_calls.insert(
            id.clone(),
            ToolCallAccumulator {
                id,
                name,
                arguments: String::new(),
            },
        );
    }

    /// Appends arguments to a tool call.
    pub fn append_tool_arguments(&mut self, id: &str, delta: &str) {
        if let Some(tc) = self.tool_calls.get_mut(id) {
            tc.arguments.push_str(delta);
        }
    }

    /// Returns the accumulated text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Parses accumulated tool call arguments into [`ToolCall`] values.
    /// Unparseable JSON arguments produce a [`serde_json::Value::Null`] fallback.
    #[must_use]
    pub fn tool_calls(&self) -> Vec<ToolCall> {
        self.tool_calls
            .values()
            .map(|tc| {
                let arguments =
                    serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);
                ToolCall::new(tc.id.clone(), tc.name.clone(), arguments)
            })
            .collect()
    }

    /// Assembles an assistant [`Message`] from the accumulated text and
    /// tool calls. Handles three cases:
    /// - Neither text nor tool calls → empty assistant message.
    /// - Text only → [`Message::assistant_text`].
    /// - Tool calls present (with optional text) → structured [`Message::Assistant`].
    #[must_use]
    pub fn to_message(&self) -> Message {
        let tool_calls = self.tool_calls();
        if tool_calls.is_empty() && self.text.is_empty() {
            Message::Assistant {
                content: vec![],
                tool_calls: vec![],
            }
        } else if tool_calls.is_empty() {
            Message::assistant_text(&self.text)
        } else {
            Message::Assistant {
                content: if self.text.is_empty() {
                    vec![]
                } else {
                    vec![ContentPart::text(&self.text)]
                },
                tool_calls,
            }
        }
    }

    /// Clears the accumulator.
    pub fn clear(&mut self) {
        self.text.clear();
        self.tool_calls.clear();
    }
}

/// Accumulator for a single tool call.
#[derive(Debug)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn accumulate_text() {
        let mut acc = StreamAccumulator::new();
        acc.append_text("Hello");
        acc.append_text(" ");
        acc.append_text("World");
        assert_eq!(acc.text(), "Hello World");
    }

    #[test]
    fn accumulate_tool_call() {
        let mut acc = StreamAccumulator::new();
        acc.start_tool_call("call_1".to_string(), "get_weather".to_string());
        acc.append_tool_arguments("call_1", r#"{"location":"#);
        acc.append_tool_arguments("call_1", r#""Paris"}"#);

        let calls = acc.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "Paris");
    }

    #[test]
    fn to_message_text_only() {
        let mut acc = StreamAccumulator::new();
        acc.append_text("Response");
        let msg = acc.to_message();
        match msg {
            Message::Assistant {
                content,
                tool_calls,
            } => {
                assert!(tool_calls.is_empty());
                assert!(!content.is_empty());
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn to_message_with_tool_calls() {
        let mut acc = StreamAccumulator::new();
        acc.append_text("Thinking...");
        acc.start_tool_call("call_1".to_string(), "tool".to_string());
        acc.append_tool_arguments("call_1", "{}");

        let msg = acc.to_message();
        match msg {
            Message::Assistant {
                content,
                tool_calls,
            } => {
                assert_eq!(tool_calls.len(), 1);
                assert!(!content.is_empty());
            }
            _ => panic!("Expected Assistant message with tool calls"),
        }
    }

    #[test]
    fn to_message_empty_accumulator_returns_empty_assistant() {
        let acc = StreamAccumulator::new();

        let msg = acc.to_message();
        match msg {
            Message::Assistant {
                content,
                tool_calls,
            } => {
                assert!(content.is_empty());
                assert!(tool_calls.is_empty());
            }
            _ => panic!("Expected empty Assistant message"),
        }
    }

    #[test]
    fn invalid_tool_arguments_fall_back_to_null() {
        let mut acc = StreamAccumulator::new();
        acc.start_tool_call("call_1".to_string(), "tool".to_string());
        acc.append_tool_arguments("call_1", "{invalid json");

        let calls = acc.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, Value::Null);
    }

    #[test]
    fn clear_resets_text_and_tool_calls() {
        let mut acc = StreamAccumulator::new();
        acc.append_text("partial");
        acc.start_tool_call("call_1".to_string(), "tool".to_string());
        acc.append_tool_arguments("call_1", "{}");

        acc.clear();

        assert_eq!(acc.text(), "");
        assert!(acc.tool_calls().is_empty());
        match acc.to_message() {
            Message::Assistant {
                content,
                tool_calls,
            } => {
                assert!(content.is_empty());
                assert!(tool_calls.is_empty());
            }
            _ => panic!("Expected empty Assistant message after clear"),
        }
    }

    #[test]
    fn starting_same_tool_call_id_replaces_previous_state() {
        let mut acc = StreamAccumulator::new();
        acc.start_tool_call("call_1".to_string(), "first_tool".to_string());
        acc.append_tool_arguments("call_1", r#"{"old":"value"}"#);

        acc.start_tool_call("call_1".to_string(), "second_tool".to_string());
        acc.append_tool_arguments("call_1", r#"{"fresh":true}"#);

        let calls = acc.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "second_tool");
        assert_eq!(calls[0].arguments["fresh"], true);
        assert_eq!(calls[0].arguments.get("old"), None);
    }
}
