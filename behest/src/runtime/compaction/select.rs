//! Message selection for compaction.
//!
//! Implements turn-based selection: conversation messages are grouped into
//! "turns" (bounded by non-compaction user messages), and the most recent
//! `tail_turns` turns are preserved. Within the preserved range, a token
//! budget (`keep_tokens`) is enforced from-back, so the oldest retained
//! messages are dropped if the budget is exceeded.
//!
//! Ported from OpenCode V1: `packages/opencode/src/session/compaction.ts` select().

use crate::store::MessageRecord;
use crate::token::estimate_record_tokens;

/// Result of the selection algorithm.
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Messages to be compacted (summarised by the compaction LLM).
    pub head: Vec<MessageRecord>,
    /// Messages to retain as recent context.
    pub tail: Vec<MessageRecord>,
    /// The message ID of the first retained message (tail boundary).
    pub tail_start_id: Option<uuid::Uuid>,
}

/// A conversation turn — bounded by non-compaction user messages.
#[derive(Debug)]
struct Turn {
    /// Indices into the source message slice.
    indices: Vec<usize>,
}

/// Groups messages into turns, where each turn starts at a non-compaction
/// user message and extends until the next non-compaction user message
/// (or end of slice).
fn turns(messages: &[MessageRecord]) -> Vec<Turn> {
    let mut result = Vec::new();
    let mut current = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        if !msg.is_compaction && msg.role == crate::store::MessageRole::User && !current.is_empty()
        {
            result.push(Turn {
                indices: std::mem::take(&mut current),
            });
        }
        current.push(i);
    }

    if !current.is_empty() {
        result.push(Turn { indices: current });
    }

    result
}

/// Selects which messages to compact and which to retain.
///
/// # Arguments
/// * `messages` - The full session message history, in chronological order.
/// * `tail_turns` - Number of recent turns to preserve intact (default 2).
/// * `keep_tokens` - Token budget for the retained tail.
///
/// # Returns
/// A [`SelectionResult`] with `head` (to compact), `tail` (to retain),
/// and `tail_start_id` (first retained message ID).
#[must_use]
pub fn select(
    messages: &[MessageRecord],
    tail_turns: usize,
    keep_tokens: usize,
) -> SelectionResult {
    if messages.is_empty() {
        return SelectionResult {
            head: Vec::new(),
            tail: Vec::new(),
            tail_start_id: None,
        };
    }

    let all_turns = turns(messages);

    if all_turns.is_empty() {
        return SelectionResult {
            head: messages.to_vec(),
            tail: Vec::new(),
            tail_start_id: None,
        };
    }

    // Take the most recent N turns
    let preserved_turns = if all_turns.len() <= tail_turns {
        all_turns.len()
    } else {
        tail_turns
    };

    let head_turns = &all_turns[..all_turns.len() - preserved_turns];
    let tail_turns_slice = &all_turns[all_turns.len() - preserved_turns..];

    // Collect head indices
    let head_indices: Vec<usize> = head_turns
        .iter()
        .flat_map(|t| t.indices.iter().copied())
        .collect();

    // Walk backward through tail turns, accumulating tokens
    let mut tail_indices: Vec<usize> = Vec::new();
    let mut accumulated = 0usize;

    for turn in tail_turns_slice.iter().rev() {
        let mut turn_accumulated = 0usize;
        let mut turn_indices: Vec<usize> = Vec::new();

        for &idx in turn.indices.iter().rev() {
            let tokens = estimate_record_tokens(&messages[idx]);

            if accumulated + tokens > keep_tokens && !tail_indices.is_empty() {
                // Budget exceeded — stop adding more from this turn
                break;
            }

            turn_accumulated += tokens;
            turn_indices.push(idx);
        }

        if accumulated + turn_accumulated > keep_tokens && !tail_indices.is_empty() {
            // This partial turn pushes us over budget; stop here.
            // The remaining items in this turn and earlier turns become head.
            let overflow: Vec<usize> = turn
                .indices
                .iter()
                .copied()
                .filter(|i| !turn_indices.contains(i))
                .collect();
            let mut full_head: Vec<usize> = head_indices.into_iter().chain(overflow).collect();
            full_head.sort_unstable();

            tail_indices.reverse();

            let head: Vec<MessageRecord> = full_head.iter().map(|&i| messages[i].clone()).collect();
            let tail: Vec<MessageRecord> =
                tail_indices.iter().map(|&i| messages[i].clone()).collect();
            let tail_start_id = tail.first().map(|m| m.id);

            return SelectionResult {
                head,
                tail,
                tail_start_id,
            };
        }

        accumulated += turn_accumulated;

        // Reverse to get chronological order inside this turn
        turn_indices.reverse();
        let mut combined = turn_indices;
        combined.append(&mut tail_indices);
        tail_indices = combined;
    }

    // If we got here, all preserved turns fit within budget.
    // Any overflow from an earlier incomplete accumulation goes to head.
    tail_indices.sort_unstable();

    let head: Vec<MessageRecord> = head_indices.iter().map(|&i| messages[i].clone()).collect();
    let tail: Vec<MessageRecord> = tail_indices.iter().map(|&i| messages[i].clone()).collect();
    let tail_start_id = tail.first().map(|m| m.id);

    SelectionResult {
        head,
        tail,
        tail_start_id,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::provider::ContentPart;
    use uuid::Uuid;

    fn make_msg(role: crate::store::MessageRole, text: &str) -> MessageRecord {
        MessageRecord::new(Uuid::now_v7(), role, vec![ContentPart::text(text)])
    }

    fn make_user(text: &str) -> MessageRecord {
        make_msg(crate::store::MessageRole::User, text)
    }

    fn make_assistant(text: &str) -> MessageRecord {
        make_msg(crate::store::MessageRole::Assistant, text)
    }

    fn make_tool(text: &str) -> MessageRecord {
        make_msg(crate::store::MessageRole::Tool, text)
    }

    #[test]
    fn empty_messages_returns_empty() {
        let result = select(&[], 2, 8_000);
        assert!(result.head.is_empty());
        assert!(result.tail.is_empty());
        assert!(result.tail_start_id.is_none());
    }

    #[test]
    fn single_turn_within_budget() {
        let messages = vec![make_user("Hello"), make_assistant("Hi there!")];

        let total = estimate_record_tokens(&messages[0]) + estimate_record_tokens(&messages[1]);

        // Budget is large enough
        let result = select(&messages, 2, total * 2);
        assert!(result.head.is_empty());
        assert_eq!(result.tail.len(), 2);
    }

    #[test]
    fn multiple_turns_preserves_recent() {
        let messages = vec![
            make_user("Turn 1"),
            make_assistant("Response 1"),
            make_user("Turn 2"),
            make_assistant("Response 2"),
            make_user("Turn 3"),
            make_assistant("Response 3"),
        ];

        // Keep only 1 turn
        let result = select(&messages, 1, 8_000);
        assert!(!result.head.is_empty(), "head should have older turns");
        assert!(!result.tail.is_empty(), "tail should have recent turn");
        // Tail should start with Turn 3
        assert_eq!(result.tail[0].role, crate::store::MessageRole::User);
    }

    #[test]
    fn compact_tool_messages_are_in_turn() {
        let messages = vec![
            make_user("Use tool"),
            make_assistant("Calling tool..."),
            make_tool("Tool result"),
            make_assistant("Done with tool"),
        ];

        // All messages should be in one turn (one user message)
        let result = select(&messages, 2, 8_000);
        // head should be empty since it's just one turn
        assert!(result.head.is_empty());
        assert_eq!(result.tail.len(), 4);
    }

    #[test]
    fn tiny_budget_forces_compaction() {
        let messages: Vec<MessageRecord> = (0..5)
            .flat_map(|i| {
                vec![
                    make_user(&format!("Question {i}")),
                    make_assistant(&format!("Answer {i}")),
                ]
            })
            .collect();

        // Extremely tight budget
        let result = select(&messages, 2, 10);
        // Head should contain most messages
        assert!(!result.head.is_empty());
        // Tail should still have at least the last user message
        assert!(!result.tail.is_empty());
    }
}
