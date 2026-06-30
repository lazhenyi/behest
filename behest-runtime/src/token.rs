//! Token estimation using character-based heuristics.
//!
//! Core functions re-exported from `behest-core`.
//! Store-dependent estimation functions are defined here.

pub use behest_core::token::{
    estimate_content_part_tokens, estimate_message_tokens, estimate_messages_tokens,
    estimate_tokens, estimate_tool_call_tokens,
};

use behest_store::MessageRecord;

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

    let role_overhead = match record.role {
        behest_store::MessageRole::System
        | behest_store::MessageRole::User
        | behest_store::MessageRole::Assistant => 8,
        behest_store::MessageRole::Tool => 10,
        _ => 8,
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
    use behest_core::message::ContentPart;
    use uuid::Uuid;

    #[test]
    fn estimate_record() {
        let record = MessageRecord::new(
            Uuid::now_v7(),
            behest_store::MessageRole::User,
            vec![ContentPart::text("Hello")],
        );
        let tokens = estimate_record_tokens(&record);
        assert_eq!(tokens, 10);
    }

    #[test]
    fn estimate_records_slice() {
        let records = vec![
            MessageRecord::new(
                Uuid::now_v7(),
                behest_store::MessageRole::System,
                vec![ContentPart::text("System")],
            ),
            MessageRecord::new(
                Uuid::now_v7(),
                behest_store::MessageRole::User,
                vec![ContentPart::text("User")],
            ),
        ];
        let total = estimate_records_tokens(&records);
        assert!(total > 0);
    }
}
