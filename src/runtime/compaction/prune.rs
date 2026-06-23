//! Tool output pruning.
//!
//! When conversation history grows long, older tool call outputs are
//! marked as "pruned" — their content is replaced with a placeholder
//! string to free context tokens without permanently deleting data.
//!
//! Pruning is distinct from compaction: pruning operates on individual
//! tool output messages (cheap, no LLM call), while compaction
//! summarises entire conversation segments (expensive, LLM call).
//!
//! Ported from OpenCode V1: `packages/opencode/src/session/compaction.ts` prune().

use crate::store::MessageRecord;
use crate::token::estimate_record_tokens;

/// Minimum number of tokens that must be pruned to make it worthwhile.
/// Pruning fewer tokens than this is skipped.
const PRUNE_MINIMUM: usize = 20_000;

/// Number of recent tokens to protect from pruning.
const PRUNE_PROTECT: usize = 40_000;

/// Tool names whose outputs are never pruned.
const PRUNE_PROTECTED_TOOLS: &[&str] = &["skill"];

/// Placeholder text replacing pruned tool outputs.
pub const PRUNED_PLACEHOLDER: &str = "[Old tool result content cleared]";

/// Result of a pruning pass.
#[derive(Debug, Clone)]
pub struct PruneResult {
    /// Number of tool output messages marked as pruned.
    pub pruned_count: usize,
    /// Estimated number of tokens freed.
    pub tokens_freed: usize,
}

/// Indices of messages to prune, collected before mutation to ensure
/// atomicity with respect to the [`PRUNE_MINIMUM`] threshold.
#[derive(Debug, Clone, Copy)]
struct PruneTarget {
    index: usize,
}

/// Marks old tool output messages for pruning.
///
/// Two-pass strategy: first pass collects indices of messages that
/// *would* be pruned and computes the total token savings. Only if
/// savings exceed [`PRUNE_MINIMUM`] does the second pass apply the
/// in-place mutations. This prevents partial data corruption when
/// the minimum threshold is not met.
///
/// Walks backward through messages, protecting the most recent
/// `PRUNE_PROTECT` tokens worth of tool outputs. Older tool outputs
/// with sufficient total token savings are marked as pruned.
///
/// # Arguments
/// * `messages` - Mutable slice of messages (in chronological order).
/// * `skip_turns` - Number of recent turns to skip entirely (default 2).
///
/// # Returns
/// The number of messages marked for pruning and estimated tokens freed.
#[must_use]
pub fn prune_tool_outputs(messages: &mut [MessageRecord], skip_turns: usize) -> PruneResult {
    if messages.is_empty() {
        return PruneResult {
            pruned_count: 0,
            tokens_freed: 0,
        };
    }

    // Count turns from the end to skip
    let mut turn_count = 0usize;
    let mut skip_until: Option<usize> = None;

    for (i, msg) in messages.iter().enumerate().rev() {
        if !msg.is_compaction && msg.role == crate::store::MessageRole::User {
            turn_count += 1;
            if turn_count >= skip_turns {
                skip_until = Some(i);
                break;
            }
        }
    }

    // --- Pass 1: collect targets, check threshold ---
    let mut protected = 0usize;
    let mut targets: Vec<PruneTarget> = Vec::new();
    let mut estimated_freed = 0usize;

    for (idx, msg) in messages.iter().enumerate().rev() {
        // Stop if we hit a compaction summary
        if msg.is_summary {
            break;
        }

        // Skip protected recent turns
        if skip_until.is_some() && protected < PRUNE_PROTECT {
            protected += estimate_record_tokens(msg);
            continue;
        }

        // Only prune tool messages
        if msg.role != crate::store::MessageRole::Tool {
            continue;
        }

        // Never prune protected tools
        if let Some(ref tool_name) = msg.tool_name {
            if PRUNE_PROTECTED_TOOLS.contains(&tool_name.as_str()) {
                continue;
            }
        }

        // Skip already pruned messages
        if msg.is_summary {
            continue;
        }

        let tokens = estimate_record_tokens(msg);

        if protected < PRUNE_PROTECT {
            protected += tokens;
            continue;
        }

        targets.push(PruneTarget { index: idx });
        estimated_freed += tokens;
    }

    if estimated_freed < PRUNE_MINIMUM {
        return PruneResult {
            pruned_count: 0,
            tokens_freed: 0,
        };
    }

    // --- Pass 2: apply mutations ---
    for target in &targets {
        let msg = &mut messages[target.index];
        msg.is_summary = true;
        msg.content = vec![crate::provider::ContentPart::text(PRUNED_PLACEHOLDER)];
    }

    PruneResult {
        pruned_count: targets.len(),
        tokens_freed: estimated_freed,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::provider::ContentPart;
    use uuid::Uuid;

    fn make_tool_msg(name: &str, output: &str) -> MessageRecord {
        MessageRecord {
            id: Uuid::now_v7(),
            session_id: Uuid::now_v7(),
            role: crate::store::MessageRole::Tool,
            content: vec![ContentPart::text(output)],
            tool_calls: Vec::new(),
            tool_call_id: Some("call_1".to_owned()),
            tool_name: Some(name.to_owned()),
            usage: None,
            created_at: chrono::Utc::now(),
            is_compaction: false,
            is_summary: false,
            compaction_meta: None,
        }
    }

    #[test]
    fn empty_messages() {
        let mut messages = Vec::new();
        let result = prune_tool_outputs(&mut messages, 2);
        assert_eq!(result.pruned_count, 0);
    }

    #[test]
    fn skill_tools_are_protected() {
        let mut messages = vec![make_tool_msg("skill", "important skill output")];
        let result = prune_tool_outputs(&mut messages, 0);
        assert_eq!(result.pruned_count, 0);
    }

    #[test]
    fn small_output_not_pruned() {
        let mut messages = vec![make_tool_msg("read_file", "short")];
        let result = prune_tool_outputs(&mut messages, 0);
        // PRUNE_MINIMUM is 20_000 — a single short tool output won't reach it
        assert_eq!(result.pruned_count, 0);
    }
}
