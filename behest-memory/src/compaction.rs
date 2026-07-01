//! Three-tier context compaction strategies.
//!
//! All compaction functions are pure: they take messages and a policy, and
//! return modified messages plus a report. No I/O, no LLM calls — the user
//! is responsible for invoking an LLM for full-compact summary generation
//! and calling [`apply_summary`] with the result.
//!
//! # Tier overview (cost-ascending)
//!
//! 1. **Microcompact** — zero LLM cost. Clears old tool outputs, replaces
//!    them with one-line structured summaries (`[tool_name] args -> result`).
//! 2. **Autocompact** — zero LLM cost. Drops the oldest message groups
//!    (split at user messages), keeping the most recent groups up to a
//!    token budget.
//! 3. **Full compact** — user-provided LLM summary. The user calls
//!    [`compact_candidates`] to get messages for summarization, then
//!    calls [`apply_summary`] with the LLM-generated text.
//!
//! # Example
//!
//! ```ignore
//! use behest_memory::compaction;
//! use behest_core::budget::TokenBudget;
//!
//! // Tier 1: try microcompact first
//! let report = compaction::microcompact(&mut messages, MicrocompactConfig::default());
//!
//! // Tier 2: if still over budget, try autocompact
//! let budget = TokenBudget::new(100_000, 20_000);
//! let report = compaction::autocompact(&mut messages, &budget);
//!
//! // Tier 3: if still over budget, user gets candidates for LLM summarization
//! let candidates = compaction::compact_candidates(&messages, 6);
//! // ... user sends candidates to LLM, gets summary ...
//! compaction::apply_summary(&mut messages, &summary, 6);
//! ```

use behest_core::budget::TokenBudget;
use behest_core::message::Message;
use behest_core::token::estimate_message_tokens;

// ── Tier 1: Microcompact ──

/// Configuration for [`microcompact`].
#[derive(Debug, Clone, Copy)]
pub struct MicrocompactConfig {
    /// Number of most recent messages to always preserve verbatim.
    pub keep_recent: usize,
    /// Maximum number of recent shell/tool outputs to preserve verbatim.
    pub keep_recent_tool_outputs: usize,
}

impl Default for MicrocompactConfig {
    fn default() -> Self {
        Self {
            keep_recent: 6,
            keep_recent_tool_outputs: 8,
        }
    }
}

/// Report from a [`microcompact`] operation.
#[derive(Debug, Clone, Default)]
pub struct MicrocompactReport {
    /// Number of messages whose content was replaced.
    pub changed_messages: usize,
    /// Estimated token savings.
    pub tokens_freed: usize,
}

/// Clears old tool output content, replacing it with one-line summaries.
///
/// This is the cheapest compaction tier — no LLM calls, no message removal.
/// Only tool-result messages are affected; user and assistant messages are
/// left untouched. The most recent `config.keep_recent` messages and
/// `config.keep_recent_tool_outputs` tool results are preserved verbatim.
///
/// Cleared tool outputs are replaced with a structured one-line summary:
/// `[tool_name] args_summary -> result_summary`
#[must_use]
pub fn microcompact(messages: &mut [Message], config: MicrocompactConfig) -> MicrocompactReport {
    if messages.len() <= config.keep_recent {
        return MicrocompactReport::default();
    }

    let before_tokens: usize = messages.iter().map(estimate_message_tokens).sum();

    let recent_start = messages.len().saturating_sub(config.keep_recent);
    let mut changed = 0usize;

    // Count tool outputs from the end to know which to preserve
    // Pre-compute: which tool outputs are "recent"?
    let mut tool_output_indices: Vec<usize> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if matches!(msg, Message::Tool { .. }) {
            tool_output_indices.push(i);
        }
    }
    let tool_keep_start = tool_output_indices
        .len()
        .saturating_sub(config.keep_recent_tool_outputs);
    let tool_keep: std::collections::HashSet<usize> = tool_output_indices
        .iter()
        .skip(tool_keep_start)
        .copied()
        .collect();

    #[allow(clippy::needless_range_loop)]
    for i in 0..messages.len() {
        // Always preserve recent messages
        if i >= recent_start {
            continue;
        }

        let replacement = match &messages[i] {
            Message::Tool {
                name,
                content,
                tool_call_id: _,
            } if !tool_keep.contains(&i) => {
                let summary = summarize_tool_output(name, content);
                // Preserve tool message role but replace content with summary
                Some(Message::Tool {
                    tool_call_id: format!("compacted-{i}"),
                    name: name.clone(),
                    content: vec![behest_core::message::ContentPart::text(summary)],
                })
            }
            _ => None,
        };

        if let Some(replacement) = replacement {
            changed += 1;
            messages[i] = replacement;
        }
    }

    let after_tokens: usize = messages.iter().map(estimate_message_tokens).sum();
    let tokens_freed = before_tokens.saturating_sub(after_tokens);

    MicrocompactReport {
        changed_messages: changed,
        tokens_freed,
    }
}

/// Generates a one-line summary of a tool output.
///
/// Format: `[tool_name] args... -> exit N, M lines` or
/// `[tool_name] args... -> N chars`
fn summarize_tool_output(name: &str, content: &[behest_core::message::ContentPart]) -> String {
    let text: String = content
        .iter()
        .filter_map(|part| match part {
            behest_core::message::ContentPart::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ");

    // Try to extract exit code if present
    let exit_info = text
        .lines()
        .find(|line| line.contains("exit") || line.contains("returncode"))
        .map(|line| {
            line.trim()
                .trim_start_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        });

    let line_count = text.lines().count();
    let char_count = text.len();

    if let Some(exit) = exit_info {
        format!("[{name}] → {exit}, {line_count} lines")
    } else if line_count > 1 {
        format!("[{name}] → {line_count} lines, {char_count} chars")
    } else {
        let truncated: String = text.chars().take(80).collect();
        if text.len() > 80 {
            format!("[{name}] → {truncated}...")
        } else {
            format!("[{name}] → {truncated}")
        }
    }
}

// ── Tier 2: Autocompact ──

/// Report from an [`autocompact`] operation.
#[derive(Debug, Clone, Default)]
pub struct AutocompactReport {
    /// Number of message groups dropped.
    pub groups_dropped: usize,
    /// Number of groups kept.
    pub groups_kept: usize,
    /// Estimated token savings.
    pub tokens_freed: usize,
}

/// Drops the oldest message groups to fit within a token budget.
///
/// Messages are partitioned into groups at user-message boundaries. The
/// oldest groups are dropped until the remaining messages fit within
/// `budget.effective_window()`. At least one group is always kept.
///
/// Returns a report; messages are modified in place.
#[must_use]
pub fn autocompact(messages: &mut Vec<Message>, budget: &TokenBudget) -> AutocompactReport {
    if messages.is_empty() {
        return AutocompactReport::default();
    }

    let before_tokens: usize = messages.iter().map(estimate_message_tokens).sum();
    let budget_limit = budget.thresholds().compact_threshold;

    if before_tokens <= budget_limit {
        return AutocompactReport {
            groups_kept: 1,
            ..Default::default()
        };
    }

    // Partition into groups at user message boundaries
    let groups = split_into_groups(messages);
    if groups.len() <= 1 {
        return AutocompactReport {
            groups_kept: 1,
            ..Default::default()
        };
    }

    // Keep groups from newest to oldest until we fit the budget
    let mut kept_tokens: usize = 0;
    let mut kept_count: usize = 0;

    for group in groups.iter().rev() {
        let group_tokens: usize = group
            .1
            .iter()
            .map(|i| estimate_message_tokens(&messages[*i]))
            .sum();
        if kept_count > 0 && kept_tokens + group_tokens > budget_limit {
            break;
        }
        kept_tokens += group_tokens;
        kept_count += 1;
    }

    let dropped_count = groups.len() - kept_count;

    if dropped_count == 0 {
        return AutocompactReport {
            groups_kept: kept_count,
            ..Default::default()
        };
    }

    // Drop oldest groups: keep only the last `kept_count` groups
    let keep_start = groups[groups.len() - kept_count].0; // first index of the oldest kept group
    *messages = messages.split_off(keep_start);

    let after_tokens: usize = messages.iter().map(estimate_message_tokens).sum();
    let tokens_freed = before_tokens.saturating_sub(after_tokens);

    AutocompactReport {
        groups_dropped: dropped_count,
        groups_kept: kept_count,
        tokens_freed,
    }
}

/// Splits messages into groups at user-message boundaries.
///
/// Returns `(start_index, Vec<indices>)` for each group. A new group starts
/// at every user message (or at the beginning if the first message is not
/// a user message).
fn split_into_groups(messages: &[Message]) -> Vec<(usize, Vec<usize>)> {
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut current_start: usize = 0;
    let mut current_indices: Vec<usize> = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        if is_user_message(msg) && !current_indices.is_empty() {
            groups.push((current_start, std::mem::take(&mut current_indices)));
            current_start = i;
        }
        current_indices.push(i);
    }

    if !current_indices.is_empty() {
        groups.push((current_start, current_indices));
    }

    groups
}

fn is_user_message(msg: &Message) -> bool {
    matches!(msg, Message::User { .. })
}

// ── Tier 3: Full compact ──

/// Returns the messages that should be summarized by an LLM.
///
/// The most recent `keep_recent` messages are excluded — they will be
/// preserved verbatim. The remaining older messages are candidates for
/// summarization.
///
/// The user should send these candidates to an LLM for summarization,
/// then call [`apply_summary`] with the result.
#[must_use]
pub fn compact_candidates(messages: &[Message], keep_recent: usize) -> Vec<Message> {
    if messages.len() <= keep_recent {
        return Vec::new();
    }
    messages[..messages.len() - keep_recent].to_vec()
}

/// The recommended system prompt for an LLM generating a compaction summary.
///
/// Use this as the system message when calling the model to summarize
/// compact candidates.
#[must_use]
pub fn compaction_system_prompt() -> String {
    "You are a conversation compaction engine. Produce faithful, detailed summaries \
     of agent conversations. Respond with TEXT ONLY — never call tools.\n\
     Your summary must be under the requested token limit.\n\
     Format your response as:\n\
     <summary>\n\
     1. Primary Request and Intent\n\
     2. Key Technical Concepts\n\
     3. Files, Commands, and Tools Used\n\
     4. Errors and Fixes\n\
     5. Problem Solving\n\
     6. All User Messages\n\
     7. Pending Tasks\n\
     8. Current Work\n\
     9. Optional Next Step\n\
     </summary>"
        .to_string()
}

/// The recommended user prompt for an LLM generating a compaction summary.
///
/// When `has_retained_context` is true (there's already a previous summary),
/// the prompt instructs the model to focus on the recent portion only.
#[must_use]
pub fn compaction_user_prompt(has_retained_context: bool) -> String {
    if has_retained_context {
        "Create a detailed summary of the RECENT portion of the conversation only. \
         Earlier retained context will remain available after compaction. \
         Focus on what happened in the recent messages and where the current work stopped. \
         Respond with a <summary> block."
            .to_string()
    } else {
        "Create a detailed summary of the conversation so far so the agent can continue \
         working without losing context. \
         The summary must preserve the user's intent, the technical state, and the exact \
         place where work stopped. \
         Respond with a <summary> block."
            .to_string()
    }
}

/// Report from a [`apply_summary`] operation.
#[derive(Debug, Clone, Default)]
pub struct FullCompactReport {
    /// Number of messages replaced by the summary.
    pub messages_replaced: usize,
    /// Estimated tokens before compaction.
    pub tokens_before: usize,
    /// Estimated tokens after compaction.
    pub tokens_after: usize,
    /// Whether the summary was extracted from a `<summary>` tag.
    pub summary_extracted: bool,
}

/// Applies an LLM-generated summary, replacing old messages.
///
/// The most recent `keep_recent` messages are preserved. All older messages
/// are replaced with a single system message containing the summary.
///
/// If the summary text contains a `<summary>...</summary>` block, only the
/// content inside the tag is used.
#[must_use]
pub fn apply_summary(
    messages: &mut Vec<Message>,
    summary: &str,
    keep_recent: usize,
) -> FullCompactReport {
    if messages.len() <= keep_recent {
        return FullCompactReport::default();
    }

    let before_tokens: usize = messages.iter().map(estimate_message_tokens).sum();

    // Extract summary from tags if present
    let (summary_text, extracted) = extract_summary_tag(summary);

    // Split off the tail (messages to keep)
    let tail = messages.split_off(messages.len() - keep_recent);
    let replaced = std::mem::replace(messages, tail);

    let replaced_count = replaced.len();

    // Insert summary as a system message at the front
    let resume_msg = build_resume_message(&summary_text);
    messages.insert(0, resume_msg);

    let after_tokens: usize = messages.iter().map(estimate_message_tokens).sum();

    FullCompactReport {
        messages_replaced: replaced_count,
        tokens_before: before_tokens,
        tokens_after: after_tokens,
        summary_extracted: extracted,
    }
}

/// Extracts content from inside `<summary>...</summary>` tags.
///
/// Returns `(extracted_or_original, was_extracted)`.
fn extract_summary_tag(text: &str) -> (String, bool) {
    if let Some(start) = text.find("<summary>") {
        let after_open = start + "<summary>".len();
        if let Some(end) = text[after_open..].find("</summary>") {
            let content = text[after_open..after_open + end].trim().to_string();
            if !content.is_empty() {
                return (content, true);
            }
        }
    }
    (text.trim().to_string(), false)
}

/// Builds a resume message from a summary.
fn build_resume_message(summary: &str) -> Message {
    let content = format!(
        "This session is being continued from an earlier conversation that ran out of context.\n\
         \n\
         Summary:\n\
         {summary}\n\
         \n\
         Continue the conversation from where it left off without asking the user to repeat \
         context. Resume directly, do not acknowledge the summary, and do not narrate that \
         you are resuming."
    );
    Message::system_text(content)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use behest_core::budget::TokenBudget;
    use behest_core::message::Message;

    fn tool_msg(name: &str, content: &str) -> Message {
        Message::Tool {
            tool_call_id: format!("call_{name}"),
            name: name.to_string(),
            content: vec![behest_core::message::ContentPart::text(content)],
        }
    }

    #[test]
    fn microcompact_preserves_recent_messages() {
        let mut msgs = vec![
            Message::user_text("q1"),
            tool_msg(
                "echo",
                "some output\n42 lines\n42 lines\n42 lines\n42 lines\n42 lines",
            ),
        ];
        let config = MicrocompactConfig {
            keep_recent: 2,
            ..Default::default()
        };
        let report = microcompact(&mut msgs, config);
        // Both messages are within keep_recent=2, so nothing changes
        assert_eq!(report.changed_messages, 0);
    }

    #[test]
    fn microcompact_clears_old_tool_outputs() {
        let mut msgs = vec![
            Message::user_text("q1"),
            tool_msg("echo", &"very long output\n".repeat(20)),
            Message::user_text("q2"),
            Message::assistant_text("answer"),
        ];
        let config = MicrocompactConfig {
            keep_recent: 2,              // only q2 + answer are recent
            keep_recent_tool_outputs: 0, // don't preserve any tool outputs
        };
        let report = microcompact(&mut msgs, config);
        assert!(report.changed_messages > 0);
        // The old tool message should have been replaced
        let compacted = &msgs[1];
        if let Message::Tool { content, .. } = compacted {
            if let behest_core::message::ContentPart::Text { text, .. } = &content[0] {
                assert!(
                    text.contains("[echo]"),
                    "expected [echo] prefix, got: {text}"
                );
            }
        } else {
            panic!("expected tool message, got: {compacted:?}");
        }
    }

    #[test]
    fn microcompact_preserves_non_tool_messages() {
        let mut msgs = vec![
            Message::user_text("q1"),
            Message::assistant_text("answer1"),
            tool_msg("bash", "output1"),
            Message::user_text("q2"),
            Message::assistant_text("answer2"),
        ];
        let config = MicrocompactConfig {
            keep_recent: 2, // only q2 + answer2 are recent
            ..Default::default()
        };
        let _report = microcompact(&mut msgs, config);
        // User messages should be untouched
        assert!(matches!(msgs[0], Message::User { .. }));
        assert!(matches!(msgs[3], Message::User { .. }));
    }

    #[test]
    fn autocompact_drops_oldest_groups() {
        // MIN_EFFECTIVE_WINDOW=1000 means we need enough tokens to exceed ~670 (compact threshold)
        // Each char is ~0.25 tokens; 2000 chars = ~500 tokens per message
        let long_text = "x".repeat(2000);
        let mut msgs = vec![
            Message::user_text(&long_text),
            Message::assistant_text(&long_text),
            Message::user_text(&long_text),
            Message::assistant_text(&long_text),
            Message::user_text(&long_text),
            Message::assistant_text(&long_text),
        ];
        let budget = TokenBudget::new(2000, 0);
        let total_tokens: usize = msgs.iter().map(estimate_message_tokens).sum();
        let report = autocompact(&mut msgs, &budget);
        assert!(
            report.groups_dropped > 0,
            "total_tokens={total_tokens} groups_dropped={} groups_kept={}",
            report.groups_dropped,
            report.groups_kept
        );
        assert!(report.groups_kept >= 1);
        assert!(msgs.len() < 6);
    }

    #[test]
    fn autocompact_keeps_all_when_under_budget() {
        let mut msgs = vec![Message::user_text("q1"), Message::assistant_text("a1")];
        let budget = TokenBudget::new(1_000_000, 0);
        let report = autocompact(&mut msgs, &budget);
        assert_eq!(report.groups_dropped, 0);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn compact_candidates_excludes_recent() {
        let msgs = vec![
            Message::user_text("old1"),
            Message::assistant_text("old2"),
            Message::user_text("new1"),
            Message::assistant_text("new2"),
        ];
        let candidates = compact_candidates(&msgs, 2);
        assert_eq!(candidates.len(), 2);
        assert!(matches!(candidates[0], Message::User { .. }));
    }

    #[test]
    fn compact_candidates_empty_when_all_recent() {
        let msgs = vec![Message::user_text("hi"), Message::assistant_text("hey")];
        let candidates = compact_candidates(&msgs, 3);
        assert!(candidates.is_empty());
    }

    #[test]
    fn apply_summary_replaces_old_messages() {
        let mut msgs = vec![
            Message::user_text("old question"),
            Message::assistant_text("old answer"),
            Message::user_text("new question"),
            Message::assistant_text("new answer"),
        ];
        let report = apply_summary(
            &mut msgs,
            "User asked about X. Assistant did Y.",
            2, // keep last 2
        );
        assert_eq!(report.messages_replaced, 2);
        assert_eq!(msgs.len(), 3); // summary + 2 recent
        // First message should be a system message with the summary
        if let Message::System { content } = &msgs[0] {
            if let behest_core::message::ContentPart::Text { text, .. } = &content[0] {
                assert!(
                    text.contains("being continued"),
                    "expected resume message, got: {text}"
                );
            }
        } else {
            panic!("expected system message, got: {:?}", msgs[0]);
        }
    }

    #[test]
    fn extract_summary_tag_works() {
        let input = "Some text\n<summary>\nThe actual summary\n</summary>\nMore text";
        let (result, extracted) = extract_summary_tag(input);
        assert!(extracted);
        assert_eq!(result, "The actual summary");
    }

    #[test]
    fn extract_summary_tag_falls_back_to_full_text() {
        let input = "Just a plain summary, no tags";
        let (result, extracted) = extract_summary_tag(input);
        assert!(!extracted);
        assert_eq!(result, input.trim());
    }

    #[test]
    fn summarize_tool_output_handles_exit_code() {
        let content = vec![behest_core::message::ContentPart::text(
            "installing packages...\nexit 0\n42 lines total",
        )];
        let result = summarize_tool_output("bash", &content);
        assert!(result.contains("[bash]"));
        assert!(result.contains("exit 0"));
    }

    #[test]
    fn summarize_tool_output_handles_multiline() {
        let content = vec![behest_core::message::ContentPart::text(
            "line1\nline2\nline3\nline4\nline5\nline6",
        )];
        let result = summarize_tool_output("read_file", &content);
        assert!(result.contains("[read_file]"));
        assert!(result.contains("6 lines"));
    }

    #[test]
    fn empty_messages_no_compaction() {
        let mut msgs: Vec<Message> = Vec::new();
        let report = microcompact(&mut msgs, MicrocompactConfig::default());
        assert_eq!(report.changed_messages, 0);

        let report = autocompact(&mut msgs, &TokenBudget::default());
        assert_eq!(report.groups_dropped, 0);
    }

    #[test]
    fn compaction_prompts_contain_required_sections() {
        let system = compaction_system_prompt();
        assert!(system.contains("Primary Request"));
        assert!(system.contains("Key Technical Concepts"));
        assert!(system.contains("Pending Tasks"));
        assert!(system.contains("Current Work"));
        assert!(system.contains("Optional Next Step"));

        let user_prompt = compaction_user_prompt(false);
        assert!(user_prompt.contains("detailed summary"));

        let partial_prompt = compaction_user_prompt(true);
        assert!(partial_prompt.contains("RECENT portion"));
    }
}
