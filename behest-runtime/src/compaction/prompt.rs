//! Anchored summary prompt template for compaction.
//!
//! The compaction LLM is asked to produce a structured summary using
//! the anchored summary format. When a previous compaction exists, the
//! model is instructed to update it rather than regenerate from scratch.
//!
//! Ported from OpenCode V1/V2:
//! - `packages/core/src/session/compaction.ts` (V2 anchored summary template)
//! - `packages/opencode/src/agent/prompt/compaction.txt` (V1 compaction agent prompt)

use std::fmt::Write;

use crate::token::estimate_record_tokens;
use behest_store::MessageRecord;

/// The maximum number of characters to include from tool outputs in the
/// compaction prompt. Longer outputs are truncated to avoid bloating the
/// compaction request.
const TOOL_OUTPUT_MAX_CHARS: usize = 2_000;

/// Anchored summary template instructing the compaction LLM.
///
/// The placeholder `{previous_summary}` is replaced with the prior
/// compaction's summary text (for incremental updates). If absent, the
/// model is instructed to create a new summary.
const COMPACTION_PROMPT_TEMPLATE: &str = "\
You are a conversation summarizer. Your task is to produce a structured, \
dense summary of the conversation history below so another instance of \
the same model can pick up where it left off.

{previous_instruction}

Output ONLY the summary — no preamble, no commentary, no markdown code fences.

## Goal
<!-- Single-sentence description of the task the user is trying to accomplish -->

## Constraints & Preferences
<!-- Explicit constraints, preferences, style guides, or rules mentioned -->

## Progress
### Done
<!-- What has been completed so far -->
### In Progress
<!-- What is currently being worked on -->
### Blocked
<!-- Anything that is blocked and why -->

## Key Decisions
<!-- Important technical or design decisions made during the conversation -->

## Next Steps
<!-- What the model should do next, in priority order -->

## Critical Context
<!-- Any context the model MUST know to continue (file paths, error messages, \
API responses, etc.) -->

## Relevant Files
<!-- Files that were created, modified, or discussed. Format: path:line_number -->
";

/// Builds the full compaction prompt from messages to summarise and
/// an optional previous summary for incremental updates.
#[must_use]
pub fn build_prompt(
    messages_to_compact: &[MessageRecord],
    previous_summary: Option<&str>,
) -> String {
    let previous_instruction = match previous_summary {
        Some(prev) => format!(
            "You are updating an existing summary. \
Below is the previous summary — keep what is still relevant, \
update what has changed, and add new information since the last compaction.\n\n\
## Previous Summary\n```\n{prev}\n```"
        ),
        None => "Create a new anchored summary from the conversation below.".to_owned(),
    };

    let prompt =
        COMPACTION_PROMPT_TEMPLATE.replace("{previous_instruction}", &previous_instruction);

    let messages_text = serialize_messages(messages_to_compact);

    format!("{prompt}\n## Messages to Summarize\n{messages_text}")
}

/// Serialises a slice of messages into a readable text format for the
/// compaction LLM. Tool outputs are truncated to `TOOL_OUTPUT_MAX_CHARS`.
fn serialize_messages(messages: &[MessageRecord]) -> String {
    let mut buf = String::new();

    for msg in messages {
        let role_label = match msg.role {
            behest_store::MessageRole::System => "[System]",
            behest_store::MessageRole::User => "[User]",
            behest_store::MessageRole::Assistant => "[Assistant]",
            behest_store::MessageRole::Tool => "[Tool Result]",
            _ => "[Message]",
        };

        buf.push_str(role_label);
        buf.push_str(": ");

        for part in &msg.content {
            match part {
                behest_provider::ContentPart::Text { text, .. } => {
                    let text = truncate_if_too_long(text, TOOL_OUTPUT_MAX_CHARS);
                    buf.push_str(&text);
                }
                behest_provider::ContentPart::Json { value, .. } => {
                    let json_str = value.to_string();
                    let json_str = truncate_if_too_long(&json_str, TOOL_OUTPUT_MAX_CHARS);
                    buf.push_str(&json_str);
                }
                behest_provider::ContentPart::ImageUrl { url, .. } => {
                    let _ = write!(buf, "[Image: {url}]");
                }
                _ => {}
            }
        }

        for tc in &msg.tool_calls {
            let _ = write!(
                buf,
                "\n  [Tool Call: {}({})]",
                tc.name,
                truncate_if_too_long(&tc.arguments.to_string(), 500)
            );
        }

        buf.push('\n');
        // Estimate token count and stop if the prompt is getting too long.
        // The compaction model itself has a context limit; we stop building
        // the prompt at ~80% of a reasonable compaction model context (64K).
        if estimate_record_tokens(msg) > 0 && crate::token::estimate_tokens(&buf) > 50_000 {
            buf.push_str(
                "\n[... further messages truncated to stay within compaction model context ...]\n",
            );
            break;
        }
    }

    buf
}

fn truncate_if_too_long(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_owned()
    } else {
        let truncated: String = text.chars().take(max_chars).collect();
        format!(
            "{truncated}\n[truncated: omitted {} chars]",
            text.len() - max_chars
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use behest_provider::ContentPart;
    use uuid::Uuid;

    fn make_user_record(text: &str) -> MessageRecord {
        MessageRecord::new(
            Uuid::now_v7(),
            behest_store::MessageRole::User,
            vec![ContentPart::text(text)],
        )
    }

    #[test]
    fn build_prompt_without_previous_summary() {
        let messages = vec![make_user_record("Hello, can you help me write a function?")];
        let prompt = build_prompt(&messages, None);
        assert!(prompt.contains("Create a new anchored summary"));
        assert!(prompt.contains("## Goal"));
        assert!(prompt.contains("Hello, can you help me write a function?"));
    }

    #[test]
    fn build_prompt_with_previous_summary() {
        let messages = vec![make_user_record("Now add error handling.")];
        let prev = "## Goal\nWrite a function\n## Progress\n### Done\nCreated function";
        let prompt = build_prompt(&messages, Some(prev));
        assert!(prompt.contains("updating an existing summary"));
        assert!(prompt.contains("## Previous Summary"));
        assert!(prompt.contains("Created function"));
    }

    #[test]
    fn serialize_includes_role_labels() {
        let records = vec![
            MessageRecord::new(
                Uuid::now_v7(),
                behest_store::MessageRole::User,
                vec![ContentPart::text("Hi")],
            ),
            MessageRecord::new(
                Uuid::now_v7(),
                behest_store::MessageRole::Assistant,
                vec![ContentPart::text("Hello!")],
            ),
        ];
        let text = serialize_messages(&records);
        assert!(text.contains("[User]: Hi"));
        assert!(text.contains("[Assistant]: Hello!"));
    }

    #[test]
    fn truncate_long_content() {
        let long = "x".repeat(3_000);
        let result = truncate_if_too_long(&long, 2_000);
        assert!(result.len() < 3_000);
        assert!(result.contains("[truncated"));
    }

    #[test]
    fn truncate_empty() {
        assert_eq!(truncate_if_too_long("", 100), "");
    }
}
