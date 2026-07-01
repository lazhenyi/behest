//! LoopDriver recovery — explicit helpers for common failure recovery.
//!
//! When an agent turn fails, the caller decides how to recover. These
//! helpers inject the right recovery messages into a [`Journal`] and provide
//! snip operations for prompt-too-long situations. There is no automatic
//! retry loop — the user inspects the failure, picks a helper, calls it,
//! and then retries the turn explicitly.
//!
//! # Failure → helper mapping
//!
//! | Failure | Helper | Effect |
//! |---------|--------|--------|
//! | `PromptTooLong` | [`snip_before_offset`] | Drops journal entries before an offset |
//! | `MaxTokensExhausted` | [`inject_max_tokens_recovery`] | Tells the model to resume mid-thought |
//! | `PermissionDenied` | [`inject_permission_retry`] | Tells the model to retry with safer input |
//! | Unfinished task | [`inject_completion_reminder`] | Reminds the model of pending requirements |
//!
//! # Example
//!
//! ```ignore
//! use behest_core::journal::Journal;
//! use behest_core::recovery;
//!
//! let mut journal = Journal::new();
//! // ... turn fails with MaxTokensExhausted ...
//! recovery::inject_max_tokens_recovery(&mut journal, 1);
//! // user retries the turn
//! ```

use crate::journal::{Journal, SessionEvent};
use crate::message::Message;

/// A completion requirement the agent must satisfy before finishing.
///
/// Used by [`inject_completion_reminder`] to remind the model of unfinished
/// work (e.g., a file that must be created).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionRequirement {
    /// A workspace file that must be created or updated.
    WorkspaceFile {
        /// Required file path, or `None` for "any file".
        path: Option<String>,
    },
}

impl CompletionRequirement {
    /// Creates a requirement for a specific file path.
    #[must_use]
    pub fn file(path: impl Into<String>) -> Self {
        Self::WorkspaceFile {
            path: Some(path.into()),
        }
    }

    /// Creates a requirement for any file write.
    #[must_use]
    pub fn any_file() -> Self {
        Self::WorkspaceFile { path: None }
    }

    /// Renders the requirement as a human-readable instruction line.
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            Self::WorkspaceFile { path: Some(path) } => {
                format!("You must create or update the workspace file `{path}` before replying with a final answer.")
            }
            Self::WorkspaceFile { path: None } => {
                "You must create or update a file in the workspace containing the requested result before replying with a final answer.".to_string()
            }
        }
    }
}

/// Injects a max-tokens recovery message into the journal.
///
/// Use this when the model's output was truncated by `max_tokens`. The
/// injected message tells the model to resume mid-thought without apologizing
/// or recapping. Returns the offset of the injected message.
///
/// # Example
///
/// ```ignore
/// use behest_core::journal::Journal;
/// use behest_core::recovery;
///
/// let mut journal = Journal::new();
/// let offset = recovery::inject_max_tokens_recovery(&mut journal, 3);
/// ```
pub fn inject_max_tokens_recovery(journal: &mut Journal, turn: usize) -> u64 {
    let msg = Message::user_text(max_tokens_recovery_message(turn));
    journal.record(SessionEvent::MessageAppended { message: msg })
}

/// Injects a permission-denied retry message into the journal.
///
/// Use this when a tool call was denied by the permission gate and the hook
/// requested a retry. The injected message tells the model to retry with
/// safer input or a narrower scope. Returns the offset of the injected
/// message.
pub fn inject_permission_retry(journal: &mut Journal, turn: usize, tool_names: &[String]) -> u64 {
    let msg = Message::user_text(permission_retry_message(turn, tool_names));
    journal.record(SessionEvent::MessageAppended { message: msg })
}

/// Injects a completion-requirement reminder into the journal.
///
/// Use this when a run ended but `CompletionRequirement`s remain unsatisfied.
/// The injected message lists the requirements and tells the model to use a
/// filesystem tool now. Returns the offset of the injected message.
pub fn inject_completion_reminder(
    journal: &mut Journal,
    turn: usize,
    requirements: &[CompletionRequirement],
) -> u64 {
    let msg = Message::user_text(completion_reminder_message(turn, requirements));
    journal.record(SessionEvent::MessageAppended { message: msg })
}

/// Snips (drops) all journal entries before the given offset.
///
/// Use this when the provider returns a `prompt_too_long` error. The
/// `suggested_snip_offset` from the error tells you where to cut. After
/// snipping, the next `build_prompt` call produces a smaller prompt.
///
/// Returns the number of entries removed. **Note:** this rewrites the
/// journal's history — entries are permanently removed. Use a
/// [`SessionCheckpoint`](crate::journal::SessionCheckpoint) instead if you
/// need to preserve the compacted summary.
///
/// # Example
///
/// ```ignore
/// use behest_core::journal::Journal;
/// use behest_core::recovery;
///
/// let mut journal = Journal::new();
/// // ... entries 0..10 ...
/// let removed = recovery::snip_before_offset(&mut journal, 5);
/// assert_eq!(removed, 5);
/// ```
pub fn snip_before_offset(journal: &mut Journal, offset: u64) -> usize {
    let before = journal.len();
    if before == 0 {
        return 0;
    }
    // Keep only entries at or after the offset (clone to release the borrow)
    let kept: Vec<_> = journal
        .entries()
        .iter()
        .filter(|e| e.offset >= offset)
        .cloned()
        .collect();
    let removed = before - kept.len();
    // Rebuild the journal from the kept entries, preserving offsets
    *journal = Journal::from_entries(kept);
    removed
}

/// Builds a checkpoint that compacts all entries before `until_offset`,
/// preserving the given messages as the recent tail.
///
/// Use this as a non-destructive alternative to [`snip_before_offset`]: the
/// old entries are summarized into a checkpoint rather than dropped. The
/// caller provides the summary text (typically LLM-generated).
///
/// Returns the new checkpoint. The caller should record it via
/// `journal.record(SessionEvent::CheckpointCreated { checkpoint })`.
#[must_use]
pub fn build_compaction_checkpoint(
    until_offset: u64,
    summary: String,
    preserved_messages: Vec<Message>,
) -> crate::journal::SessionCheckpoint {
    crate::journal::SessionCheckpoint {
        compacted_until_offset: until_offset,
        summary: Some(summary),
        preserved_messages,
    }
}

// ── Message templates (from kheish engine.rs) ──

/// The max-tokens recovery message: tells the model to resume mid-thought.
#[must_use]
pub fn max_tokens_recovery_message(turn: usize) -> String {
    let _ = turn;
    "Output token limit hit. Resume directly — no apology, no recap of what \
     you were doing. Pick up mid-thought if that is where the cut happened. \
     Break remaining work into smaller pieces."
        .to_string()
}

/// The permission-denied retry message.
#[must_use]
pub fn permission_retry_message(turn: usize, tool_names: &[String]) -> String {
    let _ = turn;
    let list = tool_names.join(", ");
    format!(
        "A permission-denied hook requested one retry for the blocked tool \
         call(s): {list}.\n\
         Continue from where you left off. If you still need the action, \
         retry with safer input or a narrower scope instead of stopping."
    )
}

/// The completion-requirement reminder message.
#[must_use]
pub fn completion_reminder_message(turn: usize, requirements: &[CompletionRequirement]) -> String {
    let _ = turn;
    let mut lines = vec![
        "Continue from where you left off.".to_string(),
        "The task is not complete yet.".to_string(),
    ];
    for req in requirements {
        lines.push(req.render());
    }
    lines.push("Use a filesystem tool now instead of describing the next step.".to_string());
    lines.join("\n")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::journal::{Journal, SessionEvent};
    use crate::message::{ContentPart, Message};

    /// Extracts concatenated text from a User message's content parts.
    fn extract_user_text(msg: &Message) -> String {
        if let Message::User { content, .. } = msg {
            content
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        } else {
            String::new()
        }
    }

    #[test]
    fn inject_max_tokens_recovery_appends_message() {
        let mut journal = Journal::new();
        let offset = inject_max_tokens_recovery(&mut journal, 1);
        let prompt = journal.build_prompt(&[]);
        assert_eq!(prompt.messages.len(), 1);
        let text = extract_user_text(&prompt.messages[0]);
        assert!(text.contains("Output token limit hit"));
        assert_eq!(offset, 0);
    }

    #[test]
    fn inject_permission_retry_appends_message() {
        let mut journal = Journal::new();
        inject_permission_retry(&mut journal, 2, &["bash".to_string()]);
        let prompt = journal.build_prompt(&[]);
        let text = extract_user_text(&prompt.messages[0]);
        assert!(text.contains("permission-denied"));
        assert!(text.contains("bash"));
    }

    #[test]
    fn inject_completion_reminder_lists_requirements() {
        let mut journal = Journal::new();
        let reqs = vec![
            CompletionRequirement::file("src/main.rs"),
            CompletionRequirement::any_file(),
        ];
        inject_completion_reminder(&mut journal, 3, &reqs);
        let prompt = journal.build_prompt(&[]);
        let text = extract_user_text(&prompt.messages[0]);
        assert!(text.contains("src/main.rs"));
        assert!(text.contains("filesystem tool"));
    }

    #[test]
    fn snip_before_offset_drops_early_entries() {
        let mut journal = Journal::new();
        for i in 0..10 {
            journal.record(SessionEvent::MessageAppended {
                message: Message::user_text(format!("msg {i}")),
            });
        }
        assert_eq!(journal.len(), 10);
        let removed = snip_before_offset(&mut journal, 5);
        assert_eq!(removed, 5);
        assert_eq!(journal.len(), 5);
        // Remaining entries start at offset 5
        assert_eq!(journal.entries()[0].offset, 5);
    }

    #[test]
    fn snip_before_offset_zero_removes_all() {
        let mut journal = Journal::new();
        journal.record(SessionEvent::MessageAppended {
            message: Message::user_text("msg"),
        });
        let removed = snip_before_offset(&mut journal, 1);
        assert_eq!(removed, 1);
        assert!(journal.is_empty());
    }

    #[test]
    fn snip_before_offset_at_or_past_end_removes_all() {
        let mut journal = Journal::new();
        journal.record(SessionEvent::MessageAppended {
            message: Message::user_text("msg"),
        });
        // Offset 1 is past the single entry at offset 0 → removes it
        let removed = snip_before_offset(&mut journal, 1);
        assert_eq!(removed, 1);
        assert!(journal.is_empty());
    }

    #[test]
    fn snip_before_offset_zero_keeps_all() {
        let mut journal = Journal::new();
        journal.record(SessionEvent::MessageAppended {
            message: Message::user_text("msg"),
        });
        // Offset 0: every entry has offset >= 0, so nothing is removed
        let removed = snip_before_offset(&mut journal, 0);
        assert_eq!(removed, 0);
        assert_eq!(journal.len(), 1);
    }

    #[test]
    fn build_compaction_checkpoint_preserves_messages() {
        let cp = build_compaction_checkpoint(
            5,
            "Summary of early convo".to_string(),
            vec![Message::assistant_text("recent answer")],
        );
        assert_eq!(cp.compacted_until_offset, 5);
        assert_eq!(cp.summary.as_deref(), Some("Summary of early convo"));
        assert_eq!(cp.preserved_messages.len(), 1);
    }

    #[test]
    fn max_tokens_recovery_message_resumes_directly() {
        let msg = max_tokens_recovery_message(1);
        assert!(msg.contains("Resume directly"));
        assert!(msg.contains("no apology"));
    }

    #[test]
    fn permission_retry_message_lists_tools() {
        let msg = permission_retry_message(1, &["bash".to_string(), "rm".to_string()]);
        assert!(msg.contains("bash"));
        assert!(msg.contains("rm"));
    }

    #[test]
    fn completion_reminder_message_for_file_requirement() {
        let reqs = vec![CompletionRequirement::file("output.txt")];
        let msg = completion_reminder_message(1, &reqs);
        assert!(msg.contains("output.txt"));
        assert!(msg.contains("filesystem tool"));
    }

    #[test]
    fn completion_requirement_file_constructor() {
        let req = CompletionRequirement::file("test.rs");
        assert!(matches!(
            req,
            CompletionRequirement::WorkspaceFile { path: Some(_) }
        ));
        assert!(req.render().contains("test.rs"));
    }

    #[test]
    fn completion_requirement_any_file_constructor() {
        let req = CompletionRequirement::any_file();
        assert!(matches!(
            req,
            CompletionRequirement::WorkspaceFile { path: None }
        ));
        assert!(req.render().contains("a file in the workspace"));
    }

    #[test]
    fn recovery_messages_are_nonempty() {
        assert!(!max_tokens_recovery_message(1).is_empty());
        assert!(!permission_retry_message(1, &[]).is_empty());
        assert!(!completion_reminder_message(1, &[]).is_empty());
    }
}
