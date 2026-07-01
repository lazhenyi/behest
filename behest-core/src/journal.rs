//! Event-sourcing journal for conversation state.
//!
//! An append-only event log that records every state change in a conversation.
//! Unlike directly mutating a `Vec<Message>`, the journal preserves the complete
//! history so that:
//!
//! - Prompts can be reconstructed at any offset (via [`Journal::build_prompt`])
//! - State can be restored after a crash (via [`Journal::from_entries`])
//! - Checkpoints allow truncating old events without losing information
//! - Incremental persistence is trivial: just export new entries
//!
//! # Example
//!
//! ```ignore
//! use behest_core::journal::{Journal, SessionEvent};
//! use behest_core::message::Message;
//!
//! let mut journal = Journal::new();
//! journal.record(SessionEvent::InputReceived { input: "hello".into() });
//! journal.record(SessionEvent::MessageAppended {
//!     message: Message::user_text("hello"),
//! });
//!
//! // Build prompt from current journal state
//! let prompt = journal.build_prompt(&[]);
//! assert_eq!(prompt.messages.len(), 1);
//! ```

use serde::{Deserialize, Serialize};

use crate::message::Message;
use crate::tool_types::ToolCall;

/// Events that change the state of a conversation.
///
/// Each variant represents one atomic state change. The journal records these
/// in order with monotonically increasing offsets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SessionEvent {
    /// User input was received.
    InputReceived {
        /// The raw user input text.
        input: String,
    },
    /// A message was appended to the conversation.
    MessageAppended {
        /// The message that was appended.
        message: Message,
    },
    /// A tool call was started (model requested a tool).
    ToolCallStarted {
        /// The tool call that was started.
        call: ToolCall,
    },
    /// A tool call finished executing.
    ToolCallFinished {
        /// The result of the tool call.
        result: ToolResult,
    },
    /// A checkpoint was created (summary of compacted events).
    CheckpointCreated {
        /// The checkpoint that was created.
        checkpoint: SessionCheckpoint,
    },
    /// A compaction boundary was recorded.
    CompactionBoundary {
        /// The boundary describing what was compacted.
        boundary: CompactionBoundary,
    },
}

/// The result of executing a tool call.
///
/// Mirrors [`ToolCall`] with the addition of output text and error information.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Matches the tool call's `id`.
    pub call_id: String,
    /// The tool name that was called.
    pub tool_name: String,
    /// The tool's output as a JSON string, or error text.
    pub output: String,
    /// Whether this result represents an error.
    pub is_error: bool,
    /// Optional structured output (JSON value).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
}

impl ToolResult {
    /// Creates a successful tool result.
    #[must_use]
    pub fn success(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        output: impl Into<String>,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            output: output.into(),
            is_error: false,
            value: None,
        }
    }

    /// Creates a tool result representing an error.
    #[must_use]
    pub fn error(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            output: error.into(),
            is_error: true,
            value: None,
        }
    }

    /// Attaches a structured JSON value to this result.
    #[must_use]
    pub fn with_value(mut self, value: serde_json::Value) -> Self {
        self.value = Some(value);
        self
    }
}

/// A single entry in the journal, carrying an event with its offset and timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogEntry {
    /// Monotonically increasing offset (assigned at record time).
    pub offset: u64,
    /// Milliseconds since the Unix epoch when this entry was recorded.
    pub timestamp_ms: u64,
    /// The event that occurred.
    pub event: SessionEvent,
}

/// An append-only event log for a conversation.
///
/// All mutation goes through [`Journal::record`].  Reading is done via
/// [`Journal::build_prompt`] (pure projection) or [`Journal::entries_since`]
/// (for incremental persistence).
pub struct Journal {
    entries: Vec<LogEntry>,
    next_offset: u64,
}

impl std::fmt::Debug for Journal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Journal")
            .field("entries", &self.entries.len())
            .field("next_offset", &self.next_offset)
            .finish()
    }
}

impl Default for Journal {
    fn default() -> Self {
        Self::new()
    }
}

impl Journal {
    /// Creates an empty journal.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_offset: 0,
        }
    }

    /// Restores a journal from previously exported entries.
    ///
    /// The caller is responsible for preserving entry order. The `next_offset`
    /// is set to one past the highest offset in the input (or 0 if empty).
    #[must_use]
    pub fn from_entries(entries: Vec<LogEntry>) -> Self {
        let next_offset = entries
            .last()
            .map(|e| e.offset.saturating_add(1))
            .unwrap_or(0);
        Self {
            entries,
            next_offset,
        }
    }

    /// Records an event, returning the assigned offset.
    pub fn record(&mut self, event: SessionEvent) -> u64 {
        let offset = self.next_offset;
        self.next_offset += 1;
        let timestamp_ms = current_timestamp_ms();
        self.entries.push(LogEntry {
            offset,
            timestamp_ms,
            event,
        });
        offset
    }

    /// Returns the next offset that will be assigned.
    #[must_use]
    pub fn next_offset(&self) -> u64 {
        self.next_offset
    }

    /// Returns the number of entries in the journal.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the journal contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns all entries recorded at or after the given offset.
    ///
    /// Use this for incremental persistence: export only new entries since
    /// the last time you persisted.
    #[must_use]
    pub fn entries_since(&self, offset: u64) -> &[LogEntry] {
        match self.entries.binary_search_by_key(&offset, |e| e.offset) {
            Ok(idx) | Err(idx) => &self.entries[idx..],
        }
    }

    /// Returns all entries in the journal.
    #[must_use]
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Builds the current prompt projection from the journal and checkpoints.
    ///
    /// This is a pure projection — it does not modify the journal. Given the
    /// full journal and zero or more checkpoints, it reconstructs the
    /// conversation state as a [`PromptProjection`] suitable for sending to
    /// an LLM provider.
    ///
    /// Checkpoints are processed in order. Each checkpoint's
    /// `compacted_until_offset` tells the builder to skip journal entries up
    /// to that offset and instead use the checkpoint's preserved messages and
    /// optional summary.
    #[must_use]
    pub fn build_prompt(&self, checkpoints: &[SessionCheckpoint]) -> PromptProjection {
        let mut messages = Vec::new();
        let mut open_tool_calls: Vec<ToolCall> = Vec::new();
        let mut summary: Option<String> = None;
        let mut compacted_until: u64 = 0;

        // Apply the latest checkpoint
        if let Some(last_cp) = checkpoints.last() {
            compacted_until = last_cp.compacted_until_offset;
            summary = last_cp.summary.clone();
            messages = last_cp.preserved_messages.clone();
            open_tool_calls = reopen_tool_calls(&messages);
        }

        // Collect entries to replay (all entries after compacted_until)
        // Use a plain for loop instead of filter() to avoid borrow issues
        for entry in &self.entries {
            if entry.offset < compacted_until {
                continue;
            }
            match &entry.event {
                SessionEvent::MessageAppended { message } => {
                    let calls = message.tool_calls();
                    for call in calls {
                        if !open_tool_calls.iter().any(|c| c.id == call.id) {
                            open_tool_calls.push(call.clone());
                        }
                    }
                    messages.push(message.clone());
                }
                SessionEvent::ToolCallStarted { call } => {
                    if !open_tool_calls.iter().any(|c| c.id == call.id) {
                        open_tool_calls.push(call.clone());
                    }
                }
                SessionEvent::ToolCallFinished { result } => {
                    open_tool_calls.retain(|c| c.id != result.call_id);
                }
                SessionEvent::CheckpointCreated { checkpoint } => {
                    // Later checkpoint overrides earlier state
                    compacted_until = checkpoint.compacted_until_offset;
                    summary = checkpoint.summary.clone();
                    messages = checkpoint.preserved_messages.clone();
                    open_tool_calls = reopen_tool_calls(&messages);
                }
                SessionEvent::InputReceived { .. } | SessionEvent::CompactionBoundary { .. } => {
                    // These don't directly affect the prompt
                }
            }
        }

        PromptProjection {
            summary,
            messages,
            open_tool_calls,
        }
    }
}

/// A checkpoint created after compacting old journal entries.
///
/// Instead of keeping every event forever, the user can periodically create
/// checkpoints that summarize old events and preserve recent messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionCheckpoint {
    /// All journal entries up to this offset have been compacted.
    pub compacted_until_offset: u64,
    /// Optional LLM-generated summary of the compacted portion.
    pub summary: Option<String>,
    /// Messages preserved from the compacted portion (typically the tail).
    pub preserved_messages: Vec<Message>,
}

/// Describes what happened during a compaction operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactionBoundary {
    /// A microcompact cleared old tool outputs.
    Microcompact {
        /// Token savings from the operation.
        tokens_freed: usize,
    },
    /// An autocompact dropped the oldest message groups.
    Autocompact {
        /// How many message groups were dropped.
        groups_dropped: usize,
        /// Token savings from the operation.
        tokens_freed: usize,
    },
    /// An LLM-driven full compaction produced a summary.
    FullCompact {
        /// Length of the generated summary in characters.
        summary_chars: usize,
        /// Token savings from the operation.
        tokens_freed: usize,
    },
}

/// The reconstructed state ready for an LLM prompt.
#[derive(Debug, Clone)]
pub struct PromptProjection {
    /// Optional summary of compacted events (injected as a system message).
    pub summary: Option<String>,
    /// Messages to include in the prompt.
    pub messages: Vec<Message>,
    /// Tool calls that have been started but not yet finished.
    pub open_tool_calls: Vec<ToolCall>,
}

impl PromptProjection {
    /// Builds a vector of prompt messages, optionally prepending the summary
    /// as a system message.
    #[must_use]
    pub fn to_prompt_messages(&self) -> Vec<Message> {
        let mut out = Vec::new();
        if let Some(ref summary) = self.summary {
            out.push(Message::system_text(summary));
        }
        out.extend(self.messages.clone());
        out
    }
}

/// Circuit breaker for compaction operations.
///
/// When consecutive LLM-driven compaction attempts fail, the circuit breaker
/// opens to prevent wasting tokens on repeated failed attempts. The user is
/// responsible for checking [`is_open`](CompactCircuitBreaker::is_open) before
/// attempting compaction and calling [`record_success`](CompactCircuitBreaker::record_success)
/// or [`record_failure`](CompactCircuitBreaker::record_failure) afterward.
pub struct CompactCircuitBreaker {
    max_consecutive_failures: usize,
    consecutive_failures: usize,
}

impl std::fmt::Debug for CompactCircuitBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactCircuitBreaker")
            .field("is_open", &self.is_open())
            .field("consecutive_failures", &self.consecutive_failures)
            .field("max_consecutive_failures", &self.max_consecutive_failures)
            .finish()
    }
}

impl CompactCircuitBreaker {
    /// Creates a new circuit breaker that opens after the given number of
    /// consecutive failures.
    #[must_use]
    pub fn new(max_consecutive_failures: usize) -> Self {
        Self {
            max_consecutive_failures,
            consecutive_failures: 0,
        }
    }

    /// Returns `true` when the circuit is open and compaction should be
    /// skipped to avoid wasting tokens.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.consecutive_failures >= self.max_consecutive_failures
    }

    /// Returns the number of consecutive failures so far.
    #[must_use]
    pub fn consecutive_failures(&self) -> usize {
        self.consecutive_failures
    }

    /// Records a successful compaction, resetting the failure counter.
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
    }

    /// Records a failed compaction attempt.
    pub fn record_failure(&mut self) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
    }

    /// Resets the circuit breaker to the closed state.
    pub fn reset(&mut self) {
        self.consecutive_failures = 0;
    }
}

// ── Helpers ──

fn reopen_tool_calls(messages: &[Message]) -> Vec<ToolCall> {
    let mut open: Vec<ToolCall> = Vec::new();

    for msg in messages {
        // Assistant messages may start tool calls
        let calls = msg.tool_calls();
        for call in calls {
            if !open.iter().any(|c| c.id == call.id) {
                open.push(call.clone());
            }
        }
        // Tool messages close their corresponding tool call
        if let Message::Tool { tool_call_id, .. } = msg {
            open.retain(|c| c.id != *tool_call_id);
        }
    }

    open
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::message::Message;
    use crate::tool_types::ToolCall;

    #[test]
    fn empty_journal_produces_empty_prompt() {
        let journal = Journal::new();
        let prompt = journal.build_prompt(&[]);
        assert!(prompt.messages.is_empty());
        assert!(prompt.summary.is_none());
        assert!(prompt.open_tool_calls.is_empty());
    }

    #[test]
    fn record_and_build_prompt() {
        let mut journal = Journal::new();

        journal.record(SessionEvent::MessageAppended {
            message: Message::user_text("hello"),
        });
        journal.record(SessionEvent::MessageAppended {
            message: Message::assistant_text("hi there"),
        });

        let prompt = journal.build_prompt(&[]);
        // user + assistant = 2 messages
        assert_eq!(prompt.messages.len(), 2);
    }

    #[test]
    fn checkpoint_skips_compacted_entries() {
        let mut journal = Journal::new();

        // Record some early messages
        journal.record(SessionEvent::MessageAppended {
            message: Message::user_text("old question"),
        });
        journal.record(SessionEvent::MessageAppended {
            message: Message::assistant_text("old answer"),
        });

        // Create a checkpoint that compacts everything so far
        let cp = SessionCheckpoint {
            compacted_until_offset: 2,
            summary: Some("Earlier: user asked a question, assistant answered.".into()),
            preserved_messages: vec![Message::assistant_text("old answer")],
        };

        // Record new messages after the checkpoint
        journal.record(SessionEvent::MessageAppended {
            message: Message::user_text("follow up"),
        });

        let prompt = journal.build_prompt(&[cp]);
        // preserved message (1) + new follow up (1) = 2 messages
        assert!(prompt.summary.is_some());
        assert_eq!(prompt.messages.len(), 2);
    }

    #[test]
    fn tool_call_tracking() {
        let mut journal = Journal::new();

        journal.record(SessionEvent::MessageAppended {
            message: Message::user_text("search for cats"),
        });

        let call = ToolCall::new("call_1", "search", serde_json::json!({"query": "cats"}));
        journal.record(SessionEvent::ToolCallStarted { call: call.clone() });

        let prompt = journal.build_prompt(&[]);
        assert_eq!(prompt.open_tool_calls.len(), 1);
        assert_eq!(prompt.open_tool_calls[0].id, "call_1");

        journal.record(SessionEvent::ToolCallFinished {
            result: ToolResult::success("call_1", "search", "found 3 cats"),
        });

        let prompt = journal.build_prompt(&[]);
        assert!(prompt.open_tool_calls.is_empty());
    }

    #[test]
    fn entries_since_returns_only_new_entries() {
        let mut journal = Journal::new();

        journal.record(SessionEvent::InputReceived {
            input: "first".into(),
        });
        journal.record(SessionEvent::InputReceived {
            input: "second".into(),
        });
        let since = journal.next_offset();
        journal.record(SessionEvent::InputReceived {
            input: "third".into(),
        });

        let new_entries = journal.entries_since(since);
        assert_eq!(new_entries.len(), 1);
    }

    #[test]
    fn from_entries_restores_offset() {
        let mut journal = Journal::new();
        journal.record(SessionEvent::InputReceived {
            input: "hello".into(),
        });
        journal.record(SessionEvent::InputReceived {
            input: "world".into(),
        });

        let entries = journal.entries().to_vec();
        let restored = Journal::from_entries(entries);

        // next_offset should be one past the last entry
        assert_eq!(restored.next_offset(), 2);
        assert_eq!(restored.len(), 2);
    }

    #[test]
    fn circuit_breaker_opens_after_max_failures() {
        let mut cb = CompactCircuitBreaker::new(3);
        assert!(!cb.is_open());

        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open());

        cb.record_failure(); // 3rd failure
        assert!(cb.is_open());

        cb.record_success();
        assert!(!cb.is_open());
    }

    #[test]
    fn circuit_breaker_reset() {
        let mut cb = CompactCircuitBreaker::new(2);
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_open());

        cb.reset();
        assert!(!cb.is_open());
        assert_eq!(cb.consecutive_failures(), 0);
    }

    #[test]
    fn tool_result_builder() {
        let result = ToolResult::success("call_1", "echo", "hello world")
            .with_value(serde_json::json!({"echo": "hello world"}));
        assert!(!result.is_error);
        assert_eq!(result.call_id, "call_1");
        assert_eq!(result.tool_name, "echo");
        assert!(result.value.is_some());
    }

    #[test]
    fn checkpoint_created_event_updates_prompt() {
        let mut journal = Journal::new();

        journal.record(SessionEvent::MessageAppended {
            message: Message::user_text("original question"),
        });
        journal.record(SessionEvent::MessageAppended {
            message: Message::assistant_text("original answer"),
        });

        let cp = SessionCheckpoint {
            compacted_until_offset: 2,
            summary: Some("Compacted summary".into()),
            preserved_messages: vec![Message::assistant_text("original answer")],
        };
        journal.record(SessionEvent::CheckpointCreated { checkpoint: cp });

        journal.record(SessionEvent::MessageAppended {
            message: Message::user_text("new question"),
        });

        let prompt = journal.build_prompt(&[]);
        assert!(prompt.summary.is_some());
        assert_eq!(prompt.messages.len(), 2);
    }
}
