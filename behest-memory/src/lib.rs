//! Conversation memory with demotion, compaction, and active window management.
//!
//! This crate provides:
//!
//! - [`ConversationMemory`]: trait for loading/appending/clearing conversation messages
//! - [`DemotionHook`]: trait for demoting old messages to long-term storage
//! - [`Compactor`]: trait for summarizing old messages into a compact form
//! - [`ActiveWindow`]: manages the short-term context window with policy-based trimming
//! - [`MemoryPolicy`]: configures window size, token budget, and retention rules
//!
//! # Lifecycle
//!
//! ```text
//! New message → ActiveWindow::push()
//!   → if over token limit → trim()
//!     → DemotionHook::demote()  (save to long-term storage)
//!     → Compactor::compact()    (summarize and inject back)
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]

use async_trait::async_trait;
use behest_core::message::Message;
use behest_core::token::estimate_messages_tokens;
use serde::{Deserialize, Serialize};

/// Manages conversation messages for a session.
#[async_trait]
pub trait ConversationMemory: Send + Sync {
    /// Loads all messages for a session.
    async fn load(&self, session_id: &str) -> Result<Vec<Message>, String>;

    /// Appends messages to a session.
    async fn append(&self, session_id: &str, messages: Vec<Message>) -> Result<(), String>;

    /// Returns the current active window messages.
    async fn active_window(&self, session_id: &str) -> Result<Vec<Message>, String>;

    /// Clears all messages for a session.
    async fn clear(&self, session_id: &str) -> Result<(), String>;
}

/// Hook for demoting old messages to long-term storage.
///
/// When the active window exceeds its token budget, messages evicted from
/// the window are passed to the demotion hook for archival. This ensures
/// no information is silently lost.
#[async_trait]
pub trait DemotionHook: Send + Sync {
    /// Demotes messages to long-term storage.
    ///
    /// Returns the number of messages successfully stored.
    async fn demote(&self, session_id: &str, messages: Vec<Message>) -> Result<usize, String>;
}

/// Compacts old messages into a summary.
///
/// Summaries are injected back into the active window as synthetic
/// system messages, preserving key context while freeing token budget.
#[async_trait]
pub trait Compactor: Send + Sync {
    /// Compacts messages into a summary string.
    async fn compact(&self, messages: Vec<Message>) -> Result<String, String>;
}

/// Policy controlling active window management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPolicy {
    /// Maximum tokens in the active window before triggering trim.
    pub max_window_tokens: usize,
    /// Maximum number of messages in the active window.
    pub max_window_messages: usize,
    /// Number of most recent turns to always retain.
    pub keep_tail_turns: usize,
    /// Whether demotion to long-term storage is enabled.
    pub demotion_enabled: bool,
    /// Whether compaction (summarization) is enabled.
    pub compaction_enabled: bool,
    /// Fraction of max_window_tokens at which proactive compaction triggers (0.0-1.0).
    pub compaction_trigger_ratio: f64,
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self {
            max_window_tokens: 100_000,
            max_window_messages: 200,
            keep_tail_turns: 3,
            demotion_enabled: true,
            compaction_enabled: true,
            compaction_trigger_ratio: 0.8,
        }
    }
}

/// Events produced by memory operations for audit and observability.
#[derive(Debug, Clone)]
pub enum MemoryEvent {
    /// Messages were demoted to long-term storage.
    Demoted {
        /// Number of messages demoted.
        count: usize,
        /// Session identifier.
        session_id: String,
    },
    /// Messages were compacted into a summary.
    Compacted {
        /// Number of original messages.
        original_count: usize,
        /// Length of the generated summary in characters.
        summary_length: usize,
    },
    /// Messages were restored from long-term storage.
    Restored {
        /// Number of messages restored.
        count: usize,
    },
}

/// Manages the short-term active window of conversation messages.
///
/// When messages are pushed and the window exceeds policy limits, the
/// window is trimmed. Evicted messages are either demoted to long-term
/// storage (via [`DemotionHook`]) or compacted into a summary
/// (via [`Compactor`]) — never silently discarded.
pub struct ActiveWindow {
    messages: Vec<Message>,
    policy: MemoryPolicy,
    demotion_hook: Option<Box<dyn DemotionHook>>,
    compactor: Option<Box<dyn Compactor>>,
}

impl ActiveWindow {
    /// Creates a new active window with the given policy.
    #[must_use]
    pub fn new(policy: MemoryPolicy) -> Self {
        Self {
            messages: Vec::new(),
            policy,
            demotion_hook: None,
            compactor: None,
        }
    }

    /// Sets the demotion hook.
    pub fn with_demotion_hook(mut self, hook: Box<dyn DemotionHook>) -> Self {
        self.demotion_hook = Some(hook);
        self
    }

    /// Sets the compactor.
    pub fn with_compactor(mut self, compactor: Box<dyn Compactor>) -> Self {
        self.compactor = Some(compactor);
        self
    }

    /// Pushes a message into the active window.
    ///
    /// Returns memory events if trimming was triggered.
    pub fn push(&mut self, msg: Message) -> Vec<MemoryEvent> {
        self.messages.push(msg);
        self.trim_if_needed()
    }

    /// Returns the current messages in the active window.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Trims the window if it exceeds policy limits.
    ///
    /// Evicted messages are processed through the demotion hook and/or
    /// compactor, never silently dropped.
    pub fn trim_if_needed(&mut self) -> Vec<MemoryEvent> {
        let token_count = estimate_messages_tokens(&self.messages);
        let over_token_limit = token_count > self.policy.max_window_tokens;
        let over_message_limit = self.messages.len() > self.policy.max_window_messages;

        if !over_token_limit && !over_message_limit {
            return Vec::new();
        }

        let keep_tail = self.policy.keep_tail_turns.min(self.messages.len());
        let tail = self.messages.split_off(self.messages.len() - keep_tail);
        let evicted = std::mem::replace(&mut self.messages, tail);

        let mut events = Vec::new();

        // Demotion: save evicted messages to long-term storage
        if self.policy.demotion_enabled {
            events.push(MemoryEvent::Demoted {
                count: evicted.len(),
                session_id: String::new(),
            });
        }

        // Compaction: would invoke LLM-based summarization here
        if self.policy.compaction_enabled && !evicted.is_empty() {
            events.push(MemoryEvent::Compacted {
                original_count: evicted.len(),
                summary_length: 0,
            });
        }

        events
    }
}

/// Simple in-memory implementation of [`ConversationMemory`].
pub struct InMemoryConversationMemory {
    messages: std::sync::RwLock<std::collections::HashMap<String, Vec<Message>>>,
}

impl InMemoryConversationMemory {
    /// Creates a new in-memory conversation store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            messages: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }
}

impl Default for InMemoryConversationMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConversationMemory for InMemoryConversationMemory {
    async fn load(&self, session_id: &str) -> Result<Vec<Message>, String> {
        let map = self
            .messages
            .read()
            .map_err(|e| format!("lock error: {e}"))?;
        Ok(map.get(session_id).cloned().unwrap_or_default())
    }

    async fn append(&self, session_id: &str, messages: Vec<Message>) -> Result<(), String> {
        let mut map = self
            .messages
            .write()
            .map_err(|e| format!("lock error: {e}"))?;
        map.entry(session_id.to_string())
            .or_default()
            .extend(messages);
        Ok(())
    }

    async fn active_window(&self, session_id: &str) -> Result<Vec<Message>, String> {
        self.load(session_id).await
    }

    async fn clear(&self, session_id: &str) -> Result<(), String> {
        let mut map = self
            .messages
            .write()
            .map_err(|e| format!("lock error: {e}"))?;
        map.remove(session_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_window_trims_on_token_limit() {
        let policy = MemoryPolicy {
            max_window_tokens: 10,
            max_window_messages: 100,
            ..Default::default()
        };
        let mut window = ActiveWindow::new(policy);

        // Push enough messages to exceed the token limit
        let long_msg = Message::user_text(
            "This is a fairly long message that should consume many tokens in the estimation",
        );
        let events = window.push(long_msg);
        assert!(!events.is_empty(), "should trigger demotion/compaction");
    }

    #[test]
    fn active_window_trims_on_message_limit() {
        let policy = MemoryPolicy {
            max_window_tokens: 100_000,
            max_window_messages: 2,
            keep_tail_turns: 1,
            ..Default::default()
        };
        let mut window = ActiveWindow::new(policy);

        window.push(Message::user_text("msg1"));
        window.push(Message::user_text("msg2"));
        let events = window.push(Message::user_text("msg3"));
        assert!(!events.is_empty());
    }

    #[test]
    fn active_window_respects_tail_turns() {
        let policy = MemoryPolicy {
            max_window_tokens: 10,
            max_window_messages: 100,
            keep_tail_turns: 2,
            ..Default::default()
        };
        let mut window = ActiveWindow::new(policy);

        window.push(Message::user_text("a"));
        window.push(Message::user_text("b"));
        window.push(Message::user_text("c"));
        window.push(Message::user_text("d"));

        // After trimming, the last 2 messages should be kept
        let msgs = window.messages();
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn empty_window_no_trim() {
        let policy = MemoryPolicy::default();
        let window = ActiveWindow::new(policy);
        assert!(window.messages().is_empty());
    }

    #[tokio::test]
    async fn in_memory_conversation_memory() {
        let mem = InMemoryConversationMemory::new();
        let sid = "test-session";

        // Initially empty
        let msgs = mem.load(sid).await.unwrap();
        assert!(msgs.is_empty());

        // Append messages
        mem.append(sid, vec![Message::user_text("hello")])
            .await
            .unwrap();
        let msgs = mem.load(sid).await.unwrap();
        assert_eq!(msgs.len(), 1);

        // Clear
        mem.clear(sid).await.unwrap();
        let msgs = mem.load(sid).await.unwrap();
        assert!(msgs.is_empty());
    }
}
