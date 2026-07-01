//! Sub-agent engine — isolated child engine with bounded fan-out.
//!
//! A [`SubEngine`] wraps a fresh [`Journal`] with an optional tool allowlist
//! and explicit [`SubagentBounds`]. It shares nothing mutable with its
//! parent — the caller decides what model connection, tool runtime, or
//! context to pass into each turn. This keeps the sub-agent fully under the
//! caller's control: no implicit state sharing, no background tasks.
//!
//! # Bounds
//!
//! [`SubagentBounds`] enforces fan-out safety: maximum nesting depth, live
//! children per parent, and total descendants from the root. These prevent
//! runaway recursion or fan-out storms. A sub-engine created from another
//! sub-engine inherits attenuated bounds (depth decremented, child counters
//! shared).
//!
//! # Example
//!
//! ```ignore
//! use behest_core::sub_engine::{SubEngine, SubagentBounds, SubEngineConfig};
//! use behest_core::journal::Journal;
//!
//! let parent = SubEngine::root(Journal::new(), SubagentBounds::default());
//!
//! let mut child = parent.create_sub_engine(SubEngineConfig {
//!     allowed_tools: Some(vec!["read_file".into(), "grep".into()]),
//!     system_prompt: Some("You are a diagnostic sub-agent. Read only.".into()),
//!     max_turns: 5,
//! }).expect("within bounds");
//!
//! // child has its own journal; caller drives turns explicitly
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::journal::{Journal, SessionEvent};
use crate::message::Message;

/// A time-windowed quota tracker for sub-agent creation.
///
/// Tracks the timestamps of descendant-creation events within a sliding
/// window. Use [`check`](Self::check) to see how many creations remain
/// within the quota, and [`record`](Self::record) to consume one.
///
/// This is purely advisory — the caller decides whether to honor the quota.
/// There is no automatic enforcement; the caller calls `check` before
/// creating a child and `record` after.
///
/// # Example
///
/// ```ignore
/// use behest_core::sub_engine::QuotaWindow;
/// use std::time::Duration;
///
/// let mut quota = QuotaWindow::new(5, Duration::from_secs(60));
/// if quota.check() > 0 {
///     // create a child
///     quota.record();
/// }
/// ```
pub struct QuotaWindow {
    max_creations: u32,
    window: Duration,
    timestamps: Vec<Instant>,
}

impl std::fmt::Debug for QuotaWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Note: we report raw timestamp count without eviction (which needs
        // &mut self) — the active count may include expired entries until
        // the next check/record call.
        f.debug_struct("QuotaWindow")
            .field("max_creations", &self.max_creations)
            .field("window_secs", &self.window.as_secs())
            .field("raw_timestamp_count", &self.timestamps.len())
            .finish()
    }
}

impl QuotaWindow {
    /// Creates a new quota window allowing `max_creations` within `window`.
    #[must_use]
    pub fn new(max_creations: u32, window: Duration) -> Self {
        Self {
            max_creations,
            window,
            timestamps: Vec::new(),
        }
    }

    /// Returns the maximum number of creations allowed per window.
    #[must_use]
    pub fn max_creations(&self) -> u32 {
        self.max_creations
    }

    /// Returns the sliding window duration.
    #[must_use]
    pub fn window(&self) -> Duration {
        self.window
    }

    /// Returns the number of creations recorded in the current window
    /// (after evicting expired entries).
    #[must_use]
    pub fn active_count(&mut self) -> u32 {
        self.evict_expired();
        self.timestamps.len() as u32
    }

    /// Returns the number of remaining creations allowed in the current
    /// window. Returns 0 when the quota is exhausted.
    #[must_use]
    pub fn check(&mut self) -> u32 {
        self.evict_expired();
        self.max_creations
            .saturating_sub(self.timestamps.len() as u32)
    }

    /// Records a creation event, consuming one unit of quota.
    ///
    /// Returns `true` if the creation was within quota (and recorded),
    /// `false` if the quota was already exhausted (and the record was
    /// rejected).
    pub fn record(&mut self) -> bool {
        self.evict_expired();
        if (self.timestamps.len() as u32) >= self.max_creations {
            return false;
        }
        self.timestamps.push(Instant::now());
        true
    }

    /// Clears all recorded timestamps.
    pub fn reset(&mut self) {
        self.timestamps.clear();
    }

    /// Evicts timestamps that fall outside the sliding window.
    fn evict_expired(&mut self) {
        let now = Instant::now();
        let cutoff = now.checked_sub(self.window);
        match cutoff {
            Some(cutoff_time) => self.timestamps.retain(|&t| t >= cutoff_time),
            None => {
                // Underflow: window is longer than elapsed time since boot.
                // Keep all timestamps (none can be older than the window).
            }
        }
    }
}

impl Default for QuotaWindow {
    fn default() -> Self {
        Self::new(24, Duration::from_secs(60))
    }
}

/// Safety bounds for sub-agent fan-out.
///
/// All fields have conservative defaults. The caller can override any field
/// when constructing a root sub-engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentBounds {
    /// Maximum nesting depth. `0` means this engine cannot create children.
    pub max_child_depth: u32,
    /// Maximum number of live (not-yet-dropped) children at this level.
    pub max_live_children: u32,
    /// Maximum total descendants from the root.
    pub max_total_descendants: u32,
}

impl Default for SubagentBounds {
    fn default() -> Self {
        Self {
            max_child_depth: 4,
            max_live_children: 6,
            max_total_descendants: 24,
        }
    }
}

/// Errors produced by sub-engine creation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SubagentError {
    /// The maximum nesting depth was exceeded.
    #[error("max child depth ({max}) exceeded at depth {current}")]
    MaxDepthExceeded {
        /// Configured maximum.
        max: u32,
        /// Current depth.
        current: u32,
    },

    /// The live-children limit was reached.
    #[error("max live children ({max}) reached")]
    MaxChildrenExceeded {
        /// Configured maximum.
        max: u32,
    },

    /// The total-descendants limit was reached.
    #[error("max total descendants ({max}) reached (current: {current})")]
    MaxDescendantsExceeded {
        /// Configured maximum.
        max: u32,
        /// Current count.
        current: u32,
    },
}

/// Configuration for creating a sub-engine.
#[derive(Debug, Clone, Default)]
pub struct SubEngineConfig {
    /// If non-empty, only these tool names are visible to the sub-engine.
    pub allowed_tools: Option<Vec<String>>,
    /// Optional system prompt prepended to the sub-engine's conversation.
    pub system_prompt: Option<String>,
    /// Maximum number of turns the sub-engine may run.
    pub max_turns: usize,
}

/// An isolated sub-agent engine.
///
/// Wraps a fresh [`Journal`] with bounds and an optional tool allowlist.
/// The caller drives turns explicitly by recording events and building
/// prompts — there is no built-in run loop.
pub struct SubEngine {
    journal: Journal,
    config: SubEngineConfig,
    bounds: SubagentBounds,
    depth: u32,
    /// Shared with the parent and all siblings. Incremented when a child is
    /// created; decremented when it drops.
    live_children: Arc<AtomicU32>,
    /// Shared across the entire tree from the root. Only increments.
    total_descendants: Arc<AtomicU32>,
}

impl std::fmt::Debug for SubEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubEngine")
            .field("depth", &self.depth)
            .field("bounds", &self.bounds)
            .field("live_children", &self.live_children.load(Ordering::Relaxed))
            .field(
                "total_descendants",
                &self.total_descendants.load(Ordering::Relaxed),
            )
            .field("allowed_tools", &self.config.allowed_tools)
            .field("max_turns", &self.config.max_turns)
            .finish()
    }
}

impl SubEngine {
    /// Creates a root sub-engine with the given bounds.
    ///
    /// The root owns the live-children and total-descendants counters.
    /// Children created from it share these counters (with attenuated bounds).
    #[must_use]
    pub fn root(journal: Journal, bounds: SubagentBounds) -> Self {
        Self {
            journal,
            config: SubEngineConfig::default(),
            bounds,
            depth: 0,
            live_children: Arc::new(AtomicU32::new(0)),
            total_descendants: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Creates a root sub-engine with default bounds and an empty journal.
    #[must_use]
    pub fn new() -> Self {
        Self::root(Journal::new(), SubagentBounds::default())
    }

    /// Returns the current depth (0 for root).
    #[must_use]
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Returns the configured bounds.
    #[must_use]
    pub fn bounds(&self) -> SubagentBounds {
        self.bounds
    }

    /// Returns the tool allowlist, if set.
    #[must_use]
    pub fn allowed_tools(&self) -> Option<&[String]> {
        self.config.allowed_tools.as_deref()
    }

    /// Returns the configured max turns.
    #[must_use]
    pub fn max_turns(&self) -> usize {
        self.config.max_turns
    }

    /// Returns the optional system prompt.
    #[must_use]
    pub fn system_prompt(&self) -> Option<&str> {
        self.config.system_prompt.as_deref()
    }

    /// Returns the number of live children currently held by this engine's
    /// parent level.
    #[must_use]
    pub fn live_children_count(&self) -> u32 {
        self.live_children.load(Ordering::Relaxed)
    }

    /// Returns the total number of descendants created from the root.
    #[must_use]
    pub fn total_descendants_count(&self) -> u32 {
        self.total_descendants.load(Ordering::Relaxed)
    }

    /// Returns `true` if this engine is allowed to create children.
    #[must_use]
    pub fn can_create_child(&self) -> bool {
        let depth_ok = self.depth < self.bounds.max_child_depth;
        let children_ok =
            self.live_children.load(Ordering::Relaxed) < self.bounds.max_live_children;
        let total_ok =
            self.total_descendants.load(Ordering::Relaxed) < self.bounds.max_total_descendants;
        depth_ok && children_ok && total_ok
    }

    /// Creates a child sub-engine with attenuated bounds.
    ///
    /// The child gets a fresh empty [`Journal`]. Its `depth` is one greater
    /// than this engine's. The bounds are inherited but the depth ceiling
    /// is enforced: if this engine is already at `max_child_depth`, creation
    /// fails with [`SubagentError::MaxDepthExceeded`].
    ///
    /// The live-children counter is incremented atomically; it decrements
    /// when the child is dropped. The total-descendants counter increments
    /// atomically and never decrements.
    ///
    /// # Errors
    ///
    /// Returns [`SubagentError`] if any bound would be exceeded.
    pub fn create_sub_engine(&self, config: SubEngineConfig) -> Result<SubEngine, SubagentError> {
        // Depth check
        if self.depth >= self.bounds.max_child_depth {
            return Err(SubagentError::MaxDepthExceeded {
                max: self.bounds.max_child_depth,
                current: self.depth,
            });
        }
        // Live children check
        let current_children = self.live_children.load(Ordering::Relaxed);
        if current_children >= self.bounds.max_live_children {
            return Err(SubagentError::MaxChildrenExceeded {
                max: self.bounds.max_live_children,
            });
        }
        // Total descendants check
        let current_total = self.total_descendants.load(Ordering::Relaxed);
        if current_total >= self.bounds.max_total_descendants {
            return Err(SubagentError::MaxDescendantsExceeded {
                max: self.bounds.max_total_descendants,
                current: current_total,
            });
        }

        // Increment counters
        self.live_children.fetch_add(1, Ordering::Relaxed);
        self.total_descendants.fetch_add(1, Ordering::Relaxed);

        Ok(SubEngine {
            journal: Journal::new(),
            config,
            // Attenuated: same bounds struct, but depth+1 means the child's
            // own max_child_depth check is relative to its new depth.
            bounds: self.bounds,
            depth: self.depth + 1,
            live_children: Arc::clone(&self.live_children),
            total_descendants: Arc::clone(&self.total_descendants),
        })
    }

    /// Records an event in the sub-engine's journal.
    ///
    /// Returns the assigned offset. Use this to build the conversation
    /// explicitly — there is no automatic run loop.
    pub fn record(&mut self, event: SessionEvent) -> u64 {
        self.journal.record(event)
    }

    /// Appends a message to the sub-engine's journal.
    ///
    /// Convenience wrapper around [`record`](Self::record) for the
    /// [`MessageAppended`](SessionEvent::MessageAppended) event.
    pub fn append_message(&mut self, message: Message) -> u64 {
        self.record(SessionEvent::MessageAppended { message })
    }

    /// Builds the prompt projection from the sub-engine's journal.
    ///
    /// Delegates to [`Journal::build_prompt`]. If a `system_prompt` was
    /// configured, callers should prepend it to the returned messages.
    #[must_use]
    pub fn build_prompt(&self) -> crate::journal::PromptProjection {
        self.journal.build_prompt(&[])
    }

    /// Returns a reference to the sub-engine's journal.
    #[must_use]
    pub fn journal(&self) -> &Journal {
        &self.journal
    }
}

impl Drop for SubEngine {
    fn drop(&mut self) {
        // Only non-root engines contribute to the live-children counter
        // (the root's counter starts at 0 and is only incremented by children).
        if self.depth > 0 {
            self.live_children.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Default for SubEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::journal::SessionEvent;
    use crate::message::Message;

    #[test]
    fn root_engine_has_depth_zero() {
        let engine = SubEngine::new();
        assert_eq!(engine.depth(), 0);
        assert_eq!(engine.bounds(), SubagentBounds::default());
        assert!(engine.can_create_child());
    }

    #[test]
    fn create_child_increments_counters() {
        let root = SubEngine::new();
        assert_eq!(root.live_children_count(), 0);
        assert_eq!(root.total_descendants_count(), 0);

        let child = root.create_sub_engine(SubEngineConfig::default()).unwrap();
        assert_eq!(child.depth(), 1);
        assert_eq!(root.live_children_count(), 1);
        assert_eq!(root.total_descendants_count(), 1);
    }

    #[test]
    fn dropping_child_decrements_live_children() {
        let root = SubEngine::new();
        {
            let _child = root.create_sub_engine(SubEngineConfig::default()).unwrap();
            assert_eq!(root.live_children_count(), 1);
        }
        assert_eq!(root.live_children_count(), 0);
        // total_descendants does NOT decrement
        assert_eq!(root.total_descendants_count(), 1);
    }

    #[test]
    fn depth_limit_blocks_creation() {
        let bounds = SubagentBounds {
            max_child_depth: 1,
            ..Default::default()
        };
        let root = SubEngine::root(Journal::new(), bounds);
        let child = root.create_sub_engine(SubEngineConfig::default()).unwrap();
        // child is at depth 1, max_child_depth=1 → cannot create grandchild
        let result = child.create_sub_engine(SubEngineConfig::default());
        assert!(matches!(
            result,
            Err(SubagentError::MaxDepthExceeded { max: 1, current: 1 })
        ));
    }

    #[test]
    fn live_children_limit_blocks_creation() {
        let bounds = SubagentBounds {
            max_live_children: 2,
            ..Default::default()
        };
        let root = SubEngine::root(Journal::new(), bounds);
        let _c1 = root.create_sub_engine(SubEngineConfig::default()).unwrap();
        let _c2 = root.create_sub_engine(SubEngineConfig::default()).unwrap();
        // Third child should fail
        let result = root.create_sub_engine(SubEngineConfig::default());
        assert!(matches!(
            result,
            Err(SubagentError::MaxChildrenExceeded { max: 2 })
        ));
    }

    #[test]
    fn total_descendants_limit_blocks_creation() {
        let bounds = SubagentBounds {
            max_total_descendants: 2,
            ..Default::default()
        };
        let root = SubEngine::root(Journal::new(), bounds);
        let _c1 = root.create_sub_engine(SubEngineConfig::default()).unwrap();
        let _c2 = root.create_sub_engine(SubEngineConfig::default()).unwrap();
        // Third exceeds total
        let result = root.create_sub_engine(SubEngineConfig::default());
        assert!(matches!(
            result,
            Err(SubagentError::MaxDescendantsExceeded { max: 2, current: 2 })
        ));
    }

    #[test]
    fn child_inherits_attenuated_depth() {
        let root = SubEngine::new();
        let child = root.create_sub_engine(SubEngineConfig::default()).unwrap();
        let grandchild = child.create_sub_engine(SubEngineConfig::default()).unwrap();
        assert_eq!(grandchild.depth(), 2);
        assert_eq!(root.total_descendants_count(), 2);
    }

    #[test]
    fn allowed_tools_visible_in_config() {
        let root = SubEngine::new();
        let child = root
            .create_sub_engine(SubEngineConfig {
                allowed_tools: Some(vec!["read_file".into(), "grep".into()]),
                system_prompt: Some("You are read-only.".into()),
                max_turns: 3,
            })
            .unwrap();
        assert_eq!(child.allowed_tools().map(|t| t.len()), Some(2));
        assert_eq!(child.system_prompt(), Some("You are read-only."));
        assert_eq!(child.max_turns(), 3);
    }

    #[test]
    fn append_message_and_build_prompt() {
        let mut engine = SubEngine::new();
        engine.append_message(Message::user_text("hello"));
        engine.append_message(Message::assistant_text("hi"));
        let prompt = engine.build_prompt();
        assert_eq!(prompt.messages.len(), 2);
    }

    #[test]
    fn record_returns_offset() {
        let mut engine = SubEngine::new();
        let off0 = engine.record(SessionEvent::InputReceived {
            input: "test".into(),
        });
        let off1 = engine.record(SessionEvent::InputReceived {
            input: "test2".into(),
        });
        assert_eq!(off0, 0);
        assert_eq!(off1, 1);
    }

    #[test]
    fn can_create_child_false_at_depth_limit() {
        let bounds = SubagentBounds {
            max_child_depth: 0,
            ..Default::default()
        };
        let root = SubEngine::root(Journal::new(), bounds);
        assert!(!root.can_create_child());
    }

    #[test]
    fn debug_format_includes_counters() {
        let root = SubEngine::new();
        let _child = root.create_sub_engine(SubEngineConfig::default()).unwrap();
        let debug = format!("{root:?}");
        assert!(debug.contains("live_children"));
        assert!(debug.contains("total_descendants"));
    }

    #[test]
    fn sibling_counters_shared() {
        let root = SubEngine::new();
        let c1 = root.create_sub_engine(SubEngineConfig::default()).unwrap();
        let c2 = root.create_sub_engine(SubEngineConfig::default()).unwrap();
        // Both siblings see the same live_children count (2)
        assert_eq!(c1.live_children_count(), 2);
        assert_eq!(c2.live_children_count(), 2);
        // Drop one — the other sees the decrement
        drop(c1);
        assert_eq!(c2.live_children_count(), 1);
    }

    // ── QuotaWindow tests ──

    #[test]
    fn quota_window_starts_full() {
        let mut q = QuotaWindow::new(5, Duration::from_secs(60));
        assert_eq!(q.check(), 5);
        assert_eq!(q.active_count(), 0);
    }

    #[test]
    fn quota_window_record_decrements_check() {
        let mut q = QuotaWindow::new(3, Duration::from_secs(60));
        assert!(q.record());
        assert!(q.record());
        assert_eq!(q.check(), 1);
        assert_eq!(q.active_count(), 2);
    }

    #[test]
    fn quota_window_rejects_when_exhausted() {
        let mut q = QuotaWindow::new(1, Duration::from_secs(60));
        assert!(q.record());
        assert!(!q.record()); // exhausted
        assert_eq!(q.check(), 0);
    }

    #[test]
    fn quota_window_reset_clears_history() {
        let mut q = QuotaWindow::new(2, Duration::from_secs(60));
        assert!(q.record());
        assert!(q.record());
        assert_eq!(q.check(), 0);
        q.reset();
        assert_eq!(q.check(), 2);
    }

    #[test]
    fn quota_window_evicts_expired_entries() {
        let mut q = QuotaWindow::new(5, Duration::from_millis(10));
        // Record some entries
        assert!(q.record());
        assert!(q.record());
        assert_eq!(q.active_count(), 2);
        // Wait for them to expire
        std::thread::sleep(Duration::from_millis(20));
        // After eviction, quota is full again
        assert_eq!(q.check(), 5);
        assert_eq!(q.active_count(), 0);
    }

    #[test]
    fn quota_window_default_is_24_per_60s() {
        let q = QuotaWindow::default();
        assert_eq!(q.max_creations(), 24);
        assert_eq!(q.window(), Duration::from_secs(60));
    }

    #[test]
    fn quota_window_debug_includes_active_count() {
        let mut q = QuotaWindow::new(5, Duration::from_secs(60));
        q.record();
        let debug = format!("{q:?}");
        assert!(debug.contains("raw_timestamp_count"));
        assert!(debug.contains("max_creations"));
    }

    #[test]
    fn quota_window_check_does_not_record() {
        let mut q = QuotaWindow::new(3, Duration::from_secs(60));
        // Calling check() repeatedly should not consume quota
        assert_eq!(q.check(), 3);
        assert_eq!(q.check(), 3);
        assert_eq!(q.active_count(), 0);
    }
}
