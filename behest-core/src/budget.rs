//! Token budget and context pressure level calculation.
//!
//! Pure functions — no I/O, no state mutation. Users query the budget state
//! and decide what action to take.
//!
//! # Example
//!
//! ```ignore
//! use behest_core::budget::{TokenBudget, PressureLevel};
//! use behest_core::token::estimate_messages_tokens;
//!
//! let budget = TokenBudget::new(100_000, 20_000);
//! let tokens = estimate_messages_tokens(&messages);
//! let state = budget.state_for(tokens);
//!
//! match state.pressure {
//!     PressureLevel::Normal => { /* continue */ }
//!     PressureLevel::Warning => { tracing::warn!("approaching limit"); }
//!     PressureLevel::Compact => { /* user decides: compact or stop */ }
//!     PressureLevel::Blocking => { /* user must compact or return error */ }
//! }
//! ```

/// Default overall context window size (GPT-4o / Claude Sonnet level).
pub const DEFAULT_CONTEXT_WINDOW: usize = 100_000;

/// Default tokens reserved for model output.
pub const DEFAULT_RESERVED_OUTPUT: usize = 20_000;

/// Default buffer between Warning and Compact thresholds.
pub const DEFAULT_WARNING_BUFFER: usize = 20_000;

/// Default buffer between Compact and Blocking thresholds.
pub const DEFAULT_COMPACT_BUFFER: usize = 13_000;

/// Default buffer between Blocking threshold and hard context limit.
pub const DEFAULT_BLOCKING_BUFFER: usize = 3_000;

/// Hard floor for the effective context window (prevents degenerate configs).
const MIN_EFFECTIVE_WINDOW: usize = 1_000;

/// Minimum prompt budget when computing reserved output for small windows.
const MIN_PROMPT_BUDGET: usize = 8_000;

/// Context pressure level, indicating how full the context window is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PressureLevel {
    /// Plenty of room remaining — no action needed.
    Normal,
    /// Approaching the warning threshold — user may want to prepare.
    Warning,
    /// Above the compact threshold — user should compact.
    Compact,
    /// At or above the blocking threshold — user must compact before proceeding.
    Blocking,
}

/// Configuration for token budget calculation.
///
/// All values are in estimated tokens. Use [`TokenBudget::state_for`] to
/// compute the current [`BudgetState`] for a given token count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenBudget {
    /// Total context window size (tokens received by the model).
    pub context_window: usize,
    /// Tokens reserved for model output.
    pub reserved_output: usize,
    /// Buffer between Warning and Compact thresholds.
    pub warning_buffer: usize,
    /// Buffer between Compact and Blocking thresholds.
    pub compact_buffer: usize,
    /// Buffer between Blocking threshold and hard context limit.
    pub blocking_buffer: usize,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            context_window: DEFAULT_CONTEXT_WINDOW,
            reserved_output: DEFAULT_RESERVED_OUTPUT,
            warning_buffer: DEFAULT_WARNING_BUFFER,
            compact_buffer: DEFAULT_COMPACT_BUFFER,
            blocking_buffer: DEFAULT_BLOCKING_BUFFER,
        }
    }
}

impl TokenBudget {
    /// Creates a budget with the given context window and reserved output,
    /// using default buffer values.
    #[must_use]
    pub fn new(context_window: usize, reserved_output: usize) -> Self {
        Self {
            context_window: context_window.max(MIN_EFFECTIVE_WINDOW),
            reserved_output,
            ..Self::default()
        }
    }

    /// Creates a budget with full control over all parameters.
    #[must_use]
    pub const fn with_buffers(
        context_window: usize,
        reserved_output: usize,
        warning_buffer: usize,
        compact_buffer: usize,
        blocking_buffer: usize,
    ) -> Self {
        Self {
            context_window,
            reserved_output,
            warning_buffer,
            compact_buffer,
            blocking_buffer,
        }
    }

    /// The effective context window = total window minus reserved output tokens.
    #[must_use]
    pub fn effective_window(&self) -> usize {
        effective_context_window(self.context_window, self.reserved_output)
    }

    /// Computes the three thresholds (warning, compact, blocking) from the budget.
    #[must_use]
    pub fn thresholds(&self) -> BudgetThresholds {
        calculate_thresholds(self)
    }

    /// Computes the full budget state for a given estimated token count.
    #[must_use]
    pub fn state_for(&self, estimated_tokens: usize) -> BudgetState {
        calculate_budget_state(estimated_tokens, self)
    }

    /// Returns the remaining token capacity before the blocking threshold.
    #[must_use]
    pub fn remaining_before_blocking(&self, estimated_tokens: usize) -> usize {
        let thresholds = self.thresholds();
        thresholds
            .blocking_threshold
            .saturating_sub(estimated_tokens)
    }

    /// Returns the remaining token capacity in the effective window.
    #[must_use]
    pub fn remaining(&self, estimated_tokens: usize) -> usize {
        self.effective_window().saturating_sub(estimated_tokens)
    }
}

/// Computed thresholds derived from a [`TokenBudget`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetThresholds {
    /// Total tokens available for the prompt (context_window - reserved_output).
    pub effective_window: usize,
    /// Token count at which pressure rises to Warning.
    pub warning_threshold: usize,
    /// Token count at which pressure rises to Compact.
    pub compact_threshold: usize,
    /// Token count at which pressure rises to Blocking.
    pub blocking_threshold: usize,
}

/// Full budget state for a given token count, including pressure level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetState {
    /// Current estimated token count.
    pub estimated_tokens: usize,
    /// Effective context window size.
    pub effective_window: usize,
    /// Threshold for Warning level.
    pub warning_threshold: usize,
    /// Threshold for Compact level.
    pub compact_threshold: usize,
    /// Threshold for Blocking level.
    pub blocking_threshold: usize,
    /// Percentage of effective window used (0–100).
    pub percent_used: u8,
    /// Percentage of effective window remaining (0–100).
    pub percent_remaining: u8,
    /// Current pressure level.
    pub pressure: PressureLevel,
    /// True when estimated_tokens >= warning_threshold.
    pub is_above_warning: bool,
    /// True when estimated_tokens >= compact_threshold.
    pub is_above_compact: bool,
    /// True when estimated_tokens >= blocking_threshold.
    pub is_above_blocking: bool,
}

// ── Pure computation helpers ──

/// Computes the effective context window, clamping to [`MIN_EFFECTIVE_WINDOW`].
#[must_use]
pub fn effective_context_window(context_window: usize, reserved_output: usize) -> usize {
    let reserved = effective_reserved_output(context_window, reserved_output);
    context_window
        .saturating_sub(reserved)
        .max(MIN_EFFECTIVE_WINDOW)
}

/// Computes the effective reserved output, ensuring at least half the
/// context window (or [`MIN_PROMPT_BUDGET`]) is available for the prompt.
#[must_use]
pub fn effective_reserved_output(context_window: usize, reserved_output: usize) -> usize {
    let context_window = context_window.max(1);
    let min_prompt = (context_window / 2).clamp(1, MIN_PROMPT_BUDGET);
    let max_reserve = context_window.saturating_sub(min_prompt);
    reserved_output.min(max_reserve)
}

/// Calculates thresholds from a budget. Buffers are capped to 1/3 of the
/// effective window to prevent pathological thresholds.
#[must_use]
pub fn calculate_thresholds(budget: &TokenBudget) -> BudgetThresholds {
    let effective = budget.effective_window();
    let auto_buffer = budget.compact_buffer.min(effective / 3);
    let warn_buffer = budget.warning_buffer.min(effective / 3);
    let block_buffer = budget.blocking_buffer.min(effective / 10).max(1);

    let compact_threshold = effective.saturating_sub(auto_buffer).max(1);
    let warning_threshold = compact_threshold
        .saturating_sub(warn_buffer)
        .max(1)
        .min(compact_threshold);
    let blocking_threshold = effective
        .saturating_sub(block_buffer)
        .max(compact_threshold)
        .min(effective);

    BudgetThresholds {
        effective_window: effective,
        warning_threshold,
        compact_threshold,
        blocking_threshold,
    }
}

/// Computes the full budget state.
#[must_use]
pub fn calculate_budget_state(estimated_tokens: usize, budget: &TokenBudget) -> BudgetState {
    let thresholds = budget.thresholds();
    let percent_used = estimated_tokens
        .saturating_mul(100)
        .checked_div(thresholds.effective_window)
        .unwrap_or(100)
        .min(100) as u8;
    let percent_remaining = 100u8.saturating_sub(percent_used);

    let is_above_blocking = estimated_tokens >= thresholds.blocking_threshold;
    let is_above_compact = estimated_tokens >= thresholds.compact_threshold;
    let is_above_warning = estimated_tokens >= thresholds.warning_threshold;

    let pressure = if is_above_blocking {
        PressureLevel::Blocking
    } else if is_above_compact {
        PressureLevel::Compact
    } else if is_above_warning {
        PressureLevel::Warning
    } else {
        PressureLevel::Normal
    };

    BudgetState {
        estimated_tokens,
        effective_window: thresholds.effective_window,
        warning_threshold: thresholds.warning_threshold,
        compact_threshold: thresholds.compact_threshold,
        blocking_threshold: thresholds.blocking_threshold,
        percent_used,
        percent_remaining,
        pressure,
        is_above_warning,
        is_above_compact,
        is_above_blocking,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_thresholds_are_ordered() {
        let budget = TokenBudget::default();
        let t = budget.thresholds();
        assert!(t.warning_threshold <= t.compact_threshold);
        assert!(t.compact_threshold <= t.blocking_threshold);
        assert!(t.blocking_threshold <= t.effective_window);
    }

    #[test]
    fn state_at_warning_threshold_is_warning() {
        let budget = TokenBudget::new(10_000, 1_000);
        let t = budget.thresholds();
        let state = budget.state_for(t.warning_threshold);
        assert_eq!(state.pressure, PressureLevel::Warning);
        assert!(state.is_above_warning);
        assert!(!state.is_above_compact);
    }

    #[test]
    fn state_at_compact_threshold_is_compact() {
        let budget = TokenBudget::new(10_000, 1_000);
        let t = budget.thresholds();
        let state = budget.state_for(t.compact_threshold);
        assert_eq!(state.pressure, PressureLevel::Compact);
        assert!(state.is_above_warning);
        assert!(state.is_above_compact);
        assert!(!state.is_above_blocking);
    }

    #[test]
    fn state_at_blocking_threshold_is_blocking() {
        let budget = TokenBudget::new(10_000, 1_000);
        let t = budget.thresholds();
        let state = budget.state_for(t.blocking_threshold);
        assert_eq!(state.pressure, PressureLevel::Blocking);
        assert!(state.is_above_blocking);
    }

    #[test]
    fn small_window_clamps_effective_to_min() {
        let budget = TokenBudget::new(2_000, 20_000);
        // With 2000 window and 20000 reserve, effective should clamp to 1000
        assert_eq!(budget.effective_window(), MIN_EFFECTIVE_WINDOW);
    }

    #[test]
    fn percent_saturates_at_100() {
        let budget = TokenBudget::new(10_000, 1_000);
        let state = budget.state_for(1_000_000);
        assert_eq!(state.percent_used, 100);
        assert_eq!(state.percent_remaining, 0);
    }

    #[test]
    fn remaining_before_blocking_returns_zero_when_at_threshold() {
        let budget = TokenBudget::new(10_000, 1_000);
        let t = budget.thresholds();
        assert_eq!(budget.remaining_before_blocking(t.blocking_threshold), 0);
    }

    #[test]
    fn remaining_before_blocking_positive_when_below() {
        let budget = TokenBudget::new(100_000, 20_000);
        // Way below threshold, should have plenty remaining
        assert!(budget.remaining_before_blocking(1_000) > 0);
    }

    #[test]
    fn pressure_level_ordering_is_correct() {
        assert!(PressureLevel::Normal < PressureLevel::Warning);
        assert!(PressureLevel::Warning < PressureLevel::Compact);
        assert!(PressureLevel::Compact < PressureLevel::Blocking);
    }

    #[test]
    fn custom_buffers_respect_order() {
        let budget = TokenBudget::with_buffers(100_000, 20_000, 5_000, 5_000, 1_000);
        let t = budget.thresholds();
        assert!(t.warning_threshold <= t.compact_threshold);
        assert!(t.compact_threshold <= t.blocking_threshold);
    }

    #[test]
    fn buffers_are_capped_to_one_third_of_effective() {
        // If buffer exceeds 1/3 of effective window, it should be capped
        let budget = TokenBudget::with_buffers(10_000, 1_000, 9_000, 9_000, 9_000);
        let t = budget.thresholds();
        // effective = 10_000 - 1_000 = 9_000, buffer cap = 3_000
        // compact_threshold = 9_000 - min(9_000, 3_000) = 6_000
        assert!(t.compact_threshold > 0);
        assert!(t.blocking_threshold >= t.compact_threshold);
    }
}
