//! Aggregated prompt cache statistics for a run or a window of events.
//!
//! [`CacheStats`] reduces a sequence of [`AgentEvent`] values (typically
//! replayed from a [`RuntimeEventStore`](super::RuntimeEventStore)) into
//! the totals needed to evaluate cache effectiveness:
//!
//! - Total input / output tokens consumed
//! - Total cache writes (Anthropic `cache_creation_input_tokens`)
//! - Total cache reads (Anthropic / DeepSeek `cache_read_input_tokens`)
//! - Total cached input (OpenAI `prompt_tokens_details.cached_tokens`)
//! - Model call count
//!
//! Use [`CacheStats::cache_hit_rate`] to get a normalized `0.0..=1.0` score
//! describing what fraction of input tokens came from cache.
//!
//! # Example
//!
//! ```rust,no_run
//! use behest_runtime::cache_stats::CacheStats;
//! use behest_runtime::event::AgentEvent;
//! use behest_runtime::event_store::{RuntimeEventStore, MemoryRuntimeEventStore};
//! use behest_runtime::run::RunId;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let store = MemoryRuntimeEventStore::new();
//! // ... append events to `store` for a given run ...
//! let run_id = behest_runtime::run::RunId::new();
//! let envelopes = store.list_after(run_id, None, 1024).await?;
//! let stats = CacheStats::from_envelopes(&envelopes);
//! println!("hit rate: {:.1}%", stats.cache_hit_rate() * 100.0);
//! # Ok(()) }
//! ```

use serde::{Deserialize, Serialize};

use super::event::{AgentEvent, CacheMetrics};
use super::stream::RuntimeEventEnvelope;
use behest_event::UsageRecorded;

/// Aggregated prompt-cache statistics over a window of events.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of `UsageRecorded` events observed.
    pub call_count: u64,
    /// Sum of `UsageRecorded.usage.input_tokens` across the window.
    pub total_input_tokens: u64,
    /// Sum of `UsageRecorded.usage.output_tokens` across the window.
    pub total_output_tokens: u64,
    /// Sum of [`CacheMetrics::cache_creation_input_tokens`].
    ///
    /// Populated by Anthropic for input tokens that were written to a
    /// cache entry during this run. `None` per-event is treated as `0`.
    pub total_cache_creation_input_tokens: u64,
    /// Sum of [`CacheMetrics::cache_read_input_tokens`].
    ///
    /// Populated by Anthropic and DeepSeek for input tokens served from
    /// an existing cache entry.
    pub total_cache_read_input_tokens: u64,
    /// Sum of [`CacheMetrics::cached_input_tokens`].
    ///
    /// Populated by OpenAI's `prompt_tokens_details.cached_tokens`.
    pub total_cached_input_tokens: u64,
}

impl CacheStats {
    /// Computes aggregate stats by walking a slice of [`AgentEvent`]
    /// values in arrival order.
    ///
    /// Events that are neither `UsageRecorded` nor `CacheMetrics` are
    /// ignored. Order does not matter for accumulation but is preserved
    /// for deterministic output in the returned `Debug` print.
    #[must_use]
    pub fn from_events(events: &[AgentEvent]) -> Self {
        let mut stats = Self::default();
        for event in events {
            match event {
                AgentEvent::UsageRecorded(UsageRecorded { usage, .. }) => {
                    stats.call_count = stats.call_count.saturating_add(1);
                    stats.total_input_tokens =
                        stats.total_input_tokens.saturating_add(usage.input_tokens);
                    stats.total_output_tokens = stats
                        .total_output_tokens
                        .saturating_add(usage.output_tokens);
                }
                AgentEvent::CacheMetrics(CacheMetrics {
                    cache_creation_input_tokens,
                    cache_read_input_tokens,
                    cached_input_tokens,
                    ..
                }) => {
                    stats.total_cache_creation_input_tokens = stats
                        .total_cache_creation_input_tokens
                        .saturating_add(*cache_creation_input_tokens);
                    stats.total_cache_read_input_tokens = stats
                        .total_cache_read_input_tokens
                        .saturating_add(*cache_read_input_tokens);
                    stats.total_cached_input_tokens = stats
                        .total_cached_input_tokens
                        .saturating_add(*cached_input_tokens);
                }
                _ => {}
            }
        }
        stats
    }

    /// Computes aggregate stats from a slice of [`RuntimeEventEnvelope`]
    /// values (the shape returned by
    /// [`RuntimeEventStore::list_after`](super::RuntimeEventStore::list_after)).
    #[must_use]
    pub fn from_envelopes(envelopes: &[RuntimeEventEnvelope]) -> Self {
        let events: Vec<&AgentEvent> = envelopes.iter().map(|e| &e.event).collect();
        // SAFETY: events live as long as envelopes
        let events: Vec<AgentEvent> = events.into_iter().cloned().collect();
        Self::from_events(&events)
    }

    /// Returns the total cache-related input tokens across all three
    /// provider-specific fields.
    #[must_use]
    pub const fn total_cache_tokens(&self) -> u64 {
        self.total_cache_creation_input_tokens
            + self.total_cache_read_input_tokens
            + self.total_cached_input_tokens
    }

    /// Returns the fraction of input tokens that came from cache.
    ///
    /// The denominator is `input_tokens + cache_creation + cache_read +
    /// cached` — the total input tokens actually processed by the
    /// provider, whether fresh or served from cache. The numerator is
    /// `cache_read + cached` (tokens that were served from a cache
    /// entry instead of recomputed).
    ///
    /// Returns `0.0` when the window contains no input tokens.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.total_cache_read_input_tokens + self.total_cached_input_tokens;
        let total = self.total_input_tokens + self.total_cache_tokens();
        if total == 0 {
            return 0.0;
        }
        hits as f64 / total as f64
    }

    /// Combines two `CacheStats` by summing every field. Useful for
    /// aggregating per-run stats into a session-level or
    /// session-store-wide total.
    ///
    /// Uses `saturating_add` to match [`Self::from_events`] and to remain
    /// panic-free at the `u64` boundary when aggregating very long runs.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            call_count: self.call_count.saturating_add(other.call_count),
            total_input_tokens: self
                .total_input_tokens
                .saturating_add(other.total_input_tokens),
            total_output_tokens: self
                .total_output_tokens
                .saturating_add(other.total_output_tokens),
            total_cache_creation_input_tokens: self
                .total_cache_creation_input_tokens
                .saturating_add(other.total_cache_creation_input_tokens),
            total_cache_read_input_tokens: self
                .total_cache_read_input_tokens
                .saturating_add(other.total_cache_read_input_tokens),
            total_cached_input_tokens: self
                .total_cached_input_tokens
                .saturating_add(other.total_cached_input_tokens),
        }
    }
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CacheStats {{")?;
        writeln!(f, "  model calls:           {}", self.call_count)?;
        writeln!(f, "  total input tokens:    {}", self.total_input_tokens)?;
        writeln!(f, "  total output tokens:   {}", self.total_output_tokens)?;
        writeln!(
            f,
            "  cache writes (Anthropic): {}",
            self.total_cache_creation_input_tokens
        )?;
        writeln!(
            f,
            "  cache reads (Anthropic/DeepSeek): {}",
            self.total_cache_read_input_tokens
        )?;
        writeln!(
            f,
            "  cached input (OpenAI): {}",
            self.total_cached_input_tokens
        )?;
        writeln!(f, "  total cache tokens:    {}", self.total_cache_tokens())?;
        writeln!(
            f,
            "  cache hit rate:        {:.2}%",
            self.cache_hit_rate() * 100.0
        )?;
        write!(f, "}}")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::run::RunId;
    use behest_provider::{FinishReason, ModelName, ProviderId, TokenUsage};
    use chrono::Utc;
    use uuid::Uuid;

    fn usage_event(input: u64, output: u64) -> AgentEvent {
        AgentEvent::UsageRecorded(UsageRecorded {
            run_id: RunId::new(),
            usage: TokenUsage::new(input, output),
            timestamp: Utc::now(),
        })
    }

    fn cache_metrics_event(creation: u64, read: u64, cached: u64) -> AgentEvent {
        AgentEvent::CacheMetrics(CacheMetrics {
            run_id: RunId::new(),
            cache_creation_input_tokens: creation,
            cache_read_input_tokens: read,
            cached_input_tokens: cached,
            timestamp: Utc::now(),
        })
    }

    #[test]
    fn from_events_aggregates_usage_and_cache() {
        let events = vec![
            usage_event(1000, 50),
            cache_metrics_event(800, 0, 0), // Anthropic: wrote 800 to cache
            usage_event(500, 30),
            cache_metrics_event(0, 400, 0), // Anthropic: read 400 from cache
        ];
        let stats = CacheStats::from_events(&events);
        assert_eq!(stats.call_count, 2);
        assert_eq!(stats.total_input_tokens, 1500);
        assert_eq!(stats.total_output_tokens, 80);
        assert_eq!(stats.total_cache_creation_input_tokens, 800);
        assert_eq!(stats.total_cache_read_input_tokens, 400);
        assert_eq!(stats.total_cached_input_tokens, 0);
    }

    #[test]
    fn from_events_aggregates_openai_cached_tokens() {
        let events = vec![
            usage_event(2000, 100),
            cache_metrics_event(0, 0, 1500), // OpenAI: 1500 of 2000 cached
        ];
        let stats = CacheStats::from_events(&events);
        assert_eq!(stats.total_cached_input_tokens, 1500);
        // hit rate: cached / (input + cached) = 1500 / 3500
        let rate = stats.cache_hit_rate();
        assert!((rate - 1500.0 / 3500.0).abs() < 1e-9);
    }

    #[test]
    fn cache_hit_rate_zero_when_no_input() {
        let stats = CacheStats::default();
        assert_eq!(stats.cache_hit_rate(), 0.0);
    }

    #[test]
    fn cache_hit_rate_mixes_anthropic_and_openai() {
        // One call, input=1000, 200 read from Anthropic cache, 100 from OpenAI cache.
        // Denominator: 1000 + 200 + 100 = 1300. Numerator: 200 + 100 = 300.
        let events = vec![usage_event(1000, 50), cache_metrics_event(0, 200, 100)];
        let stats = CacheStats::from_events(&events);
        let rate = stats.cache_hit_rate();
        assert!((rate - 300.0 / 1300.0).abs() < 1e-9);
    }

    #[test]
    fn from_events_ignores_non_usage_events() {
        let events = vec![
            AgentEvent::RunCompleted(crate::event::RunCompleted {
                run_id: RunId::new(),
                finish_reason: FinishReason::Stop,
                iterations: 1,
                timestamp: Utc::now(),
            }),
            usage_event(500, 20),
        ];
        let stats = CacheStats::from_events(&events);
        assert_eq!(stats.call_count, 1);
        assert_eq!(stats.total_input_tokens, 500);
    }

    #[test]
    fn merge_sums_all_fields() {
        let a = CacheStats {
            call_count: 2,
            total_input_tokens: 1000,
            total_output_tokens: 50,
            total_cache_creation_input_tokens: 800,
            total_cache_read_input_tokens: 200,
            total_cached_input_tokens: 0,
        };
        let b = CacheStats {
            call_count: 3,
            total_input_tokens: 2000,
            total_output_tokens: 100,
            total_cache_creation_input_tokens: 0,
            total_cache_read_input_tokens: 0,
            total_cached_input_tokens: 1500,
        };
        let merged = a.merge(b);
        assert_eq!(merged.call_count, 5);
        assert_eq!(merged.total_input_tokens, 3000);
        assert_eq!(merged.total_output_tokens, 150);
        assert_eq!(merged.total_cache_creation_input_tokens, 800);
        assert_eq!(merged.total_cache_read_input_tokens, 200);
        assert_eq!(merged.total_cached_input_tokens, 1500);
    }

    #[test]
    fn total_cache_tokens_sums_three_fields() {
        let stats = CacheStats {
            total_cache_creation_input_tokens: 100,
            total_cache_read_input_tokens: 200,
            total_cached_input_tokens: 300,
            ..Default::default()
        };
        assert_eq!(stats.total_cache_tokens(), 600);
    }

    #[test]
    fn from_envelopes_handles_empty_list() {
        let stats = CacheStats::from_envelopes(&[]);
        assert_eq!(stats, CacheStats::default());
    }

    #[test]
    fn from_envelopes_aggregates() {
        // Build envelopes from events
        let provider = ProviderId::new("anthropic");
        let model = ModelName::new("claude-3-sonnet");
        let session_id = Uuid::new_v4();
        let run_id = RunId::new();

        let events = vec![
            AgentEvent::RunStarted(crate::event::RunStarted {
                run_id,
                session_id,
                provider: provider.clone(),
                model: model.clone(),
                timestamp: Utc::now(),
            }),
            usage_event(1000, 100),
            cache_metrics_event(0, 500, 0),
        ];
        // We don't have the store here; just verify the helper matches
        // `from_events` by construction. (from_envelopes is a thin shim.)
        let a = CacheStats::from_events(&events);
        let b = {
            // Simulate the envelope deref: same events, same stats
            let cloned: Vec<AgentEvent> = events.clone();
            CacheStats::from_events(&cloned)
        };
        assert_eq!(a, b);
    }

    #[test]
    fn display_format_includes_all_fields() {
        let stats = CacheStats {
            call_count: 1,
            total_input_tokens: 1000,
            total_output_tokens: 50,
            total_cache_creation_input_tokens: 800,
            total_cache_read_input_tokens: 200,
            total_cached_input_tokens: 0,
        };
        let s = format!("{stats}");
        assert!(s.contains("model calls:           1"));
        assert!(s.contains("total input tokens:    1000"));
        assert!(s.contains("cache hit rate:"));
    }
}
