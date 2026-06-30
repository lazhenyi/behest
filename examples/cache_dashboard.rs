//! Demonstrates aggregating prompt cache statistics from a runtime event
//! store.
//!
//! Synthesizes a `MemoryRuntimeEventStore` with `RunStarted`,
//! `UsageRecorded`, and `CacheMetrics` events across three model calls,
//! then replays them through `CacheStats::from_envelopes` to print the
//! aggregated cache effectiveness.
//!
//! Run with:
//! ```bash
//! cargo run --example cache_dashboard
//! ```

use behest::provider::{FinishReason, ModelName, ProviderId, TokenUsage};
use behest::runtime::event::{CacheMetrics, RunCompleted, RunStarted, UsageRecorded};
use behest::runtime::{AgentEvent, CacheStats, MemoryRuntimeEventStore, RunId, RuntimeEventStore};
use chrono::Utc;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = MemoryRuntimeEventStore::new();
    let run_id = RunId::new();
    let session_id = Uuid::new_v4();
    let provider = ProviderId::new("anthropic");
    let model = ModelName::new("claude-3-sonnet");
    let now = Utc::now();

    // 1. Run start.
    store
        .append(AgentEvent::RunStarted(RunStarted {
            run_id,
            session_id,
            provider: provider.clone(),
            model: model.clone(),
            timestamp: now,
        }))
        .await?;

    // 2. Three model calls with different cache outcomes:
    //    - call 1: cold start, system + tools get written to cache
    //    - call 2: 800 of 1000 input tokens served from cache
    //    - call 3: 950 of 1000 input tokens served from cache
    let scenarios: [(u64, u64, u64, u64, u64, u64); 3] = [
        // (input, output, cache_creation, cache_read, cached, _ignored)
        (1000, 50, 800, 0, 0, 0),
        (1000, 40, 0, 800, 0, 0),
        (1000, 30, 0, 950, 0, 0),
    ];

    for (input, output, creation, read, cached, _) in scenarios {
        store
            .append(AgentEvent::UsageRecorded(UsageRecorded {
                run_id,
                usage: TokenUsage::new(input, output),
                timestamp: Utc::now(),
            }))
            .await?;
        store
            .append(AgentEvent::CacheMetrics(CacheMetrics {
                run_id,
                cache_creation_input_tokens: creation,
                cache_read_input_tokens: read,
                cached_input_tokens: cached,
                timestamp: Utc::now(),
            }))
            .await?;
    }

    // 3. Run completes.
    store
        .append(AgentEvent::RunCompleted(RunCompleted {
            run_id,
            finish_reason: FinishReason::Stop,
            iterations: 3,
            timestamp: Utc::now(),
        }))
        .await?;

    // 4. Replay and aggregate.
    let envelopes = store.list_after(run_id, None, 1024).await?;
    println!("Replayed {} events for run {}\n", envelopes.len(), run_id);

    let stats = CacheStats::from_envelopes(&envelopes);
    print!("{stats}");

    // 5. Sanity check.
    assert_eq!(stats.call_count, 3);
    assert_eq!(stats.total_input_tokens, 3000);
    assert_eq!(stats.total_output_tokens, 120);
    assert_eq!(stats.total_cache_creation_input_tokens, 800);
    assert_eq!(stats.total_cache_read_input_tokens, 1750);
    assert_eq!(stats.total_cached_input_tokens, 0);

    // Hit rate: (800 + 950) / (3000 + 800 + 1750 + 0) = 1750 / 5550 ≈ 0.3153
    let rate = stats.cache_hit_rate();
    println!("\n  expected hit rate: {:.2}%", 1750.0 / 5550.0 * 100.0);
    assert!((rate - 1750.0 / 5550.0).abs() < 1e-9);

    Ok(())
}
