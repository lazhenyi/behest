//! Demonstrates appending runtime events and replaying via `MemoryRuntimeEventStore`.
//!
//! Run with:
//! ```bash
//! cargo run --example event_store_replay
//! ```

use behest::provider::{FinishReason, ModelName, ProviderId};
use behest::runtime::{
    AgentEvent, MemoryRuntimeEventStore, RunId, RuntimeEventStore,
    event::{RunCompleted, RunStarted},
};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), behest::Error> {
    let store = MemoryRuntimeEventStore::new();

    let run_id = RunId::new();
    let session_id = Uuid::new_v4();

    let e1 = store
        .append(AgentEvent::RunStarted(RunStarted {
            run_id,
            session_id,
            provider: ProviderId::new("openai"),
            model: ModelName::new("gpt-4o"),
            timestamp: chrono::Utc::now(),
        }))
        .await
        .map_err(|e| behest::Error::Config(format!("append failed: {e}")))?;
    println!("Appended event 1: seq={}", e1.seq);

    let e2 = store
        .append(AgentEvent::RunCompleted(RunCompleted {
            run_id,
            finish_reason: FinishReason::Stop,
            iterations: 3,
            timestamp: chrono::Utc::now(),
        }))
        .await
        .map_err(|e| behest::Error::Config(format!("append failed: {e}")))?;
    println!("Appended event 2: seq={}", e2.seq);

    let events = store
        .list_after(run_id, None, 10)
        .await
        .map_err(|e| behest::Error::Config(format!("list failed: {e}")))?;
    println!("\nReplayed {} events for run {run_id}:", events.len());
    for env in &events {
        let label = match &env.event {
            AgentEvent::RunStarted(_) => "RunStarted",
            AgentEvent::RunCompleted(_) => "RunCompleted",
            AgentEvent::ModelStarted(_) => "ModelStarted",
            AgentEvent::TextDelta(_) => "TextDelta",
            _ => "Other",
        };
        println!("  seq={}: {label}", env.seq);
    }

    let after = store
        .list_after(run_id, Some(1), 10)
        .await
        .map_err(|e| behest::Error::Config(format!("list failed: {e}")))?;
    println!("\nEvents after seq=1: {} event(s)", after.len());

    let unknown = store
        .list_after(RunId::new(), None, 10)
        .await
        .map_err(|e| behest::Error::Config(format!("list failed: {e}")))?;
    println!("Events for unknown run: {}", unknown.len());

    Ok(())
}
