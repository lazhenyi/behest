//! Demonstrates snapshot persistence and compaction configuration.
//!
//! Run with:
//! ```bash
//! cargo run --example snapshot_compaction
//! ```

use std::path::PathBuf;

use behest::provider::{ModelName, ProviderId, TokenUsage};
use behest::runtime::{
    CompactionConfig, FileSnapshotStore, RunId, RunRequest, RunStatus, Snapshot, SnapshotStore,
    TurnState,
};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), behest::Error> {
    let snapshot_dir = PathBuf::from("/tmp/behest-snapshots");
    let store = FileSnapshotStore::new(snapshot_dir);

    let run_id = RunId::new();
    let session_id = Uuid::new_v4();

    let snap = Snapshot {
        run_id,
        session_id,
        status: RunStatus::CallingModel,
        iteration: 1,
        current_state: TurnState::CallingModel,
        total_usage: TokenUsage {
            input_tokens: 150,
            output_tokens: 42,
            total_tokens: 192,
        },
        last_finish: None,
        assistant_message: None,
        assistant_msg_id: None,
        request: RunRequest::new(ProviderId::new("openai"), ModelName::new("gpt-4o"), "Hello"),
        output_recovery_count: 0,
        timestamp: chrono::Utc::now(),
    };

    store
        .save(&snap)
        .await
        .map_err(|e| behest::Error::Config(e.to_string()))?;
    println!("Snapshot saved: run_id={run_id}");

    let loaded = store
        .load(run_id)
        .await
        .map_err(|e| behest::Error::Config(e.to_string()))?;
    if let Some(s) = loaded {
        println!(
            "Snapshot loaded: status={:?}, iteration={}",
            s.status, s.iteration
        );
    }

    let list = store
        .list()
        .await
        .map_err(|e| behest::Error::Config(e.to_string()))?;
    println!("Active snapshots: {}", list.len());

    store
        .delete(run_id)
        .await
        .map_err(|e| behest::Error::Config(e.to_string()))?;
    println!("Snapshot deleted: run_id={run_id}");

    let config = CompactionConfig::new()
        .with_auto_disabled()
        .with_buffer_tokens(10_000)
        .with_keep_tokens(4_000)
        .with_tail_turns(1);

    println!(
        "Compaction config: auto={}, buffer={}, keep={}, tail_turns={}",
        config.auto, config.buffer_tokens, config.keep_tokens, config.tail_turns,
    );

    Ok(())
}
