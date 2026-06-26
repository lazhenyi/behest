//! Demonstrates per-session key-value data with `MemorySessionDataStore`.
//!
//! Run with:
//! ```bash
//! cargo run --example session_data_store
//! ```

use behest::runtime::{MemorySessionDataStore, SessionDataStore};
use serde_json::json;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), behest::Error> {
    let store = MemorySessionDataStore::new();
    let sid = Uuid::new_v4();

    store
        .set(sid, "theme".into(), json!("dark"))
        .await
        .map_err(|e| behest::Error::Config(format!("set failed: {e}")))?;

    store
        .set(sid, "language".into(), json!("zh-CN"))
        .await
        .map_err(|e| behest::Error::Config(format!("set failed: {e}")))?;

    store
        .set(sid, "max_tokens".into(), json!(4096))
        .await
        .map_err(|e| behest::Error::Config(format!("set failed: {e}")))?;

    println!("Stored 3 keys for session {sid}");

    let theme = store
        .get(sid, "theme")
        .await
        .map_err(|e| behest::Error::Config(format!("get failed: {e}")))?;
    println!("theme = {theme:?}");

    let max_tokens = store
        .get(sid, "max_tokens")
        .await
        .map_err(|e| behest::Error::Config(format!("get failed: {e}")))?;
    println!("max_tokens = {max_tokens:?}");

    let missing = store
        .get(sid, "nonexistent")
        .await
        .map_err(|e| behest::Error::Config(format!("get failed: {e}")))?;
    println!("nonexistent key = {missing:?}");

    store
        .delete(sid, "theme")
        .await
        .map_err(|e| behest::Error::Config(format!("delete failed: {e}")))?;
    println!("Deleted 'theme' key");

    let after_delete = store
        .get(sid, "theme")
        .await
        .map_err(|e| behest::Error::Config(format!("get failed: {e}")))?;
    println!("theme after delete = {after_delete:?}");

    let other_sid = Uuid::new_v4();
    let other_val = store
        .get(other_sid, "language")
        .await
        .map_err(|e| behest::Error::Config(format!("get failed: {e}")))?;
    println!("other session's language = {other_val:?}");

    Ok(())
}
