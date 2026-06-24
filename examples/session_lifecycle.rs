//! Demonstrates session lifecycle with in-memory store.
//!
//! Run with:
//! ```bash
//! cargo run --example session_lifecycle
//! ```

use behest::prelude::*;
use behest::provider::ModelName;
use chrono::Utc;
use serde_json::Value;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), behest::Error> {
    let store = MemorySessionStore::new();

    let session_id = Uuid::new_v4();
    let session = Session {
        id: session_id,
        title: "Example session".to_string(),
        model: ModelName::new("gpt-4o"),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        metadata: Value::Null,
    };

    let created = store
        .create_session(session)
        .await
        .map_err(behest::Error::Storage)?;

    println!("Created session: {}", created.id);

    let loaded = store
        .get_session(&created.id)
        .await
        .map_err(behest::Error::Storage)?;

    if let Some(found) = loaded {
        println!("  title: {}", found.title);
        println!("  model: {}", found.model);
    }

    store
        .delete_session(&created.id)
        .await
        .map_err(behest::Error::Storage)?;
    println!("Deleted session: {}", created.id);

    Ok(())
}
