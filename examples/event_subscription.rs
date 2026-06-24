//! Demonstrates subscribing to runtime event streams.
//!
//! Run with:
//! ```bash
//! cargo run --example event_subscription
//! ```
//!
//! This example creates a runtime and subscribes to its broadcast channel
//! to show the event-driven architecture pattern. With `openai` or `anthropic`
//! features, you can add a provider and see real streaming events.

use behest::prelude::*;
use behest::runtime::AgentEvent;
use std::sync::Arc;
use tokio::sync::broadcast;

#[tokio::main]
async fn main() -> Result<(), behest::Error> {
    let config = AgentConfigBuilder::default().build()?;

    let runtime = Arc::new(config.into_runtime().await?);

    let mut events: broadcast::Receiver<AgentEvent> = runtime.subscribe();

    tokio::spawn(async move {
        while let Ok(event) = events.recv().await {
            tracing::info!("Received event: {event:?}");
        }
    });

    let policy = runtime.policy();
    println!("Runtime ready.");
    println!("  max_iterations:       {}", policy.max_iterations);
    println!("  max_tool_concurrency: {}", policy.max_tool_concurrency);
    println!("  tool_timeout:         {:?}", policy.tool_timeout);
    println!("  provider_timeout:     {:?}", policy.provider_timeout);

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    Ok(())
}
