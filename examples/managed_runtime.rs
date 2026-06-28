//! Demonstrates the `ManagedRuntime` lifecycle:
//!
//! - Build via `AgentConfigBuilder::build_managed`.
//! - Inspect aggregated health.
//! - Serve until shutdown (Ctrl-C or external token).
//!
//! Run with:
//! ```bash
//! cargo run --example managed_runtime
//! ```

use behest::config::AgentConfigBuilder;
use behest::runtime::ManagedRuntime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // Build a managed runtime from the default configuration.
    // In a real application you would add providers, tools, etc.
    let managed: ManagedRuntime = AgentConfigBuilder::default().build_managed().await?;

    // Inspect health before serving.
    let health = managed.health().await;
    println!("initial health: {health:?}");
    println!("overall: {:?}", managed.overall_health().await);
    println!("ready: {}", managed.is_ready().await);

    // Print the shutdown token so external code can trigger shutdown.
    let token = managed.shutdown_token();
    println!(
        "serving managed runtime… press Ctrl-C or call \
         shutdown_token.signal_shutdown() to stop"
    );

    // In a real application you would register transports (gRPC,
    // HTTP+SSE, etc.) before calling serve. For demonstration we
    // just wait for shutdown.
    tokio::select! {
        result = managed.serve() => {
            if let Err(e) = result {
                eprintln!("serve error: {e}");
            }
        }
        _ = tokio::signal::ctrl_c() => {
            println!("received Ctrl-C, signalling shutdown…");
            token.signal_shutdown();
        }
    }

    println!("managed runtime stopped");
    Ok(())
}
