//! Demonstrates hot-swap reload via `ManagedRuntime::reload`.
//!
//! The example constructs a `ManagedRuntime`, registers a mock
//! component via `register_typed`, and then performs a live reload
//! while holding an `Arc` reference to the old instance. The
//! drain-aware protocol ensures the old instance stays alive until
//! all outstanding references are dropped.
//!
//! Run with:
//! ```bash
//! cargo run --example hot_swap
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use behest::config::AgentConfigBuilder;
use behest::health::HealthStatus;
use behest::runtime::component::{Component, ComponentContext};
use schemars::JsonSchema;
use serde::Deserialize;

/// A trivial component that just holds a version string.
#[derive(Debug)]
struct VersionedComponent {
    version: String,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct VersionedConfig {
    version: String,
}

#[async_trait]
impl Component for VersionedComponent {
    const NAME: &'static str = "versioned";
    type Config = VersionedConfig;
    type Error = std::io::Error;

    async fn init(cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
        Ok(Self {
            version: cfg.version.clone(),
        })
    }

    async fn start(&self) -> Result<(), Self::Error> {
        println!("[{}] started", self.version);
        Ok(())
    }

    async fn stop(&self) -> Result<(), Self::Error> {
        println!("[{}] stopped", self.version);
        Ok(())
    }

    async fn health(&self) -> HealthStatus {
        HealthStatus::healthy()
    }

    async fn pre_replace_hook(&self) -> Result<(), Self::Error> {
        println!("[{}] pre_replace_hook", self.version);
        Ok(())
    }

    async fn post_replace_hook(&self) -> Result<(), Self::Error> {
        println!("[{}] post_replace_hook (drain complete)", self.version);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let managed = AgentConfigBuilder::default().build_managed().await?;

    // Register a component via the factory system.
    managed.registry().register_typed::<VersionedComponent>(
        "versioned",
        serde_json::json!({ "version": "v1.0.0" }),
    )?;
    managed.registry().init_all().await?;
    managed.registry().start_all().await?;

    // Hold a reference to v1 — simulates an in-flight request.
    let old_ref: Arc<VersionedComponent> = managed.component::<VersionedComponent>("versioned")?;
    println!("holding Arc to old instance: {:?}", old_ref.version);

    // Perform hot-swap: replace with v2 while old_ref is still alive.
    let v2 = VersionedComponent {
        version: "v2.0.0".to_owned(),
    };
    let old_arc = managed
        .reload::<VersionedComponent>("versioned", v2)
        .await?;
    println!(
        "reload complete; old Arc still alive: {:?}",
        old_arc.version
    );

    // Verify that the registry now holds v2.
    let current: Arc<VersionedComponent> = managed.component::<VersionedComponent>("versioned")?;
    println!("current instance: {:?}", current.version);
    assert_eq!(current.version, "v2.0.0");

    // Drop references — the old instance is naturally drained.
    drop(old_ref);
    drop(old_arc);
    println!("all references to v1 dropped — drain complete");

    // Health check after reload.
    println!("overall health: {:?}", managed.overall_health().await);

    // Trigger clean shutdown.
    managed.shutdown_token().signal_shutdown();
    managed.registry().stop_all().await?;
    println!("done");
    Ok(())
}
