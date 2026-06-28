//! Integration tests for `ManagedRuntime` lifecycle and health.

#![allow(clippy::expect_used)]

use std::sync::Arc;

use async_trait::async_trait;
use behest::config::AgentConfigBuilder;
use behest::health::HealthStatus;
use behest::runtime::ManagedRuntime;
use behest::runtime::component::{Component, ComponentContext};
use schemars::JsonSchema;
use serde::Deserialize;

async fn build_managed() -> ManagedRuntime {
    AgentConfigBuilder::default()
        .build_managed()
        .await
        .expect("build_managed")
}

/// A trivial component used in tests.
#[derive(Debug)]
struct StubComponent {
    label: String,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct StubConfig {
    label: String,
}

#[async_trait]
impl Component for StubComponent {
    const NAME: &'static str = "stub";
    type Config = StubConfig;
    type Error = std::io::Error;

    async fn init(cfg: &Self::Config, _ctx: &ComponentContext) -> Result<Self, Self::Error> {
        Ok(Self {
            label: cfg.label.clone(),
        })
    }

    async fn start(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn stop(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn health(&self) -> HealthStatus {
        HealthStatus::healthy()
    }
}

#[tokio::test]
async fn managed_runtime_health_empty_registry() {
    let managed = build_managed().await;
    // Empty registry: all components (none) are healthy.
    assert!(managed.is_healthy().await);
    assert!(managed.is_ready().await);
}

#[tokio::test]
async fn managed_runtime_component_lookup() {
    let managed = build_managed().await;
    managed
        .registry()
        .register_typed::<StubComponent>("stub", serde_json::json!({ "label": "test" }))
        .expect("register");
    managed.registry().init_all().await.expect("init");
    managed.registry().start_all().await.expect("start");

    let got: Arc<StubComponent> = managed.component::<StubComponent>("stub").expect("lookup");
    assert_eq!(got.label, "test");
}

#[tokio::test]
async fn managed_runtime_aggregated_health() {
    let managed = build_managed().await;
    managed
        .registry()
        .register_typed::<StubComponent>("ok", serde_json::json!({ "label": "ok" }))
        .expect("register");
    managed.registry().init_all().await.expect("init");
    managed.registry().start_all().await.expect("start");

    let health = managed.health().await;
    assert!(health.values().all(HealthStatus::is_healthy));
    assert!(managed.overall_health().await.is_healthy());
}

#[tokio::test]
async fn managed_runtime_reload_replaces_instance() {
    let managed = build_managed().await;
    managed
        .registry()
        .register_typed::<StubComponent>("stub", serde_json::json!({ "label": "v1" }))
        .expect("register");
    managed.registry().init_all().await.expect("init");
    managed.registry().start_all().await.expect("start");

    let v2 = StubComponent {
        label: "v2".to_owned(),
    };
    let old = managed
        .reload::<StubComponent>("stub", v2)
        .await
        .expect("reload");
    assert_eq!(old.label, "v1");

    let current: Arc<StubComponent> = managed.component::<StubComponent>("stub").expect("lookup");
    assert_eq!(current.label, "v2");
}

#[tokio::test]
async fn managed_runtime_serve_and_shutdown() {
    let managed = build_managed().await;

    let shutdown = managed.shutdown_token();
    let handle = tokio::spawn(async move { managed.serve().await });

    // Signal shutdown immediately.
    shutdown.signal_shutdown();

    let result = handle.await.expect("join");
    assert!(result.is_ok());
}

#[tokio::test]
async fn managed_runtime_healthz_json() {
    let managed = build_managed().await;
    let json = managed.healthz_json().await;
    // Should contain a "status" field and a "components" object.
    assert!(json.get("status").is_some());
    assert!(json.get("components").is_some());
}
