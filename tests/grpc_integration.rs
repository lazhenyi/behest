//! gRPC service integration tests.
//!
//! Tests gRPC service implementations directly without starting a server.

#![cfg(feature = "server")]
#![allow(clippy::expect_used)]

use std::sync::Arc;

use behest::config::AgentConfigBuilder;
use behest::transport::grpc::admin::GrpcAdminService;
use behest::transport::grpc::pb::{
    GetCompactionStatusRequest, GetJobPoolStatusRequest, GetRuntimeStatusRequest,
    admin_service_server::AdminService,
};
use behest::transport::grpc::run::RunTaskRegistry;
use behest::transport::grpc::state::GrpcState;
use tonic::Request;

async fn setup() -> Arc<GrpcState> {
    let config = AgentConfigBuilder::default().build().expect("config");
    let runtime = Arc::new(config.clone().into_runtime().await.expect("runtime"));
    let task_registry = Arc::new(RunTaskRegistry::new());
    Arc::new(GrpcState::new(
        runtime,
        Arc::new(config),
        task_registry,
        None,
    ))
}

#[tokio::test]
async fn admin_get_runtime_status() {
    let state = setup().await;
    let service = GrpcAdminService::new(state);
    let response = service
        .get_runtime_status(Request::new(GetRuntimeStatusRequest {}))
        .await
        .expect("status");
    let status = response.into_inner();
    assert_eq!(status.active_runs, 0);
    assert_eq!(status.provider_count, 0);
    assert_eq!(status.tool_count, 0);
}

#[tokio::test]
async fn admin_get_compaction_status() {
    let state = setup().await;
    let service = GrpcAdminService::new(state);
    let response = service
        .get_compaction_status(Request::new(GetCompactionStatusRequest {}))
        .await
        .expect("status");
    let status = response.into_inner();
    assert!(!status.circuit_breaker_open);
    assert_eq!(status.consecutive_failures, 0);
}

#[tokio::test]
async fn admin_get_job_pool_status() {
    let state = setup().await;
    let service = GrpcAdminService::new(state);
    let response = service
        .get_job_pool_status(Request::new(GetJobPoolStatusRequest {}))
        .await
        .expect("status");
    let status = response.into_inner();
    assert!(!status.enabled);
    assert_eq!(status.pending_jobs, 0);
}
