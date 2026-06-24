//! gRPC agent server binary.
//!
//! Starts a gRPC server exposing the agent runtime as microservices.
//! Configure via environment variables or config files.

use std::net::SocketAddr;
use std::sync::Arc;

use behest::config::AgentConfigBuilder;
use behest::grpc::state::GrpcState;
use behest::grpc::{
    pb::{
        metrics_service_server::MetricsServiceServer, model_service_server::ModelServiceServer,
        provider_service_server::ProviderServiceServer, run_service_server::RunServiceServer,
        session_service_server::SessionServiceServer, tool_service_server::ToolServiceServer,
        usage_service_server::UsageServiceServer,
    },
    provider::{GrpcModelService, GrpcProviderService},
    run::GrpcRunService,
    run::RunTaskRegistry,
    session::GrpcSessionService,
    tool::GrpcToolService,
    usage::{GrpcMetricsService, GrpcUsageService},
};

use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let config = AgentConfigBuilder::default()
        .build()
        .map_err(|e| format!("config error: {e}"))?;

    let addr: SocketAddr = config
        .grpc
        .listen_addr
        .parse()
        .map_err(|e| format!("invalid listen address: {e}"))?;

    let runtime = Arc::new(
        config
            .clone()
            .into_runtime()
            .await
            .map_err(|e| format!("runtime build error: {e}"))?,
    );

    let task_registry = Arc::new(RunTaskRegistry::new());

    let grpc_state = Arc::new(GrpcState::new(
        Arc::clone(&runtime),
        Arc::new(config),
        Arc::clone(&task_registry),
    ));

    tracing::info!("gRPC server listening on {addr}");

    Server::builder()
        .add_service(ProviderServiceServer::new(GrpcProviderService::new(
            Arc::clone(&grpc_state),
        )))
        .add_service(ModelServiceServer::new(GrpcModelService::new(Arc::clone(
            &grpc_state,
        ))))
        .add_service(SessionServiceServer::new(GrpcSessionService::new(
            Arc::clone(&grpc_state),
        )))
        .add_service(RunServiceServer::new(GrpcRunService::new(Arc::clone(
            &grpc_state,
        ))))
        .add_service(ToolServiceServer::new(GrpcToolService::new(Arc::clone(
            &grpc_state,
        ))))
        .add_service(UsageServiceServer::new(GrpcUsageService::new(Arc::clone(
            &grpc_state,
        ))))
        .add_service(MetricsServiceServer::new(GrpcMetricsService::new(
            Arc::clone(&grpc_state),
        )))
        .serve(addr)
        .await?;

    Ok(())
}
