//! gRPC agent server binary.
//!
//! Starts a gRPC server exposing the agent runtime as microservices.
//! Configure via environment variables or config files.

use std::net::SocketAddr;
use std::sync::Arc;

use behest::config::{AgentConfigBuilder, ProviderConfig, ProviderType};
use behest::grpc::state::{GrpcState, ModelCatalogEntry, ProviderConfigMap};
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
use behest::provider::ProviderId;

use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let config = AgentConfigBuilder::default()
        .build()
        .map_err(|e| format!("config error: {e}"))?;

    let runtime = Arc::new(
        AgentConfigBuilder::default()
            .build_runtime()
            .await
            .map_err(|e| format!("runtime build error: {e}"))?,
    );

    let provider_configs = ProviderConfigMap::new(config.providers.clone());
    let model_catalog = build_model_catalog(&config.providers);

    let grpc_state = Arc::new(GrpcState::new(
        Arc::clone(&runtime),
        model_catalog,
        provider_configs,
        behest::tool::ToolRegistry::new(),
    ));

    let task_registry = Arc::new(RunTaskRegistry::new());

    let addr: SocketAddr = "[::1]:50051".parse()?;
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
        .add_service(RunServiceServer::new(GrpcRunService::new(
            Arc::clone(&grpc_state),
            Arc::clone(&task_registry),
        )))
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

fn build_model_catalog(
    providers: &std::collections::HashMap<ProviderId, ProviderConfig>,
) -> Vec<ModelCatalogEntry> {
    let mut catalog = Vec::new();

    for (provider_id, cfg) in providers {
        let has_chat = match cfg.provider_type {
            #[cfg(feature = "openai")]
            Some(ProviderType::OpenAi) => true,
            #[cfg(feature = "anthropic")]
            Some(ProviderType::Anthropic) => true,
            None => false,
            #[allow(unreachable_patterns)]
            _ => false,
        };

        if let Some(ref default_model) = cfg.model {
            catalog.push(ModelCatalogEntry {
                provider: provider_id.clone(),
                model: default_model.clone(),
                streaming: has_chat,
                tool_calling: has_chat,
            });
        }

        for model in &cfg.models {
            catalog.push(ModelCatalogEntry {
                provider: provider_id.clone(),
                model: model.clone(),
                streaming: has_chat,
                tool_calling: has_chat,
            });
        }
    }

    catalog
}
