//! gRPC agent server binary.
//!
//! Starts a gRPC server exposing the agent runtime as microservices.
//! Configure via environment variables or config files.

use std::net::SocketAddr;
use std::sync::Arc;

use behest::config::AgentConfigBuilder;
use behest::grpc::auth::AuthInterceptor;
use behest::grpc::state::GrpcState;
use behest::grpc::{
    admin::GrpcAdminService,
    agent_grpc::GrpcAgentService,
    artifact::GrpcArtifactService,
    compaction::GrpcCompactionService,
    context::GrpcContextService,
    embedding::GrpcEmbeddingService,
    pb::{
        admin_service_server::AdminServiceServer, agent_service_server::AgentServiceServer,
        artifact_service_server::ArtifactServiceServer,
        compaction_service_server::CompactionServiceServer,
        context_service_server::ContextServiceServer,
        embedding_service_server::EmbeddingServiceServer,
        metrics_service_server::MetricsServiceServer, model_service_server::ModelServiceServer,
        provider_service_server::ProviderServiceServer, run_service_server::RunServiceServer,
        session_service_server::SessionServiceServer,
        snapshot_service_server::SnapshotServiceServer, tool_service_server::ToolServiceServer,
        usage_service_server::UsageServiceServer,
    },
    provider::{GrpcModelService, GrpcProviderService},
    run::GrpcRunService,
    run::RunTaskRegistry,
    session::GrpcSessionService,
    snapshot::GrpcSnapshotService,
    tool::GrpcToolService,
    usage::{GrpcMetricsService, GrpcUsageService},
};

use tonic::transport::Server;

#[tokio::main]
#[allow(clippy::too_many_lines)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

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
    let max_concurrent_runs = config.grpc.max_concurrent_runs;

    let grpc_state = Arc::new(GrpcState::new(
        Arc::clone(&runtime),
        Arc::new(config.clone()),
        Arc::clone(&task_registry),
        max_concurrent_runs,
    ));

    let auth = AuthInterceptor::new(config.grpc.auth_token.clone());

    tracing::info!("gRPC server listening on {addr}");

    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<behest::grpc::pb::provider_service_server::ProviderServiceServer<
            GrpcProviderService,
        >>()
        .await;

    let mut server = Server::builder();

    if let Some(ref tls) = config.grpc.tls {
        use tonic::transport::{Certificate, Identity, ServerTlsConfig};

        let cert = tokio::fs::read(&tls.cert_path)
            .await
            .map_err(|e| format!("failed to read TLS cert: {e}"))?;
        let key = tokio::fs::read(&tls.key_path)
            .await
            .map_err(|e| format!("failed to read TLS key: {e}"))?;
        let identity = Identity::from_pem(cert, key);

        let mut tls_config = ServerTlsConfig::new().identity(identity);

        if let Some(ref ca_path) = tls.client_ca_path {
            let ca = tokio::fs::read(ca_path)
                .await
                .map_err(|e| format!("failed to read client CA: {e}"))?;
            tls_config = tls_config.client_ca_root(Certificate::from_pem(ca));
        }

        server = server.tls_config(tls_config)?;
        tracing::info!("TLS enabled");
    }

    server
        .add_service(ProviderServiceServer::with_interceptor(
            GrpcProviderService::new(Arc::clone(&grpc_state)),
            auth.clone(),
        ))
        .add_service(ModelServiceServer::with_interceptor(
            GrpcModelService::new(Arc::clone(&grpc_state)),
            auth.clone(),
        ))
        .add_service(SessionServiceServer::with_interceptor(
            GrpcSessionService::new(Arc::clone(&grpc_state)),
            auth.clone(),
        ))
        .add_service(RunServiceServer::with_interceptor(
            GrpcRunService::new(Arc::clone(&grpc_state)),
            auth.clone(),
        ))
        .add_service(ToolServiceServer::with_interceptor(
            GrpcToolService::new(Arc::clone(&grpc_state)),
            auth.clone(),
        ))
        .add_service(UsageServiceServer::with_interceptor(
            GrpcUsageService::new(Arc::clone(&grpc_state)),
            auth.clone(),
        ))
        .add_service(MetricsServiceServer::with_interceptor(
            GrpcMetricsService::new(Arc::clone(&grpc_state)),
            auth.clone(),
        ))
        .add_service(EmbeddingServiceServer::with_interceptor(
            GrpcEmbeddingService::new(Arc::clone(&grpc_state)),
            auth.clone(),
        ))
        .add_service(ArtifactServiceServer::with_interceptor(
            GrpcArtifactService::new(Arc::clone(&grpc_state)),
            auth.clone(),
        ))
        .add_service(AgentServiceServer::with_interceptor(
            GrpcAgentService::new(Arc::clone(&grpc_state)),
            auth.clone(),
        ))
        .add_service(ContextServiceServer::with_interceptor(
            GrpcContextService::new(Arc::clone(&grpc_state)),
            auth.clone(),
        ))
        .add_service(CompactionServiceServer::with_interceptor(
            GrpcCompactionService::new(Arc::clone(&grpc_state)),
            auth.clone(),
        ))
        .add_service(SnapshotServiceServer::with_interceptor(
            GrpcSnapshotService::new(Arc::clone(&grpc_state)),
            auth.clone(),
        ))
        .add_service(AdminServiceServer::with_interceptor(
            GrpcAdminService::new(Arc::clone(&grpc_state)),
            auth,
        ))
        .add_service(health_service)
        .serve_with_shutdown(addr, async {
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("shutdown signal received, starting graceful shutdown");
        })
        .await?;

    Ok(())
}
