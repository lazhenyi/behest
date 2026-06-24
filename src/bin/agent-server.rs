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
    let max_concurrent_runs = config.grpc.max_concurrent_runs;

    let grpc_state = Arc::new(GrpcState::new(
        Arc::clone(&runtime),
        Arc::new(config.clone()),
        Arc::clone(&task_registry),
        max_concurrent_runs,
    ));

    let auth = AuthInterceptor::new(config.grpc.auth_token.clone());

    tracing::info!("gRPC server listening on {addr}");

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
            auth,
        ))
        .serve(addr)
        .await?;

    Ok(())
}
