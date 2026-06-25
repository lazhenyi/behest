//! CompactionService gRPC implementation.
//!
//! Provides RPCs for querying and updating the runtime's session
//! compaction configuration and circuit breaker state.

use tonic::{Request, Response, Status};

use crate::grpc::pb::{
    GetCircuitBreakerRequest, GetCircuitBreakerResponse, GetCompactionConfigRequest,
    GetCompactionConfigResponse, UpdateCompactionConfigRequest, UpdateCompactionConfigResponse,
    compaction_service_server::CompactionService,
};

use std::sync::Arc;

/// gRPC compaction service for session context management.
///
/// Exposes the runtime's compaction config (auto mode, pruning,
/// token budgets, circuit breaker thresholds) and allows monitoring
/// of compaction health.
pub struct GrpcCompactionService {
    state: Arc<super::state::GrpcState>,
}

impl GrpcCompactionService {
    /// Creates a new compaction service backed by the given shared state.
    #[must_use]
    pub fn new(state: Arc<super::state::GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl CompactionService for GrpcCompactionService {
    async fn get_compaction_config(
        &self,
        _request: Request<GetCompactionConfigRequest>,
    ) -> Result<Response<GetCompactionConfigResponse>, Status> {
        let compaction = self.state.runtime.compaction();
        let config = compaction.config();

        Ok(Response::new(GetCompactionConfigResponse {
            auto: config.auto,
            prune: config.prune,
            buffer_tokens: config.buffer_tokens as u64,
            keep_tokens: config.keep_tokens as u64,
            tail_turns: u32::try_from(config.tail_turns).unwrap_or(u32::MAX),
            model: config
                .model
                .as_ref()
                .map_or(String::new(), |m| m.as_str().to_string()),
            provider: config
                .provider
                .as_ref()
                .map_or(String::new(), |p| p.as_str().to_string()),
            circuit_breaker_threshold: config.circuit_breaker_threshold,
        }))
    }

    async fn update_compaction_config(
        &self,
        _request: Request<UpdateCompactionConfigRequest>,
    ) -> Result<Response<UpdateCompactionConfigResponse>, Status> {
        Err(Status::unimplemented(
            "runtime config mutation not yet supported; use AgentConfigBuilder at startup",
        ))
    }

    async fn get_circuit_breaker(
        &self,
        _request: Request<GetCircuitBreakerRequest>,
    ) -> Result<Response<GetCircuitBreakerResponse>, Status> {
        Err(Status::unimplemented(
            "circuit breaker state not yet exposed via accessor",
        ))
    }
}
