//! Admin and observability gRPC service implementation.
//!
//! Provides runtime status, compaction status, and job pool status
//! endpoints for monitoring the agent server.

use std::sync::Arc;

use tonic::{Request, Response, Status};

use super::pb::{
    GetCompactionStatusRequest, GetCompactionStatusResponse, GetJobPoolStatusRequest,
    GetJobPoolStatusResponse, GetRuntimeStatusRequest, GetRuntimeStatusResponse,
    admin_service_server::AdminService,
};
use super::state::GrpcState;

/// gRPC admin service for runtime observability.
///
/// Exposes RPCs for querying runtime status (uptime, active runs,
/// session count, provider/tool/context adapter counts), compaction
/// configuration, and background job pool status.
pub struct GrpcAdminService {
    state: Arc<GrpcState>,
}

impl GrpcAdminService {
    /// Creates a new admin service backed by the given shared state.
    #[must_use]
    pub fn new(state: Arc<GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl AdminService for GrpcAdminService {
    async fn get_runtime_status(
        &self,
        _request: Request<GetRuntimeStatusRequest>,
    ) -> Result<Response<GetRuntimeStatusResponse>, Status> {
        let uptime = self.state.started_at.elapsed().as_secs();
        let active_runs = self.state.run_tasks.active_count().await;
        let total_sessions = self
            .state
            .runtime
            .sessions()
            .list_sessions()
            .await
            .map_or(0, |s| s.len());
        let provider_count = self.state.runtime.providers().chat_ids().count();
        let tool_count = self.state.runtime.tools().registry().specs().len();
        let context_adapter_count = self.state.runtime.context().adapter_names().count();

        Ok(Response::new(GetRuntimeStatusResponse {
            uptime_seconds: uptime,
            active_runs: u32::try_from(active_runs).unwrap_or(u32::MAX),
            total_sessions: u32::try_from(total_sessions).unwrap_or(u32::MAX),
            provider_count: u32::try_from(provider_count).unwrap_or(u32::MAX),
            tool_count: u32::try_from(tool_count).unwrap_or(u32::MAX),
            context_adapter_count: u32::try_from(context_adapter_count).unwrap_or(u32::MAX),
        }))
    }

    async fn get_compaction_status(
        &self,
        _request: Request<GetCompactionStatusRequest>,
    ) -> Result<Response<GetCompactionStatusResponse>, Status> {
        let config = self.state.runtime.compaction().config();
        Ok(Response::new(GetCompactionStatusResponse {
            auto_enabled: config.auto,
            circuit_breaker_open: false,
            consecutive_failures: 0,
            threshold: config.circuit_breaker_threshold,
        }))
    }

    async fn get_job_pool_status(
        &self,
        _request: Request<GetJobPoolStatusRequest>,
    ) -> Result<Response<GetJobPoolStatusResponse>, Status> {
        let enabled = self.state.runtime.background_jobs().is_some();
        Ok(Response::new(GetJobPoolStatusResponse {
            enabled,
            pending_jobs: 0,
            active_workers: 0,
        }))
    }
}
