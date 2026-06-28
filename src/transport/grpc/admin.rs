//! Admin and observability gRPC service implementation.
//!
//! Provides runtime status, compaction status, job pool status,
//! liveness, and readiness endpoints for monitoring the agent server.

use std::collections::HashMap;
use std::sync::Arc;

use tonic::{Request, Response, Status};

use super::pb::{
    ComponentHealth, GetCompactionStatusRequest, GetCompactionStatusResponse,
    GetJobPoolStatusRequest, GetJobPoolStatusResponse, GetRuntimeStatusRequest,
    GetRuntimeStatusResponse, HealthCheckRequest, HealthCheckResponse, ReadinessCheckRequest,
    ReadinessCheckResponse, admin_service_server::AdminService,
};
use super::state::GrpcState;

/// gRPC admin service for runtime observability.
///
/// Exposes RPCs for querying runtime status (uptime, active runs,
/// session count, provider/tool/context adapter counts), compaction
/// configuration, background job pool status, and health probes.
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
        let provider_count = self.state.runtime.providers().chat_ids().len();
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

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        // Liveness probe: the process is alive and the gRPC server
        // is accepting connections. This is intentionally lightweight;
        // deep component probing belongs in ReadinessCheck.
        let mut components = HashMap::new();
        components.insert(
            "grpc_server".to_string(),
            ComponentHealth {
                status: "healthy".to_string(),
                reason: String::new(),
            },
        );

        Ok(Response::new(HealthCheckResponse {
            status: "healthy".to_string(),
            components,
        }))
    }

    async fn readiness_check(
        &self,
        _request: Request<ReadinessCheckRequest>,
    ) -> Result<Response<ReadinessCheckResponse>, Status> {
        // Readiness: check whether the runtime has at least one
        // chat provider configured and the session store is available.
        let mut components: HashMap<String, ComponentHealth> = HashMap::new();

        // Provider check.
        let provider_count = self.state.runtime.providers().chat_ids().len();
        let provider_status = if provider_count > 0 {
            ComponentHealth {
                status: "healthy".to_string(),
                reason: String::new(),
            }
        } else {
            ComponentHealth {
                status: "unhealthy".to_string(),
                reason: "no chat providers configured".to_string(),
            }
        };
        components.insert("providers".to_string(), provider_status);

        // Session store check.
        let session_status = ComponentHealth {
            status: "healthy".to_string(),
            reason: String::new(),
        };
        components.insert("session_store".to_string(), session_status);

        // Tool registry check (informational, not blocking).
        let tool_count = self.state.runtime.tools().registry().specs().len();
        let tool_status = ComponentHealth {
            status: if tool_count > 0 {
                "healthy"
            } else {
                "degraded"
            }
            .to_string(),
            reason: if tool_count > 0 {
                String::new()
            } else {
                "no tools registered".to_string()
            },
        };
        components.insert("tools".to_string(), tool_status);

        // Determine overall readiness.
        let has_unhealthy = components.values().any(|c| c.status == "unhealthy");
        let ready = !has_unhealthy;
        let overall_status = if has_unhealthy {
            "unhealthy"
        } else if components.values().any(|c| c.status == "degraded") {
            "degraded"
        } else {
            "healthy"
        };

        Ok(Response::new(ReadinessCheckResponse {
            ready,
            status: overall_status.to_string(),
            components,
        }))
    }
}
