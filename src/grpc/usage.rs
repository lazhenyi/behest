//! UsageService and MetricsService gRPC implementation.

use tonic::{Request, Response, Status};

use crate::grpc::pb::{
    GetMetricsRequest, GetMetricsResponse, GetPrometheusMetricsRequest,
    GetPrometheusMetricsResponse, GetUsageRequest, GetUsageResponse, UsageRecord as PbUsageRecord,
    metrics_service_server::MetricsService, usage_service_server::UsageService,
};

use super::pb::{Timestamp, TokenUsage};
use std::sync::Arc;

/// gRPC usage service.
pub struct GrpcUsageService {
    state: Arc<super::state::GrpcState>,
}

impl GrpcUsageService {
    /// Creates a new usage service.
    #[must_use]
    pub fn new(state: Arc<super::state::GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl UsageService for GrpcUsageService {
    async fn get_usage(
        &self,
        request: Request<GetUsageRequest>,
    ) -> Result<Response<GetUsageResponse>, Status> {
        let req = request.into_inner();
        let session_id = if req.session_id.is_empty() {
            None
        } else {
            Some(
                uuid::Uuid::parse_str(&req.session_id)
                    .map_err(|_| Status::invalid_argument("invalid session_id"))?,
            )
        };

        let records = if let Some(sid) = session_id {
            self.state
                .runtime
                .store()
                .executions()
                .list_usage(&sid)
                .await
                .map_err(|e| Status::internal(e.to_string()))?
        } else {
            let sessions = self
                .state
                .runtime
                .store()
                .sessions()
                .list_sessions()
                .await
                .map_err(|e| Status::internal(e.to_string()))?;
            let mut all = Vec::new();
            for s in &sessions {
                let mut usage = self
                    .state
                    .runtime
                    .store()
                    .executions()
                    .list_usage(&s.id)
                    .await
                    .map_err(|e| Status::internal(e.to_string()))?;
                all.append(&mut usage);
            }
            all
        };

        let pb_records: Vec<PbUsageRecord> = records
            .iter()
            .map(|r| PbUsageRecord {
                session_id: r.session_id.to_string(),
                total_tokens: Some(TokenUsage {
                    input_tokens: r.input_tokens,
                    output_tokens: r.output_tokens,
                    total_tokens: r.total_tokens,
                }),
                recorded_at: Some(Timestamp {
                    value: r.created_at.to_rfc3339(),
                }),
            })
            .collect();

        let aggregate = TokenUsage {
            input_tokens: records.iter().map(|r| r.input_tokens).sum(),
            output_tokens: records.iter().map(|r| r.output_tokens).sum(),
            total_tokens: records.iter().map(|r| r.total_tokens).sum(),
        };

        Ok(Response::new(GetUsageResponse {
            records: pb_records,
            aggregate: Some(aggregate),
        }))
    }
}

/// gRPC metrics service.
pub struct GrpcMetricsService {
    state: Arc<super::state::GrpcState>,
}

impl GrpcMetricsService {
    /// Creates a new metrics service.
    #[must_use]
    pub fn new(state: Arc<super::state::GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl MetricsService for GrpcMetricsService {
    async fn get_metrics(
        &self,
        _request: Request<GetMetricsRequest>,
    ) -> Result<Response<GetMetricsResponse>, Status> {
        let uptime = self.state.started_at.elapsed().as_secs();
        let active_runs = self.state.run_tasks.active_count().await;
        let total_sessions = self
            .state
            .runtime
            .store()
            .sessions()
            .list_sessions()
            .await
            .map_or(0, |s| s.len());

        let metrics = serde_json::json!({
            "status": "ok",
            "uptime_seconds": uptime,
            "active_runs": active_runs,
            "total_sessions": total_sessions,
        });

        Ok(Response::new(GetMetricsResponse {
            metrics: metrics.to_string(),
        }))
    }

    async fn get_prometheus_metrics(
        &self,
        _request: Request<GetPrometheusMetricsRequest>,
    ) -> Result<Response<GetPrometheusMetricsResponse>, Status> {
        let uptime = self.state.started_at.elapsed().as_secs();
        let active_runs = self.state.run_tasks.active_count().await;
        let total_sessions = self
            .state
            .runtime
            .store()
            .sessions()
            .list_sessions()
            .await
            .map_or(0, |s| s.len());

        let text = format!(
            "# HELP agent_uptime_seconds Server uptime in seconds.\n\
             # TYPE agent_uptime_seconds counter\n\
             agent_uptime_seconds {uptime}\n\
             # HELP agent_active_runs Number of active run tasks.\n\
             # TYPE agent_active_runs gauge\n\
             agent_active_runs {active_runs}\n\
             # HELP agent_total_sessions Total number of sessions.\n\
             # TYPE agent_total_sessions gauge\n\
             agent_total_sessions {total_sessions}\n"
        );

        Ok(Response::new(GetPrometheusMetricsResponse { text }))
    }
}
