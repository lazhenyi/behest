//! SnapshotService gRPC implementation.

use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::grpc::pb::{
    DeleteSnapshotRequest, DeleteSnapshotResponse, GetSnapshotRequest, GetSnapshotResponse,
    ListSnapshotsRequest, ListSnapshotsResponse, SnapshotInfo,
    snapshot_service_server::SnapshotService,
};
use crate::runtime::RunId;

use std::sync::Arc;

/// gRPC snapshot service.
pub struct GrpcSnapshotService {
    state: Arc<super::state::GrpcState>,
}

impl GrpcSnapshotService {
    /// Creates a new snapshot service.
    #[must_use]
    pub fn new(state: Arc<super::state::GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl SnapshotService for GrpcSnapshotService {
    async fn list_snapshots(
        &self,
        _request: Request<ListSnapshotsRequest>,
    ) -> Result<Response<ListSnapshotsResponse>, Status> {
        let store = self
            .state
            .runtime
            .snapshot_store()
            .ok_or_else(|| Status::unavailable("snapshot store not configured"))?;

        let snapshots = store
            .list()
            .await
            .map_err(|e| super::runtime_error_to_status(&e))?;

        let infos: Vec<SnapshotInfo> = snapshots.iter().map(snapshot_to_proto).collect();

        Ok(Response::new(ListSnapshotsResponse { snapshots: infos }))
    }

    async fn get_snapshot(
        &self,
        request: Request<GetSnapshotRequest>,
    ) -> Result<Response<GetSnapshotResponse>, Status> {
        let req = request.into_inner();

        let store = self
            .state
            .runtime
            .snapshot_store()
            .ok_or_else(|| Status::unavailable("snapshot store not configured"))?;

        let run_id = parse_run_id(&req.run_id)?;

        let snapshot = store
            .load(run_id)
            .await
            .map_err(|e| super::runtime_error_to_status(&e))?
            .ok_or_else(|| {
                Status::not_found(format!("snapshot for run '{}' not found", req.run_id))
            })?;

        Ok(Response::new(GetSnapshotResponse {
            snapshot: Some(snapshot_to_proto(&snapshot)),
        }))
    }

    async fn delete_snapshot(
        &self,
        request: Request<DeleteSnapshotRequest>,
    ) -> Result<Response<DeleteSnapshotResponse>, Status> {
        let req = request.into_inner();

        let store = self
            .state
            .runtime
            .snapshot_store()
            .ok_or_else(|| Status::unavailable("snapshot store not configured"))?;

        let run_id = parse_run_id(&req.run_id)?;

        store
            .delete(run_id)
            .await
            .map_err(|e| super::runtime_error_to_status(&e))?;

        Ok(Response::new(DeleteSnapshotResponse {}))
    }
}

#[allow(clippy::result_large_err)]
fn parse_run_id(s: &str) -> Result<RunId, Status> {
    let uuid =
        Uuid::parse_str(s).map_err(|e| Status::invalid_argument(format!("invalid run_id: {e}")))?;
    Ok(RunId::from_uuid(uuid))
}

fn snapshot_to_proto(snapshot: &crate::runtime::Snapshot) -> SnapshotInfo {
    use crate::grpc::pb::RunStatus as PbRunStatus;

    let status = match snapshot.status {
        crate::runtime::RunStatus::Pending => PbRunStatus::Pending,
        crate::runtime::RunStatus::SessionLoaded => PbRunStatus::SessionLoaded,
        crate::runtime::RunStatus::BuildingContext => PbRunStatus::BuildingContext,
        crate::runtime::RunStatus::CallingModel => PbRunStatus::CallingModel,
        crate::runtime::RunStatus::WaitingForTools => PbRunStatus::WaitingForTools,
        crate::runtime::RunStatus::Persisting => PbRunStatus::Persisting,
        crate::runtime::RunStatus::Completed => PbRunStatus::Completed,
        crate::runtime::RunStatus::Failed => PbRunStatus::Failed,
        crate::runtime::RunStatus::Cancelled => PbRunStatus::Cancelled,
    };

    SnapshotInfo {
        run_id: snapshot.run_id.to_string(),
        session_id: snapshot.session_id.to_string(),
        status: status as i32,
        iteration: u32::try_from(snapshot.iteration).unwrap_or(u32::MAX),
        total_usage: Some(crate::grpc::pb::TokenUsage {
            input_tokens: snapshot.total_usage.input_tokens,
            output_tokens: snapshot.total_usage.output_tokens,
            total_tokens: snapshot.total_usage.total_tokens,
        }),
        timestamp: Some(crate::grpc::to_prost_timestamp(snapshot.timestamp)),
    }
}
