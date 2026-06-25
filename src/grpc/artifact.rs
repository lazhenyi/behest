//! ArtifactService gRPC implementation.

use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::grpc::pb::{
    Artifact as PbArtifact, DeleteArtifactRequest, DeleteArtifactResponse,
    DeleteArtifactsBySessionRequest, DeleteArtifactsBySessionResponse, GetArtifactRequest,
    GetArtifactResponse, ListArtifactsRequest, ListArtifactsResponse, PutArtifactRequest,
    PutArtifactResponse, artifact_service_server::ArtifactService,
};

use std::sync::Arc;

/// gRPC artifact service.
pub struct GrpcArtifactService {
    state: Arc<super::state::GrpcState>,
}

impl GrpcArtifactService {
    /// Creates a new artifact service.
    #[must_use]
    pub fn new(state: Arc<super::state::GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl ArtifactService for GrpcArtifactService {
    async fn put_artifact(
        &self,
        request: Request<PutArtifactRequest>,
    ) -> Result<Response<PutArtifactResponse>, Status> {
        let req = request.into_inner();

        let store = self
            .state
            .runtime
            .store()
            .artifacts()
            .ok_or_else(|| Status::unavailable("artifact store not configured"))?;

        let session_id = Uuid::parse_str(&req.session_id)
            .map_err(|e| Status::invalid_argument(format!("invalid session_id: {e}")))?;

        let metadata: serde_json::Value = if req.metadata.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(&req.metadata)
                .map_err(|e| Status::invalid_argument(format!("invalid metadata JSON: {e}")))?
        };

        let artifact = crate::store::Artifact {
            id: Uuid::new_v4(),
            session_id: Some(session_id),
            name: req.name,
            content_type: req.content_type,
            data: req.data,
            metadata,
            created_at: chrono::Utc::now(),
        };

        let saved = store
            .put(artifact)
            .await
            .map_err(|e| super::error_to_status(e.into()))?;

        Ok(Response::new(PutArtifactResponse {
            artifact: Some(artifact_to_proto(saved)),
        }))
    }

    async fn get_artifact(
        &self,
        request: Request<GetArtifactRequest>,
    ) -> Result<Response<GetArtifactResponse>, Status> {
        let req = request.into_inner();

        let store = self
            .state
            .runtime
            .store()
            .artifacts()
            .ok_or_else(|| Status::unavailable("artifact store not configured"))?;

        let id = Uuid::parse_str(&req.id)
            .map_err(|e| Status::invalid_argument(format!("invalid id: {e}")))?;

        let artifact = store
            .get(&id)
            .await
            .map_err(|e| super::error_to_status(e.into()))?
            .ok_or_else(|| Status::not_found(format!("artifact '{id}' not found")))?;

        Ok(Response::new(GetArtifactResponse {
            artifact: Some(artifact_to_proto(artifact)),
        }))
    }

    async fn delete_artifact(
        &self,
        request: Request<DeleteArtifactRequest>,
    ) -> Result<Response<DeleteArtifactResponse>, Status> {
        let req = request.into_inner();

        let store = self
            .state
            .runtime
            .store()
            .artifacts()
            .ok_or_else(|| Status::unavailable("artifact store not configured"))?;

        let id = Uuid::parse_str(&req.id)
            .map_err(|e| Status::invalid_argument(format!("invalid id: {e}")))?;

        store
            .delete(&id)
            .await
            .map_err(|e| super::error_to_status(e.into()))?;

        Ok(Response::new(DeleteArtifactResponse {}))
    }

    async fn list_artifacts(
        &self,
        request: Request<ListArtifactsRequest>,
    ) -> Result<Response<ListArtifactsResponse>, Status> {
        let req = request.into_inner();

        let store = self
            .state
            .runtime
            .store()
            .artifacts()
            .ok_or_else(|| Status::unavailable("artifact store not configured"))?;

        let session_id = Uuid::parse_str(&req.session_id)
            .map_err(|e| Status::invalid_argument(format!("invalid session_id: {e}")))?;

        let artifacts = store
            .list_by_session(&session_id)
            .await
            .map_err(|e| super::error_to_status(e.into()))?;

        let pb_artifacts: Vec<PbArtifact> = artifacts.into_iter().map(artifact_to_proto).collect();

        Ok(Response::new(ListArtifactsResponse {
            artifacts: pb_artifacts,
        }))
    }

    async fn delete_artifacts_by_session(
        &self,
        request: Request<DeleteArtifactsBySessionRequest>,
    ) -> Result<Response<DeleteArtifactsBySessionResponse>, Status> {
        let req = request.into_inner();

        let store = self
            .state
            .runtime
            .store()
            .artifacts()
            .ok_or_else(|| Status::unavailable("artifact store not configured"))?;

        let session_id = Uuid::parse_str(&req.session_id)
            .map_err(|e| Status::invalid_argument(format!("invalid session_id: {e}")))?;

        let deleted = store
            .delete_by_session(&session_id)
            .await
            .map_err(|e| super::error_to_status(e.into()))?;

        Ok(Response::new(DeleteArtifactsBySessionResponse { deleted }))
    }
}

fn artifact_to_proto(artifact: crate::store::Artifact) -> PbArtifact {
    PbArtifact {
        id: artifact.id.to_string(),
        session_id: artifact
            .session_id
            .map_or_else(String::new, |id| id.to_string()),
        name: artifact.name,
        content_type: artifact.content_type,
        data: artifact.data,
        metadata: artifact.metadata.to_string(),
        created_at: Some(crate::grpc::to_prost_timestamp(artifact.created_at)),
    }
}
