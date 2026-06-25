//! EmbeddingService gRPC implementation.

use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::grpc::pb::{
    DeleteEmbeddingRequest, DeleteEmbeddingResponse, DeleteEmbeddingsBySessionRequest,
    DeleteEmbeddingsBySessionResponse, EmbeddingRecord as PbEmbeddingRecord,
    ScoredEmbedding as PbScoredEmbedding, SearchEmbeddingsRequest, SearchEmbeddingsResponse,
    UpsertEmbeddingRequest, UpsertEmbeddingResponse, embedding_service_server::EmbeddingService,
};

use std::sync::Arc;

/// gRPC embedding service.
pub struct GrpcEmbeddingService {
    state: Arc<super::state::GrpcState>,
}

impl GrpcEmbeddingService {
    /// Creates a new embedding service.
    #[must_use]
    pub fn new(state: Arc<super::state::GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl EmbeddingService for GrpcEmbeddingService {
    async fn upsert_embedding(
        &self,
        request: Request<UpsertEmbeddingRequest>,
    ) -> Result<Response<UpsertEmbeddingResponse>, Status> {
        let req = request.into_inner();

        let store = self
            .state
            .runtime
            .embeddings()
            .ok_or_else(|| Status::unavailable("embedding store not configured"))?;

        let session_id = Uuid::parse_str(&req.session_id)
            .map_err(|e| Status::invalid_argument(format!("invalid session_id: {e}")))?;

        let metadata: serde_json::Value = if req.metadata.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(&req.metadata)
                .map_err(|e| Status::invalid_argument(format!("invalid metadata JSON: {e}")))?
        };

        let record = crate::store::EmbeddingRecord {
            id: Uuid::new_v4(),
            session_id: Some(session_id),
            model: req.model,
            vector: req.vector,
            metadata,
            created_at: chrono::Utc::now(),
        };

        let saved = store
            .upsert(record)
            .await
            .map_err(|e| super::error_to_status(e.into()))?;

        Ok(Response::new(UpsertEmbeddingResponse {
            record: Some(embedding_record_to_proto(saved)),
        }))
    }

    async fn search_embeddings(
        &self,
        request: Request<SearchEmbeddingsRequest>,
    ) -> Result<Response<SearchEmbeddingsResponse>, Status> {
        let req = request.into_inner();

        let store = self
            .state
            .runtime
            .embeddings()
            .ok_or_else(|| Status::unavailable("embedding store not configured"))?;

        let limit = if req.limit == 0 {
            10
        } else {
            req.limit as usize
        };

        let results = store
            .search(&req.query, limit)
            .await
            .map_err(|e| super::error_to_status(e.into()))?;

        let scored: Vec<PbScoredEmbedding> = results
            .into_iter()
            .map(|s| PbScoredEmbedding {
                record: Some(embedding_record_to_proto(s.record)),
                score: f64::from(s.score),
            })
            .collect();

        Ok(Response::new(SearchEmbeddingsResponse { results: scored }))
    }

    async fn delete_embedding(
        &self,
        request: Request<DeleteEmbeddingRequest>,
    ) -> Result<Response<DeleteEmbeddingResponse>, Status> {
        let req = request.into_inner();

        let store = self
            .state
            .runtime
            .embeddings()
            .ok_or_else(|| Status::unavailable("embedding store not configured"))?;

        let id = Uuid::parse_str(&req.id)
            .map_err(|e| Status::invalid_argument(format!("invalid id: {e}")))?;

        store
            .delete(&id)
            .await
            .map_err(|e| super::error_to_status(e.into()))?;

        Ok(Response::new(DeleteEmbeddingResponse {}))
    }

    async fn delete_embeddings_by_session(
        &self,
        request: Request<DeleteEmbeddingsBySessionRequest>,
    ) -> Result<Response<DeleteEmbeddingsBySessionResponse>, Status> {
        let req = request.into_inner();

        let store = self
            .state
            .runtime
            .embeddings()
            .ok_or_else(|| Status::unavailable("embedding store not configured"))?;

        let session_id = Uuid::parse_str(&req.session_id)
            .map_err(|e| Status::invalid_argument(format!("invalid session_id: {e}")))?;

        let deleted = store
            .delete_by_session(&session_id)
            .await
            .map_err(|e| super::error_to_status(e.into()))?;

        Ok(Response::new(DeleteEmbeddingsBySessionResponse { deleted }))
    }
}

fn embedding_record_to_proto(record: crate::store::EmbeddingRecord) -> PbEmbeddingRecord {
    PbEmbeddingRecord {
        id: record.id.to_string(),
        session_id: record
            .session_id
            .map_or_else(String::new, |id| id.to_string()),
        model: record.model,
        vector: record.vector,
        metadata: record.metadata.to_string(),
        created_at: Some(crate::grpc::to_prost_timestamp(record.created_at)),
    }
}
