//! ContextService gRPC implementation.
//!
//! Provides RPCs for building session context from adapters and
//! listing available context adapters registered in the runtime.

use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::transport::grpc::pb::{
    BuildContextRequest, BuildContextResponse, ListContextAdaptersRequest,
    ListContextAdaptersResponse, context_service_server::ContextService,
};

use std::sync::Arc;

/// gRPC context service for building structured prompts.
///
/// Assembles context from registered adapters (files, tools, etc.)
/// into a message list for a given session.
pub struct GrpcContextService {
    state: Arc<super::state::GrpcState>,
}

impl GrpcContextService {
    /// Creates a new context service backed by the given shared state.
    #[must_use]
    pub fn new(state: Arc<super::state::GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl ContextService for GrpcContextService {
    async fn build_context(
        &self,
        request: Request<BuildContextRequest>,
    ) -> Result<Response<BuildContextResponse>, Status> {
        let req = request.into_inner();

        let session_id = Uuid::parse_str(&req.session_id)
            .map_err(|e| Status::invalid_argument(format!("invalid session_id: {e}")))?;

        let context = self.state.runtime.context();

        let output = context
            .build_context(
                self.state.runtime.store(),
                session_id,
                Some(&req.user_message),
            )
            .await
            .map_err(|e| super::runtime_error_to_status(&e))?;

        let messages: Vec<crate::transport::grpc::pb::Message> = output
            .messages()
            .iter()
            .map(crate::transport::grpc::session::message_to_proto)
            .collect();

        Ok(Response::new(BuildContextResponse { messages }))
    }

    async fn list_context_adapters(
        &self,
        _request: Request<ListContextAdaptersRequest>,
    ) -> Result<Response<ListContextAdaptersResponse>, Status> {
        let context = self.state.runtime.context();
        let adapter_names: Vec<String> = context.adapter_names().map(str::to_owned).collect();

        Ok(Response::new(ListContextAdaptersResponse { adapter_names }))
    }
}
