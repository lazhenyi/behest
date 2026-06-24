//! ToolService gRPC implementation.

use tonic::{Request, Response, Status};

use crate::grpc::pb::{
    GetToolRequest, GetToolResponse, InvokeToolRequest, InvokeToolResponse, ListToolsRequest,
    ListToolsResponse, ToolInfo, tool_service_server::ToolService,
};

use std::sync::Arc;

/// gRPC tool service.
pub struct GrpcToolService {
    state: Arc<super::state::GrpcState>,
}

impl GrpcToolService {
    /// Creates a new tool service.
    #[must_use]
    pub fn new(state: Arc<super::state::GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl ToolService for GrpcToolService {
    async fn list_tools(
        &self,
        _request: Request<ListToolsRequest>,
    ) -> Result<Response<ListToolsResponse>, Status> {
        let tools: Vec<ToolInfo> = self
            .state
            .runtime
            .tools()
            .registry()
            .specs()
            .into_iter()
            .map(|spec| ToolInfo {
                name: spec.name,
                description: spec.description,
                parameters_schema: spec.parameters_schema.to_string(),
            })
            .collect();

        Ok(Response::new(ListToolsResponse { tools }))
    }

    async fn get_tool(
        &self,
        request: Request<GetToolRequest>,
    ) -> Result<Response<GetToolResponse>, Status> {
        let req = request.into_inner();
        let Some(tool) = self.state.runtime.tools().registry().get(&req.name) else {
            return Err(Status::not_found(format!("tool '{}' not found", req.name)));
        };

        let spec = tool.to_spec();
        Ok(Response::new(GetToolResponse {
            tool: Some(ToolInfo {
                name: spec.name,
                description: spec.description,
                parameters_schema: spec.parameters_schema.to_string(),
            }),
        }))
    }

    async fn invoke_tool(
        &self,
        request: Request<InvokeToolRequest>,
    ) -> Result<Response<InvokeToolResponse>, Status> {
        let req = request.into_inner();
        let Some(tool) = self.state.runtime.tools().registry().get(&req.name) else {
            return Err(Status::not_found(format!("tool '{}' not found", req.name)));
        };

        let arguments: serde_json::Value = serde_json::from_str(&req.arguments)
            .map_err(|e| Status::invalid_argument(format!("invalid JSON arguments: {e}")))?;

        let output = tool
            .execute(arguments)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(InvokeToolResponse {
            name: tool.name().to_owned(),
            output: output.value.to_string(),
        }))
    }
}
