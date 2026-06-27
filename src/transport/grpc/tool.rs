//! ToolService gRPC implementation.
//!
//! Provides RPCs for listing, querying, invoking, registering, and
//! unregistering tools in the runtime tool registry.

use tonic::{Request, Response, Status};

use crate::transport::grpc::pb::{
    GetToolRequest, GetToolResponse, InvokeToolRequest, InvokeToolResponse, ListToolsRequest,
    ListToolsResponse, RegisterToolRequest, RegisterToolResponse, ToolInfo, UnregisterToolRequest,
    UnregisterToolResponse, tool_service_server::ToolService,
};

use std::sync::Arc;

/// gRPC tool service for runtime tool registry management.
///
/// Supports listing registered tools with their JSON schema,
/// invoking tools by name, and dynamic registration/removal of
/// external tools.
pub struct GrpcToolService {
    state: Arc<super::state::GrpcState>,
}

impl GrpcToolService {
    /// Creates a new tool service backed by the given shared state.
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
            .map_err(|e| super::error_to_status(e.into()))?;

        Ok(Response::new(InvokeToolResponse {
            name: tool.name().to_owned(),
            output: output.value.to_string(),
        }))
    }

    async fn register_tool(
        &self,
        request: Request<RegisterToolRequest>,
    ) -> Result<Response<RegisterToolResponse>, Status> {
        let req = request.into_inner();

        if req.name.is_empty() {
            return Err(Status::invalid_argument("tool name must not be empty"));
        }

        let schema: serde_json::Value =
            serde_json::from_str(&req.parameters_schema).map_err(|e| {
                Status::invalid_argument(format!("invalid parameters schema JSON: {e}"))
            })?;

        let mut tool = crate::tool::ExternalTool::new(&req.name, &req.description, schema);
        if !req.endpoint.is_empty() {
            tool = tool.with_endpoint(&req.endpoint);
        }

        self.state.runtime.tools().register_tool(Arc::new(tool));

        Ok(Response::new(RegisterToolResponse {}))
    }

    async fn unregister_tool(
        &self,
        request: Request<UnregisterToolRequest>,
    ) -> Result<Response<UnregisterToolResponse>, Status> {
        let req = request.into_inner();

        if req.name.is_empty() {
            return Err(Status::invalid_argument("tool name must not be empty"));
        }

        let removed = self.state.runtime.tools().unregister_tool(&req.name);
        if removed.is_none() {
            return Err(Status::not_found(format!("tool '{}' not found", req.name)));
        }

        Ok(Response::new(UnregisterToolResponse {}))
    }
}
