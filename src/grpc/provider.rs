//! ProviderService and ModelService gRPC implementation.

use tonic::{Request, Response, Status};

use crate::grpc::pb::{
    GetProviderRequest, GetProviderResponse, ListModelsRequest, ListModelsResponse,
    ListProvidersRequest, ListProvidersResponse, ModelEntry, ProviderInfo,
    model_service_server::ModelService, provider_service_server::ProviderService,
};

use super::pb::{ModelName, ProviderId};

/// gRPC provider service backed by [`crate::config::ProviderConfig`].
pub struct GrpcProviderService {
    state: std::sync::Arc<super::state::GrpcState>,
}

impl GrpcProviderService {
    /// Creates a new provider service.
    #[must_use]
    pub fn new(state: std::sync::Arc<super::state::GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl ProviderService for GrpcProviderService {
    async fn list_providers(
        &self,
        _request: Request<ListProvidersRequest>,
    ) -> Result<Response<ListProvidersResponse>, Status> {
        let providers: Vec<ProviderInfo> = self
            .state
            .config
            .providers
            .iter()
            .map(|(id, cfg)| {
                let models = cfg
                    .models
                    .iter()
                    .map(|m| ModelName {
                        value: m.as_str().to_owned(),
                    })
                    .collect();
                ProviderInfo {
                    id: Some(ProviderId {
                        value: id.to_string(),
                    }),
                    provider_type: cfg
                        .provider_type
                        .as_ref()
                        .map_or_else(String::new, |t| format!("{t:?}")),
                    default_model: cfg.model.as_ref().map(|m| ModelName {
                        value: m.as_str().to_owned(),
                    }),
                    models,
                }
            })
            .collect();

        Ok(Response::new(ListProvidersResponse { providers }))
    }

    async fn get_provider(
        &self,
        request: Request<GetProviderRequest>,
    ) -> Result<Response<GetProviderResponse>, Status> {
        let req = request.into_inner();
        let Some(req_id) = req.id else {
            return Err(Status::invalid_argument("id is required"));
        };

        let Some(cfg) = self.state.provider_config(&req_id.value) else {
            return Err(Status::not_found(format!(
                "provider '{req_id_value}' not found",
                req_id_value = req_id.value
            )));
        };

        let models = cfg
            .models
            .iter()
            .map(|m| ModelName {
                value: m.as_str().to_owned(),
            })
            .collect();
        let info = ProviderInfo {
            id: Some(ProviderId {
                value: req_id.value.clone(),
            }),
            provider_type: cfg
                .provider_type
                .as_ref()
                .map_or_else(String::new, |t| format!("{t:?}")),
            default_model: cfg.model.as_ref().map(|m| ModelName {
                value: m.as_str().to_owned(),
            }),
            models,
        };

        Ok(Response::new(GetProviderResponse {
            provider: Some(info),
        }))
    }
}

/// gRPC model service backed by the model catalog.
pub struct GrpcModelService {
    state: std::sync::Arc<super::state::GrpcState>,
}

impl GrpcModelService {
    /// Creates a new model service.
    #[must_use]
    pub fn new(state: std::sync::Arc<super::state::GrpcState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl ModelService for GrpcModelService {
    async fn list_models(
        &self,
        _request: Request<ListModelsRequest>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        let models: Vec<ModelEntry> = self
            .state
            .model_catalog()
            .iter()
            .map(|m| ModelEntry {
                provider: Some(ProviderId {
                    value: m.provider.to_string(),
                }),
                model: Some(ModelName {
                    value: m.model.as_str().to_owned(),
                }),
                streaming: m.streaming,
                tool_calling: m.tool_calling,
            })
            .collect();

        Ok(Response::new(ListModelsResponse { models }))
    }
}
