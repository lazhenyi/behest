//! Shared gRPC server state.

use std::sync::Arc;
use std::time::Instant;

use super::run::RunTaskRegistry;
#[cfg(any(feature = "openai", feature = "anthropic"))]
use crate::config::ProviderType;
use crate::config::{AgentConfig, ProviderConfig};
use crate::provider::{ModelName, ProviderId};
use crate::runtime::AgentRuntime;

/// JSON encodable model catalog entry.
#[derive(Debug, Clone)]
pub struct ModelCatalogEntry {
    /// Provider this model belongs to.
    pub provider: ProviderId,
    /// Model name.
    pub model: ModelName,
    /// Whether the model supports streaming.
    pub streaming: bool,
    /// Whether the model supports tool calling.
    pub tool_calling: bool,
}

/// Shared state for all gRPC services.
pub struct GrpcState {
    /// Agent runtime kernel.
    pub runtime: Arc<AgentRuntime>,
    /// Agent configuration (provider configs, store config, etc.).
    pub config: Arc<AgentConfig>,
    /// Server start time for uptime metrics.
    pub started_at: Instant,
    /// Active run task registry for cancellation and metrics.
    pub run_tasks: Arc<RunTaskRegistry>,
    /// Maximum number of concurrent runs (None = unlimited).
    pub max_concurrent_runs: Option<usize>,
}

impl GrpcState {
    /// Creates a new gRPC state.
    #[must_use]
    pub fn new(
        runtime: Arc<AgentRuntime>,
        config: Arc<AgentConfig>,
        run_tasks: Arc<RunTaskRegistry>,
        max_concurrent_runs: Option<usize>,
    ) -> Self {
        Self {
            runtime,
            config,
            started_at: Instant::now(),
            run_tasks,
            max_concurrent_runs,
        }
    }

    /// Returns a provider config by string identifier.
    #[must_use]
    pub fn provider_config(&self, id: &str) -> Option<&ProviderConfig> {
        self.config
            .providers
            .iter()
            .find_map(|(k, v)| if k.as_str() == id { Some(v) } else { None })
    }

    /// Builds the model catalog from provider configs.
    #[must_use]
    pub fn model_catalog(&self) -> Vec<ModelCatalogEntry> {
        build_model_catalog(&self.config.providers)
    }
}

/// Builds a model catalog from provider configurations.
pub(crate) fn build_model_catalog(
    providers: &std::collections::HashMap<ProviderId, ProviderConfig>,
) -> Vec<ModelCatalogEntry> {
    let mut catalog = Vec::new();

    for (provider_id, cfg) in providers {
        let has_chat = match cfg.provider_type {
            #[cfg(feature = "openai")]
            Some(ProviderType::OpenAi) => true,
            #[cfg(feature = "anthropic")]
            Some(ProviderType::Anthropic) => true,
            None => false,
            #[allow(unreachable_patterns)]
            _ => false,
        };

        if let Some(ref default_model) = cfg.model {
            catalog.push(ModelCatalogEntry {
                provider: provider_id.clone(),
                model: default_model.clone(),
                streaming: has_chat,
                tool_calling: has_chat,
            });
        }

        for model in &cfg.models {
            catalog.push(ModelCatalogEntry {
                provider: provider_id.clone(),
                model: model.clone(),
                streaming: has_chat,
                tool_calling: has_chat,
            });
        }
    }

    catalog
}
