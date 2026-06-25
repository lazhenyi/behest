//! Shared gRPC server state.
//!
//! Defines [`GrpcState`] — the shared state container passed to all
//! gRPC service implementations — and the [`ModelCatalogEntry`] type
//! used by the model-service endpoint.

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;

use super::run::RunTaskRegistry;
use crate::agent::AgentRegistry;
#[cfg(any(feature = "openai", feature = "anthropic"))]
use crate::config::ProviderType;
use crate::config::{AgentConfig, ProviderConfig};
use crate::provider::{ModelName, ProviderId};
use crate::runtime::AgentRuntime;

/// A single entry in the runtime model catalog.
///
/// Associates a model name with its provider and capability flags.
/// JSON-serializable for debug display.
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

/// Shared state container passed to every gRPC service implementation.
///
/// Holds the agent runtime, configuration, run-task registry, and
/// agent registry — all behind `Arc` for concurrent access across
/// multiple service instances.
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
    /// Agent registry for dynamic agent management.
    pub agent_registry: Arc<RwLock<AgentRegistry>>,
}

impl GrpcState {
    /// Creates a new gRPC state with the given runtime, config, and task registry.
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
            agent_registry: Arc::new(RwLock::new(AgentRegistry::new())),
        }
    }

    /// Returns the provider config for the given string identifier, if present.
    #[must_use]
    pub fn provider_config(&self, id: &str) -> Option<&ProviderConfig> {
        self.config
            .providers
            .iter()
            .find_map(|(k, v)| if k.as_str() == id { Some(v) } else { None })
    }

    /// Builds the model catalog from the configured providers.
    ///
    /// Returns all default models and explicitly listed models with
    /// their streaming and tool-calling capabilities.
    #[must_use]
    pub fn model_catalog(&self) -> Vec<ModelCatalogEntry> {
        build_model_catalog(&self.config.providers)
    }
}

/// Builds a model catalog from the given provider configuration map.
///
/// Iterates over every provider and collects its default model
/// (if any) and all explicitly configured models into a flat list
/// of [`ModelCatalogEntry`].
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
