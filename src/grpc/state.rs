//! Shared gRPC server state.

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::ProviderConfig;
use crate::provider::{ModelName, ProviderId};
use crate::runtime::AgentRuntime;
use crate::tool::ToolRegistry;

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

/// Provider config lookup helpers.
#[derive(Debug, Clone)]
pub struct ProviderConfigMap {
    inner: HashMap<ProviderId, ProviderConfig>,
}

impl ProviderConfigMap {
    /// Creates a new provider config map.
    #[must_use]
    pub fn new(inner: HashMap<ProviderId, ProviderConfig>) -> Self {
        Self { inner }
    }

    /// Returns a provider config by identifier.
    #[must_use]
    pub fn get(&self, id: &ProviderId) -> Option<&ProviderConfig> {
        self.inner.get(id)
    }

    /// Returns a provider config by string id.
    #[must_use]
    pub fn get_by_string(&self, id: &str) -> Option<&ProviderConfig> {
        self.inner
            .iter()
            .find_map(|(k, v)| if k.as_str() == id { Some(v) } else { None })
    }

    /// Iterator over provider configs.
    pub fn iter(&self) -> impl Iterator<Item = (&ProviderId, &ProviderConfig)> {
        self.inner.iter()
    }

    /// Returns true when no providers are configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Shared state for all gRPC services.
pub struct GrpcState {
    /// Agent runtime kernel.
    pub runtime: Arc<AgentRuntime>,
    /// Model catalog derived from provider configs.
    pub model_catalog: Vec<ModelCatalogEntry>,
    /// Provider configuration map.
    pub provider_configs: ProviderConfigMap,
    /// Tool registry clone for read-only tool queries.
    pub tool_registry: ToolRegistry,
}

impl GrpcState {
    /// Creates a new gRPC state.
    #[must_use]
    pub fn new(
        runtime: Arc<AgentRuntime>,
        model_catalog: Vec<ModelCatalogEntry>,
        provider_configs: ProviderConfigMap,
        tool_registry: ToolRegistry,
    ) -> Self {
        Self {
            runtime,
            model_catalog,
            provider_configs,
            tool_registry,
        }
    }
}
