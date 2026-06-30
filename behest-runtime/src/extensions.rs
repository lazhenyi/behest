//! [`Extensions`]: the composable, hot-pluggable facade over every
//! pluggable runtime element.
//!
//! Every category of plug-in — chat providers, embedding providers, tools,
//! context adapters, session stores, execution stores, embedding stores,
//! artifact stores, run stores, event publishers, session data stores,
//! runtime event stores, snapshot stores, and RAG adapters — is exposed
//! here as an [`ExtensionPoint<T>`]. Operators compose a runtime by
//! registering implementations by name, and can hot-swap any
//! registered instance at runtime.
//!
//! # Construction
//!
//! [`Extensions::default`] returns a fresh, empty facade suitable for
//! greenfield construction. Existing `AgentRuntime` instances
//! expose their built-in components via
//! [`AgentRuntime::extensions`](super::agent::AgentRuntime::extensions),
//! which materializes an `Extensions` from the runtime's internal
//! registries on demand.
//!
//! # Threading
//!
//! Every field is an [`ExtensionPoint<T>`], which is internally
//! `Arc`-backed and uses `RwLock` for storage. Cloning `Extensions`
//! is cheap and all clones observe the same set of registered entries.
//!
//! # Examples
//!
//! ```rust
//! use std::sync::Arc;
//! use behest_provider::{ChatProvider, ProviderId, ChatRequest, ChatResponse, ProviderResult};
//! use behest_runtime::extensions::Extensions;
//!
//! let exts = Extensions::default();
//! let _ = exts;
//! ```

#![allow(clippy::pedantic)]

use super::extension::ExtensionPoint;
use super::store::RunStore;
use behest_provider::{ChatProvider, EmbeddingProvider};
use behest_store::{ArtifactStore, EmbeddingStore, ExecutionStore, SessionStore};

#[cfg(feature = "queue")]
use behest_store::EventPublisher;

use super::event_store::RuntimeEventStore;
use super::invocation::SessionDataStore;
use super::snapshot::SnapshotStore;

/// The composable, hot-pluggable facade over every pluggable runtime
/// element.
///
/// All fields are public so callers can register, replace, and inspect
/// every category of plug-in directly. Cloning is cheap (`Arc`-backed
/// internally).
#[derive(Clone, Default)]
pub struct Extensions {
    /// Chat provider implementations, keyed by user-assigned name.
    pub chat_providers: ExtensionPoint<dyn ChatProvider>,
    /// Embedding provider implementations, keyed by user-assigned name.
    pub embedding_providers: ExtensionPoint<dyn EmbeddingProvider>,
    /// Tool implementations, keyed by user-assigned name.
    pub tools: ExtensionPoint<dyn behest_tool::Tool>,
    /// Context adapter implementations, keyed by user-assigned name.
    pub context_adapters: ExtensionPoint<dyn behest_context::ContextAdapter>,
    /// Session store implementations, keyed by user-assigned name.
    pub session_stores: ExtensionPoint<dyn SessionStore>,
    /// Execution store implementations, keyed by user-assigned name.
    pub execution_stores: ExtensionPoint<dyn ExecutionStore>,
    /// Embedding store implementations, keyed by user-assigned name.
    pub embedding_stores: ExtensionPoint<dyn EmbeddingStore>,
    /// Artifact store implementations, keyed by user-assigned name.
    pub artifact_stores: ExtensionPoint<dyn ArtifactStore>,
    /// Run store implementations, keyed by user-assigned name.
    pub run_stores: ExtensionPoint<dyn RunStore>,
    /// Event publisher implementations, keyed by user-assigned name.
    #[cfg(feature = "queue")]
    pub event_publishers: ExtensionPoint<dyn EventPublisher>,
    /// Session data store implementations, keyed by user-assigned name.
    pub session_data_stores: ExtensionPoint<dyn SessionDataStore>,
    /// Runtime event store implementations, keyed by user-assigned name.
    pub runtime_event_stores: ExtensionPoint<dyn RuntimeEventStore>,
    /// Snapshot store implementations, keyed by user-assigned name.
    pub snapshot_stores: ExtensionPoint<dyn SnapshotStore>,
}

impl Extensions {
    /// Construct a fresh, empty `Extensions`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct extension-point categories that have at least
    /// one registered entry.
    #[must_use]
    pub fn populated_categories(&self) -> usize {
        macro_rules! count {
            ($n:ident, $($field:ident),+ $(,)?) => {
                $( if !self.$field.is_empty() { $n += 1; } )+
            };
        }
        let mut n = 0;
        count!(
            n,
            chat_providers,
            embedding_providers,
            tools,
            context_adapters
        );
        count!(
            n,
            session_stores,
            execution_stores,
            embedding_stores,
            artifact_stores,
            run_stores
        );
        #[cfg(feature = "queue")]
        if !self.event_publishers.is_empty() {
            n += 1;
        }
        count!(
            n,
            session_data_stores,
            runtime_event_stores,
            snapshot_stores
        );
        n
    }
}

impl std::fmt::Debug for Extensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extensions")
            .field("chat_providers", &self.chat_providers.names())
            .field("embedding_providers", &self.embedding_providers.names())
            .field("tools", &self.tools.names())
            .field("context_adapters", &self.context_adapters.names())
            .field("session_stores", &self.session_stores.names())
            .field("execution_stores", &self.execution_stores.names())
            .field("embedding_stores", &self.embedding_stores.names())
            .field("artifact_stores", &self.artifact_stores.names())
            .field("run_stores", &self.run_stores.names())
            .field("session_data_stores", &self.session_data_stores.names())
            .field("runtime_event_stores", &self.runtime_event_stores.names())
            .field("snapshot_stores", &self.snapshot_stores.names())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_extensions_reports_zero_populated() {
        let exts = Extensions::new();
        assert_eq!(exts.populated_categories(), 0);
    }

    #[test]
    fn clone_shares_state() {
        let exts = Extensions::new();
        let exts2 = exts.clone();
        // Both clones observe the same underlying state.
        assert_eq!(exts.populated_categories(), exts2.populated_categories());
    }
}
