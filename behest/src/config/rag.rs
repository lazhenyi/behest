//! RAG (Retrieval-Augmented Generation) configuration.

use serde::{Deserialize, Serialize};

use crate::provider::{ModelName, ProviderId};

/// Configuration for the RAG (Retrieval-Augmented Generation) context adapter.
///
/// The adapter retrieves relevant snippets from an embedding store and injects
/// them into the system prompt using the configured `template`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Provider ID used for generating embedding vectors.
    pub provider_id: ProviderId,

    /// Embedding model name (e.g. `"text-embedding-3-small"`).
    pub model: ModelName,

    /// Maximum number of retrieved snippets to inject. Default: 5.
    #[serde(default = "default_rag_limit")]
    pub limit: usize,

    /// System-prompt template with `{context}` placeholder for snippet injection.
    /// Default: `"Use the following retrieved context to inform your response:\n\n{context}"`.
    #[serde(default = "default_rag_template")]
    pub template: String,

    /// Metadata field name to extract from each retrieved record. Default: `"text"`.
    #[serde(default = "default_rag_metadata_field")]
    pub metadata_field: String,
}

const fn default_rag_limit() -> usize {
    5
}

fn default_rag_template() -> String {
    String::from("Use the following retrieved context to inform your response:\n\n{context}")
}

fn default_rag_metadata_field() -> String {
    String::from("text")
}
