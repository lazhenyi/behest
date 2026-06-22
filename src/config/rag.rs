//! RAG (Retrieval-Augmented Generation) configuration.

use serde::{Deserialize, Serialize};

use crate::provider::{ModelName, ProviderId};

/// Configuration for the RAG context adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Provider ID used for embedding queries.
    pub provider_id: ProviderId,

    /// Embedding model name.
    pub model: ModelName,

    /// Maximum number of retrieved snippets.
    #[serde(default = "default_rag_limit")]
    pub limit: usize,

    /// System-prompt template with `{context}` placeholder.
    #[serde(default = "default_rag_template")]
    pub template: String,

    /// Metadata field to extract from each retrieved record.
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
