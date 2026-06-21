//! Provider-neutral embedding request and response types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::provider::{ModelName, ProviderId, TokenUsage};

/// Request for vector embeddings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Provider-specific embedding model name.
    pub model: ModelName,
    /// Inputs to embed.
    pub input: Vec<EmbeddingInput>,
    /// Optional target embedding dimension.
    pub dimensions: Option<u32>,
    /// Application metadata forwarded to provider adapters.
    pub metadata: Value,
}

impl EmbeddingRequest {
    /// Creates an embedding request from explicit inputs.
    #[must_use]
    pub fn new(model: ModelName, input: Vec<EmbeddingInput>) -> Self {
        Self {
            model,
            input,
            dimensions: None,
            metadata: Value::Null,
        }
    }

    /// Creates an embedding request for one text input.
    #[must_use]
    pub fn from_text(model: ModelName, text: impl Into<String>) -> Self {
        Self::new(model, vec![EmbeddingInput::text(text)])
    }

    /// Sets the target embedding dimension.
    #[must_use]
    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Sets adapter metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// A single embedding input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum EmbeddingInput {
    /// Plain text input.
    Text {
        /// Text to embed.
        text: String,
    },
    /// Tokenized input.
    Tokens {
        /// Provider-specific token IDs.
        tokens: Vec<u32>,
    },
}

impl EmbeddingInput {
    /// Creates a text embedding input.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Creates a token embedding input.
    #[must_use]
    pub fn tokens(tokens: Vec<u32>) -> Self {
        Self::Tokens { tokens }
    }
}

/// Response returned by an embedding provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Provider that produced the embeddings.
    pub provider: ProviderId,
    /// Model that produced the embeddings.
    pub model: ModelName,
    /// Embeddings in the same order as request inputs.
    pub embeddings: Vec<Embedding>,
    /// Token accounting, when supplied by the provider.
    pub usage: Option<TokenUsage>,
    /// Raw provider response for adapters that retain it.
    pub raw: Option<Value>,
}

/// One embedding vector and its input index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    /// Input index associated with this vector.
    pub index: usize,
    /// Dense embedding vector.
    pub vector: Vec<f32>,
}

impl Embedding {
    /// Creates one embedding vector.
    #[must_use]
    pub fn new(index: usize, vector: Vec<f32>) -> Self {
        Self { index, vector }
    }
}
