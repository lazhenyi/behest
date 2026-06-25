//! Retrieval-Augmented Generation (RAG) context adapter.
//!
//! Embeds the user message via an [`crate::provider::EmbeddingProvider`] and retrieves
//! semantically relevant context from an [`EmbeddingStore`], then injects
//! the retrieved snippets as a system message into the agent context flow.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::context::{ContextAdapter, ContextInput, ContextResult};
use crate::error::ContextError;
use crate::provider::embedding::EmbeddingRequest;
use crate::provider::{Message, ModelName};
use crate::store::{EmbeddingStore, ScoredEmbedding};

/// Retrieves context snippets via vector search and injects them as a system message.
///
/// The adapter:
/// 1. Embeds `ContextInput::user_message` using the configured embedding provider
/// 2. Queries the embedding store for the top‑k nearest documents
/// 3. Formats retrieved metadata into a single system message
///
/// When `user_message` is [`None`] the adapter produces no messages.
///
/// # Fields
/// - `provider` – embedding provider used to vectorize the user message.
/// - `store` – vector store queried for semantically relevant documents.
/// - `model` – model name passed to the embedding provider.
/// - `limit` – maximum number of retrieved snippets (default 5).
/// - `template` – system-prompt template with `{context}` placeholder.
/// - `metadata_field` – field extracted from each record's metadata (default `"text"`).
pub struct RagContextAdapter {
    provider: Arc<dyn crate::provider::traits::EmbeddingProvider>,
    store: Arc<dyn EmbeddingStore>,
    model: ModelName,
    limit: usize,
    template: String,
    metadata_field: String,
}

impl RagContextAdapter {
    /// Creates a `RagContextAdapter` with a default system-prompt template.
    ///
    /// The default template renders every retrieved snippet's metadata as JSON.
    #[must_use]
    pub fn new(
        provider: Arc<dyn crate::provider::traits::EmbeddingProvider>,
        store: Arc<dyn EmbeddingStore>,
        model: ModelName,
    ) -> Self {
        Self {
            provider,
            store,
            model,
            limit: 5,
            template: String::from(
                "Use the following retrieved context to inform your response:\n\n{context}",
            ),
            metadata_field: String::from("text"),
        }
    }

    /// Sets the maximum number of retrieved snippets (default 5).
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Sets the system-prompt template.
    ///
    /// The placeholder `{context}` will be replaced with the formatted snippets.
    #[must_use]
    pub fn with_template(mut self, template: impl Into<String>) -> Self {
        self.template = template.into();
        self
    }

    /// Sets the metadata field to extract from each [`ScoredEmbedding`] for the
    /// context snippet (default `"text"`).
    ///
    /// When the field is absent the entire metadata object is serialized.
    #[must_use]
    pub fn with_metadata_field(mut self, field: impl Into<String>) -> Self {
        self.metadata_field = field.into();
        self
    }

    fn format_context(&self, results: &[ScoredEmbedding]) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(results.len());
        for result in results {
            let snippet = match result.record.metadata.get(&self.metadata_field) {
                Some(Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => result.record.metadata.to_string(),
            };
            parts.push(format!("[score: {:.4}] {snippet}", result.score));
        }
        parts.join("\n\n")
    }
}

#[async_trait]
impl ContextAdapter for RagContextAdapter {
    fn name(&self) -> &'static str {
        "rag"
    }

    async fn produce(&self, input: &ContextInput) -> ContextResult<Vec<Message>> {
        let user_text = match &input.user_message {
            Some(text) if !text.is_empty() => text.as_str(),
            _ => return Ok(Vec::new()),
        };

        let request = EmbeddingRequest::from_text(self.model.clone(), user_text);

        let response =
            self.provider
                .embed(request)
                .await
                .map_err(|e| ContextError::AdapterFailed {
                    adapter: self.name().to_owned(),
                    message: format!("embedding failed: {e}"),
                })?;

        let vector = response
            .embeddings
            .first()
            .map(|e| e.vector.as_slice())
            .ok_or_else(|| ContextError::AdapterFailed {
                adapter: self.name().to_owned(),
                message: "embedding response contained no vectors".to_owned(),
            })?;

        let results = self.store.search(vector, self.limit).await.map_err(|e| {
            ContextError::AdapterFailed {
                adapter: self.name().to_owned(),
                message: format!("embedding store search failed: {e}"),
            }
        })?;

        if results.is_empty() {
            return Ok(Vec::new());
        }

        let context = self.format_context(&results);
        let message = self.template.replace("{context}", &context);

        Ok(vec![Message::system_text(message)])
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use async_trait::async_trait;
    use serde_json::json;

    use crate::error::ProviderError;
    use crate::provider::embedding::{Embedding, EmbeddingResponse};
    use crate::provider::traits::EmbeddingProvider;
    use crate::provider::{ProviderCapabilities, ProviderId, TokenUsage};
    use crate::store::EmbeddingRecord;
    use crate::store::memory::MemoryEmbeddingStore;

    struct StubEmbeddingProvider {
        vector: Vec<f32>,
    }

    #[async_trait]
    impl EmbeddingProvider for StubEmbeddingProvider {
        fn id(&self) -> ProviderId {
            ProviderId::new("stub")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::embeddings()
        }

        async fn embed(
            &self,
            _request: EmbeddingRequest,
        ) -> Result<EmbeddingResponse, ProviderError> {
            Ok(EmbeddingResponse {
                provider: self.id(),
                model: ModelName::new("stub-model"),
                embeddings: vec![Embedding::new(0, self.vector.clone())],
                usage: Some(TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    total_tokens: 0,
                }),
                raw: None,
            })
        }
    }

    fn make_record(_prefix: &str, text: &str, vector: Vec<f32>) -> EmbeddingRecord {
        let mut record = EmbeddingRecord::new("test-model", vector);
        record.id = uuid::Uuid::new_v4();
        record.metadata = json!({"text": text});
        record
    }

    #[tokio::test]
    async fn rag_adapter_should_retrieve_and_format_context() {
        let store = Arc::new(MemoryEmbeddingStore::new());
        let provider = Arc::new(StubEmbeddingProvider {
            vector: vec![1.0, 0.0, 0.0],
        });

        store
            .upsert(make_record(
                "a",
                "Paris is the capital of France.",
                vec![1.0, 0.0, 0.0],
            ))
            .await
            .unwrap();
        store
            .upsert(make_record(
                "b",
                "Tokyo is the capital of Japan.",
                vec![0.9, 0.1, 0.0],
            ))
            .await
            .unwrap();
        store
            .upsert(make_record(
                "c",
                "Random unrelated content.",
                vec![0.0, 0.0, 1.0],
            ))
            .await
            .unwrap();

        let adapter = RagContextAdapter::new(provider, store, ModelName::new("stub")).with_limit(2);

        let input = ContextInput {
            user_message: Some("What is the capital of France?".to_owned()),
            session_id: None,
            metadata: Value::Null,
        };

        let messages = adapter.produce(&input).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert!(matches!(&messages[0], Message::System { .. }));

        match &messages[0] {
            Message::System { content, .. } => {
                let text = content
                    .iter()
                    .filter_map(|p| match p {
                        crate::provider::ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                assert!(text.contains("Paris"));
                assert!(text.contains("France"));
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn rag_adapter_should_skip_when_no_user_message() {
        let store = Arc::new(MemoryEmbeddingStore::new());
        let provider = Arc::new(StubEmbeddingProvider {
            vector: vec![1.0, 0.0],
        });

        let adapter = RagContextAdapter::new(provider, store, ModelName::new("stub"));

        let input = ContextInput {
            user_message: None,
            session_id: None,
            metadata: Value::Null,
        };

        let messages = adapter.produce(&input).await.unwrap();
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn rag_adapter_should_handle_empty_results() {
        let store = Arc::new(MemoryEmbeddingStore::new());
        let provider = Arc::new(StubEmbeddingProvider {
            vector: vec![1.0, 0.0],
        });

        let adapter = RagContextAdapter::new(provider, store, ModelName::new("stub"));

        let input = ContextInput {
            user_message: Some("What is the capital of France?".to_owned()),
            session_id: None,
            metadata: Value::Null,
        };

        let messages = adapter.produce(&input).await.unwrap();
        assert!(messages.is_empty());
    }

    #[test]
    fn format_context_should_handle_missing_metadata_field() {
        let results = vec![ScoredEmbedding {
            score: 0.95,
            record: {
                let mut record = EmbeddingRecord::new("m", vec![1.0]);
                record.metadata = json!({"other": "value"});
                record
            },
        }];

        let adapter = RagContextAdapter::new(
            Arc::new(StubEmbeddingProvider { vector: vec![1.0] }),
            Arc::new(MemoryEmbeddingStore::new()),
            ModelName::new("stub"),
        );

        let formatted = adapter.format_context(&results);
        assert!(formatted.contains("0.9500"));
    }

    #[test]
    fn format_context_should_extract_named_field() {
        let results = vec![ScoredEmbedding {
            score: 0.88,
            record: {
                let mut record = EmbeddingRecord::new("m", vec![1.0]);
                record.metadata = json!({"text": "hello world"});
                record
            },
        }];

        let adapter = RagContextAdapter::new(
            Arc::new(StubEmbeddingProvider { vector: vec![1.0] }),
            Arc::new(MemoryEmbeddingStore::new()),
            ModelName::new("stub"),
        );

        let formatted = adapter.format_context(&results);
        assert!(formatted.contains("hello world"));
        assert!(formatted.contains("0.8800"));
    }
}
