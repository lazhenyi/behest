//! Chat provider implementation of the LLM provider trait.

use std::sync::Arc;

use async_trait::async_trait;

use crate::provider::{ChatProvider, ChatRequest, ContentPart, Message, ModelName};
use crate::reasoning::error::ReasoningError;
use crate::reasoning::operator::LlmProvider;

/// Wraps an [`Arc<dyn ChatProvider>`] as an [`LlmProvider`].
pub struct ChatLlmProvider {
    provider: Arc<dyn ChatProvider>,
    model: ModelName,
}

impl ChatLlmProvider {
    /// Creates a new [`ChatLlmProvider`].
    #[must_use]
    pub fn new(provider: Arc<dyn ChatProvider>, model: ModelName) -> Self {
        Self { provider, model }
    }

    /// Returns a reference to the inner provider.
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn ChatProvider> {
        &self.provider
    }

    /// Returns the configured model name.
    #[must_use]
    pub fn model(&self) -> &ModelName {
        &self.model
    }
}

#[async_trait]
impl LlmProvider for ChatLlmProvider {
    async fn complete(&self, prompt: &str, system: Option<&str>) -> Result<String, ReasoningError> {
        let mut request = ChatRequest::new(self.model.clone());
        if let Some(sys) = system {
            request = request.with_message(Message::system_text(sys));
        }
        request = request.with_message(Message::user_text(prompt));

        let response =
            self.provider
                .complete(request)
                .await
                .map_err(|e| ReasoningError::OperatorFailed {
                    operator: "llm_provider".into(),
                    message: e.to_string(),
                })?;

        let text = match &response.message {
            Message::Assistant { content, .. } => content
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };

        Ok(text)
    }
}
