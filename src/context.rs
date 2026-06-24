//! Context construction and multi-adapter composition.
//!
//! The [`ContextAdapter`] trait defines pluggable context sources such as
//! system prompts, conversation history, memory, or RAG retrieval. The
//! [`ContextFactory`] composes multiple adapters in order to produce
//! a complete [`Vec<Message>`] for chat requests.
//!
//! # Example
//!
//! ```rust,no_run
//! use behest::context::{ContextAdapter, ContextFactory, ContextInput, StaticAdapter};
//! use behest::provider::Message;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut factory = ContextFactory::new();
//!
//!     factory.register(StaticAdapter::system("You are a helpful assistant."));
//!     factory.register(StaticAdapter::user("Hello, how are you?"));
//!
//!     let input = ContextInput::default();
//!     let output = factory.build(&input).await.unwrap();
//!
//!     assert_eq!(output.messages().len(), 2);
//! }
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ContextError;
use crate::provider::{ChatRequest, Message, ModelName, ToolChoice, ToolSpec};
use crate::tool::ToolRegistry;

/// Result type for context operations.
pub type ContextResult<T> = std::result::Result<T, ContextError>;

/// Input provided to context adapters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextInput {
    /// Optional user message to include.
    pub user_message: Option<String>,
    /// Optional session identifier.
    pub session_id: Option<String>,
    /// Application-specific metadata.
    pub metadata: Value,
}

impl ContextInput {
    /// Creates an empty context input.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the user message.
    #[must_use]
    pub fn with_user_message(mut self, message: impl Into<String>) -> Self {
        self.user_message = Some(message.into());
        self
    }

    /// Sets the session identifier.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Sets application metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Output produced by context construction.
#[derive(Debug, Clone, Default)]
pub struct ContextOutput {
    messages: Vec<Message>,
}

impl ContextOutput {
    /// Creates an empty context output.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a context output from messages.
    #[must_use]
    pub fn from_messages(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    /// Returns the composed messages.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Consumes the output and returns the messages.
    #[must_use]
    pub fn into_messages(self) -> Vec<Message> {
        self.messages
    }

    /// Appends messages to the output.
    pub fn extend(&mut self, messages: impl IntoIterator<Item = Message>) {
        self.messages.extend(messages);
    }

    /// Builds a [`ChatRequest`] from this context output.
    #[must_use]
    pub fn into_request(self, model: ModelName) -> ChatRequest {
        ChatRequest {
            model,
            messages: self.messages,
            tools: Vec::new(),
            tool_choice: ToolChoice::default(),
            response_format: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stop: Vec::new(),
            metadata: Value::Null,
        }
    }

    /// Builds a [`ChatRequest`] with tool definitions from a registry.
    #[must_use]
    pub fn into_request_with_tools(self, model: ModelName, tools: &[ToolSpec]) -> ChatRequest {
        ChatRequest {
            model,
            messages: self.messages,
            tools: tools.to_vec(),
            tool_choice: ToolChoice::default(),
            response_format: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stop: Vec::new(),
            metadata: Value::Null,
        }
    }
}

/// Pluggable context source that produces message fragments.
#[async_trait]
pub trait ContextAdapter: Send + Sync {
    /// Returns the adapter name.
    fn name(&self) -> &str;

    /// Produces message fragments for the given input.
    async fn produce(&self, input: &ContextInput) -> ContextResult<Vec<Message>>;
}

/// Static context adapter that returns fixed messages.
pub struct StaticAdapter {
    name: String,
    messages: Vec<Message>,
}

impl StaticAdapter {
    /// Creates a static adapter with a system message.
    #[must_use]
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            name: "system".to_owned(),
            messages: vec![Message::system_text(text)],
        }
    }

    /// Creates a static adapter with a user message.
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            name: "user".to_owned(),
            messages: vec![Message::user_text(text)],
        }
    }

    /// Creates a static adapter with custom messages.
    #[must_use]
    pub fn messages(name: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            name: name.into(),
            messages,
        }
    }
}

#[async_trait]
impl ContextAdapter for StaticAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    async fn produce(&self, _input: &ContextInput) -> ContextResult<Vec<Message>> {
        Ok(self.messages.clone())
    }
}

/// Function-based context adapter.
pub struct FunctionAdapter<F> {
    name: String,
    handler: F,
}

impl<F, Fut> FunctionAdapter<F>
where
    F: Fn(ContextInput) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ContextResult<Vec<Message>>> + Send + 'static,
{
    /// Creates a function-based context adapter.
    #[must_use]
    pub fn new(name: impl Into<String>, handler: F) -> Self {
        Self {
            name: name.into(),
            handler,
        }
    }
}

#[async_trait]
impl<F, Fut> ContextAdapter for FunctionAdapter<F>
where
    F: Fn(ContextInput) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ContextResult<Vec<Message>>> + Send + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    async fn produce(&self, input: &ContextInput) -> ContextResult<Vec<Message>> {
        (self.handler)(input.clone()).await
    }
}

/// Multi-adapter context factory.
///
/// The factory maintains an ordered list of context adapters and composes
/// their output into a single [`ContextOutput`].
#[derive(Clone, Default)]
pub struct ContextFactory {
    adapters: Vec<Arc<dyn ContextAdapter>>,
}

impl ContextFactory {
    /// Creates an empty context factory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a context adapter.
    pub fn register<A>(&mut self, adapter: A)
    where
        A: ContextAdapter + 'static,
    {
        self.adapters.push(Arc::new(adapter));
    }

    /// Registers an already shared context adapter.
    pub fn register_arc(&mut self, adapter: Arc<dyn ContextAdapter>) {
        self.adapters.push(adapter);
    }

    /// Returns the number of registered adapters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    /// Returns `true` when no adapters are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Returns adapter names in registration order.
    pub fn adapter_names(&self) -> impl Iterator<Item = &str> {
        self.adapters.iter().map(|a| a.name())
    }

    /// Builds context output by invoking all adapters in order.
    ///
    /// # Errors
    ///
    /// Returns [`ContextError::AdapterFailed`] when any adapter fails.
    pub async fn build(&self, input: &ContextInput) -> ContextResult<ContextOutput> {
        let mut output = ContextOutput::new();

        for adapter in &self.adapters {
            let messages =
                adapter
                    .produce(input)
                    .await
                    .map_err(|e| ContextError::AdapterFailed {
                        adapter: adapter.name().to_owned(),
                        message: e.to_string(),
                    })?;
            output.extend(messages);
        }

        Ok(output)
    }

    /// Builds a [`ChatRequest`] with context and optional tool definitions.
    ///
    /// # Errors
    ///
    /// Returns [`ContextError::AdapterFailed`] when any adapter fails.
    pub async fn build_request(
        &self,
        input: &ContextInput,
        model: ModelName,
        tool_registry: Option<&ToolRegistry>,
    ) -> ContextResult<ChatRequest> {
        let output = self.build(input).await?;

        let request = if let Some(registry) = tool_registry {
            let specs = registry.specs();
            output.into_request_with_tools(model, &specs)
        } else {
            output.into_request(model)
        };

        Ok(request)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn context_input_should_support_builder_pattern() {
        let input = ContextInput::new()
            .with_user_message("Hello")
            .with_session_id("session_123")
            .with_metadata(json!({"key": "value"}));

        assert_eq!(input.user_message, Some("Hello".to_owned()));
        assert_eq!(input.session_id, Some("session_123".to_owned()));
        assert_eq!(input.metadata, json!({"key": "value"}));
    }

    #[test]
    fn context_output_should_be_empty_when_new() {
        let output = ContextOutput::new();
        assert!(output.messages().is_empty());
    }

    #[test]
    fn context_output_should_extend_messages() {
        let mut output = ContextOutput::new();
        output.extend(vec![
            Message::system_text("System"),
            Message::user_text("User"),
        ]);

        assert_eq!(output.messages().len(), 2);
    }

    #[test]
    fn context_output_should_convert_to_request() {
        let output = ContextOutput::from_messages(vec![
            Message::system_text("System"),
            Message::user_text("User"),
        ]);

        let request = output.into_request(ModelName::new("gpt-4"));

        assert_eq!(request.model.as_str(), "gpt-4");
        assert_eq!(request.messages.len(), 2);
        assert!(request.tools.is_empty());
    }

    #[test]
    fn context_output_should_convert_to_request_with_tools() {
        let output = ContextOutput::from_messages(vec![Message::user_text("Hello")]);
        let tools = vec![ToolSpec::new("echo", "Echo tool", json!({}))];

        let request = output.into_request_with_tools(ModelName::new("gpt-4"), &tools);

        assert_eq!(request.tools.len(), 1);
        assert_eq!(request.tools[0].name, "echo");
    }

    #[test]
    fn context_factory_should_be_empty_when_new() {
        let factory = ContextFactory::new();
        assert!(factory.is_empty());
        assert_eq!(factory.len(), 0);
    }

    #[test]
    fn context_factory_should_register_adapters() {
        let mut factory = ContextFactory::new();
        factory.register(StaticAdapter::system("System prompt"));
        factory.register(StaticAdapter::user("User message"));

        assert_eq!(factory.len(), 2);
    }

    #[test]
    fn context_factory_should_list_adapter_names() {
        let mut factory = ContextFactory::new();
        factory.register(StaticAdapter::system("System"));
        factory.register(StaticAdapter::user("User"));

        let names: Vec<&str> = factory.adapter_names().collect();
        assert_eq!(names, vec!["system", "user"]);
    }

    #[tokio::test]
    async fn context_factory_should_build_output_in_order() {
        let mut factory = ContextFactory::new();
        factory.register(StaticAdapter::system("First"));
        factory.register(StaticAdapter::user("Second"));

        let input = ContextInput::new();
        let output = factory.build(&input).await.unwrap();

        assert_eq!(output.messages().len(), 2);
    }

    #[tokio::test]
    async fn context_factory_should_build_request_with_tools() {
        let mut factory = ContextFactory::new();
        factory.register(StaticAdapter::system("You are helpful."));

        let mut registry = ToolRegistry::new();
        registry.register(crate::tool::FunctionTool::new(
            "echo",
            "Echo",
            json!({}),
            |_: Value| async { Ok(Value::Null) },
        ));

        let input = ContextInput::new().with_user_message("Hello");
        let request = factory
            .build_request(&input, ModelName::new("gpt-4"), Some(&registry))
            .await
            .unwrap();

        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.tools.len(), 1);
    }

    #[tokio::test]
    async fn static_adapter_should_produce_system_message() {
        let adapter = StaticAdapter::system("You are a helpful assistant.");
        let input = ContextInput::new();
        let messages = adapter.produce(&input).await.unwrap();

        assert_eq!(messages.len(), 1);
        assert!(matches!(messages[0], Message::System { .. }));
    }

    #[tokio::test]
    async fn static_adapter_should_produce_user_message() {
        let adapter = StaticAdapter::user("Hello");
        let input = ContextInput::new();
        let messages = adapter.produce(&input).await.unwrap();

        assert_eq!(messages.len(), 1);
        assert!(matches!(messages[0], Message::User { .. }));
    }

    #[tokio::test]
    async fn function_adapter_should_invoke_handler() {
        let adapter = FunctionAdapter::new("custom", |input: ContextInput| async move {
            let msg = input.user_message.unwrap_or_default();
            Ok(vec![Message::user_text(format!("Echo: {msg}"))])
        });

        let input = ContextInput::new().with_user_message("test");
        let messages = adapter.produce(&input).await.unwrap();

        assert_eq!(messages.len(), 1);
    }
}
