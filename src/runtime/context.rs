//! Context pipeline for runtime.
//!
//! Wraps the existing `ContextFactory` with runtime-specific adapters
//! for session history, RAG, and token budget management.

use std::sync::Arc;

use uuid::Uuid;

use crate::context::{ContextAdapter, ContextFactory, ContextInput, ContextOutput};
use crate::provider::{ChatRequest, Message, ModelName, ToolSpec};

use super::error::RuntimeResult;
use super::store::RuntimeStore;

/// Runtime context pipeline that composes context from multiple sources.
///
/// The pipeline:
/// 1. Loads session history from the store
/// 2. Invokes registered context adapters (system prompt, RAG, etc.)
/// 3. Applies message limits for token budget management
/// 4. Produces a final `ChatRequest`
pub struct ContextPipeline {
    factory: ContextFactory,
    max_history_messages: usize,
}

impl ContextPipeline {
    /// Creates a new context pipeline with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            factory: ContextFactory::new(),
            max_history_messages: 50,
        }
    }

    /// Creates a context pipeline with an existing context factory.
    #[must_use]
    pub fn with_factory(factory: ContextFactory) -> Self {
        Self {
            factory,
            max_history_messages: 50,
        }
    }

    /// Sets the maximum number of history messages to include.
    #[must_use]
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history_messages = max;
        self
    }

    /// Registers a context adapter.
    pub fn register<A>(&mut self, adapter: A)
    where
        A: ContextAdapter + 'static,
    {
        self.factory.register(adapter);
    }

    /// Registers an already shared context adapter.
    pub fn register_arc(&mut self, adapter: Arc<dyn ContextAdapter>) {
        self.factory.register_arc(adapter);
    }

    /// Builds a chat request from context.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` when context building fails.
    pub async fn build(
        &self,
        store: &RuntimeStore,
        session_id: Uuid,
        model: ModelName,
        user_message: Option<&str>,
        tools: Option<&[ToolSpec]>,
    ) -> RuntimeResult<ChatRequest> {
        let input = ContextInput {
            user_message: user_message.map(str::to_owned),
            session_id: Some(session_id.to_string()),
            metadata: serde_json::Value::Null,
        };

        let mut output = self.factory.build(&input).await.map_err(|e| {
            super::error::RuntimeError::Context(crate::error::ContextError::AdapterFailed {
                adapter: "pipeline".to_owned(),
                message: e.to_string(),
            })
        })?;

        let history = store.list_messages(session_id).await?;
        let history = trim_history(&history, self.max_history_messages);
        output.extend(history);

        if let Some(text) = user_message {
            output.extend([Message::user_text(text)]);
        }

        let request = match tools {
            Some(specs) => output.into_request_with_tools(model, specs),
            None => output.into_request(model),
        };

        Ok(request)
    }

    /// Builds context output without creating a request.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` when context building fails.
    pub async fn build_context(
        &self,
        store: &RuntimeStore,
        session_id: Uuid,
        user_message: Option<&str>,
    ) -> RuntimeResult<ContextOutput> {
        let input = ContextInput {
            user_message: user_message.map(str::to_owned),
            session_id: Some(session_id.to_string()),
            metadata: serde_json::Value::Null,
        };

        let mut output = self.factory.build(&input).await.map_err(|e| {
            super::error::RuntimeError::Context(crate::error::ContextError::AdapterFailed {
                adapter: "pipeline".to_owned(),
                message: e.to_string(),
            })
        })?;

        let history = store.list_messages(session_id).await?;
        let history = trim_history(&history, self.max_history_messages);
        output.extend(history);

        if let Some(text) = user_message {
            output.extend([Message::user_text(text)]);
        }

        Ok(output)
    }
}

impl Default for ContextPipeline {
    fn default() -> Self {
        Self::new()
    }
}

fn trim_history(messages: &[Message], max: usize) -> Vec<Message> {
    if messages.len() <= max {
        messages.to_vec()
    } else {
        let skip = messages.len() - max;
        let mut result = Vec::with_capacity(max + 1);

        if let Some(first) = messages.first() {
            if matches!(first, Message::System { .. }) && skip > 0 {
                result.push(first.clone());
            }
        }

        result.extend(messages.iter().skip(skip).cloned());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::StaticAdapter;
    use crate::provider::ContentPart;
    use crate::runtime::memory::MemoryRunStore;
    use crate::store::memory::{MemoryExecutionStore, MemorySessionStore};
    use crate::store::{MessageRecord, MessageRole, Session};

    fn make_store() -> RuntimeStore {
        let sessions = MemorySessionStore::new();
        let executions = MemoryExecutionStore::new();
        let runs = MemoryRunStore::new();
        RuntimeStore::new(Box::new(sessions), Box::new(executions), Box::new(runs))
    }

    #[tokio::test]
    async fn pipeline_should_compose_system_and_history() {
        let store = make_store();

        let session = Session::new("Test", ModelName::new("gpt-4"));
        store
            .sessions()
            .create_session(session.clone())
            .await
            .unwrap();

        let user_msg = MessageRecord::new(
            session.id,
            MessageRole::User,
            vec![ContentPart::text("Hello")],
        );
        store.sessions().append_message(user_msg).await.unwrap();

        let mut pipeline = ContextPipeline::new();
        pipeline.register(StaticAdapter::system("You are helpful."));

        let request = pipeline
            .build(
                &store,
                session.id,
                ModelName::new("gpt-4"),
                Some("How are you?"),
                None,
            )
            .await
            .unwrap();

        assert_eq!(request.messages.len(), 3);
        assert!(matches!(request.messages[0], Message::System { .. }));
        assert!(matches!(request.messages[1], Message::User { .. }));
        assert!(matches!(request.messages[2], Message::User { .. }));
    }

    #[tokio::test]
    async fn pipeline_should_trim_old_history() {
        let store = make_store();

        let session = Session::new("Test", ModelName::new("gpt-4"));
        store
            .sessions()
            .create_session(session.clone())
            .await
            .unwrap();

        for i in 0..10 {
            let msg = MessageRecord::new(
                session.id,
                MessageRole::User,
                vec![ContentPart::text(format!("Message {i}"))],
            );
            store.sessions().append_message(msg).await.unwrap();
        }

        let pipeline = ContextPipeline::new().with_max_history(5);

        let request = pipeline
            .build(&store, session.id, ModelName::new("gpt-4"), None, None)
            .await
            .unwrap();

        assert_eq!(request.messages.len(), 5);
    }

    #[tokio::test]
    async fn pipeline_should_preserve_system_on_trim() {
        let store = make_store();

        let session = Session::new("Test", ModelName::new("gpt-4"));
        store
            .sessions()
            .create_session(session.clone())
            .await
            .unwrap();

        let sys = MessageRecord::new(
            session.id,
            MessageRole::System,
            vec![ContentPart::text("System")],
        );
        store.sessions().append_message(sys).await.unwrap();

        for i in 0..10 {
            let msg = MessageRecord::new(
                session.id,
                MessageRole::User,
                vec![ContentPart::text(format!("Message {i}"))],
            );
            store.sessions().append_message(msg).await.unwrap();
        }

        let pipeline = ContextPipeline::new().with_max_history(5);

        let request = pipeline
            .build(&store, session.id, ModelName::new("gpt-4"), None, None)
            .await
            .unwrap();

        assert!(matches!(request.messages[0], Message::System { .. }));
    }

    #[test]
    fn trim_should_return_all_when_under_limit() {
        let messages = vec![Message::user_text("a"), Message::user_text("b")];
        let result = trim_history(&messages, 10);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn trim_should_skip_oldest_when_over_limit() {
        let messages: Vec<Message> = (0..10)
            .map(|i| Message::user_text(format!("msg{i}")))
            .collect();
        let result = trim_history(&messages, 3);
        assert_eq!(result.len(), 3);
    }
}
