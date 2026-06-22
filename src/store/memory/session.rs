//! In-memory session store backed by `HashMap`.

use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::provider::TokenUsage;
use crate::store::{MessageRecord, Session, SessionStore, StoreResult};

/// In-memory session store for testing and development.
///
/// Data is stored in `HashMap`s protected by `RwLock` and is lost
/// when the process exits.
#[derive(Default)]
pub struct MemorySessionStore {
    sessions: RwLock<HashMap<Uuid, Session>>,
    messages: RwLock<HashMap<Uuid, Vec<MessageRecord>>>,
}

impl MemorySessionStore {
    /// Creates an empty in-memory session store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SessionStore for MemorySessionStore {
    async fn create_session(&self, session: Session) -> StoreResult<Session> {
        let mut sessions = self.sessions.write().await;
        let id = session.id;
        sessions.insert(id, session.clone());
        self.messages.write().await.insert(id, Vec::new());
        Ok(session)
    }

    async fn list_sessions(&self) -> StoreResult<Vec<Session>> {
        let sessions = self.sessions.read().await;
        let mut result: Vec<Session> = sessions.values().cloned().collect();
        result.sort_by_key(|s| std::cmp::Reverse(s.updated_at));
        Ok(result)
    }

    async fn get_session(&self, id: &Uuid) -> StoreResult<Option<Session>> {
        let sessions = self.sessions.read().await;
        Ok(sessions.get(id).cloned())
    }

    async fn delete_session(&self, id: &Uuid) -> StoreResult<()> {
        self.sessions.write().await.remove(id);
        self.messages.write().await.remove(id);
        Ok(())
    }

    async fn append_message(&self, message: MessageRecord) -> StoreResult<MessageRecord> {
        let session_id = message.session_id;

        // Verify session exists
        {
            let sessions = self.sessions.read().await;
            if !sessions.contains_key(&session_id) {
                return Err(crate::error::StorageError::NotFound {
                    id: session_id.to_string(),
                });
            }
        }

        let mut messages = self.messages.write().await;
        messages
            .entry(session_id)
            .or_default()
            .push(message.clone());

        // Update session's updated_at timestamp
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(&session_id) {
            session.updated_at = chrono::Utc::now();
        }

        Ok(message)
    }

    async fn list_messages(&self, session_id: &Uuid) -> StoreResult<Vec<MessageRecord>> {
        let messages = self.messages.read().await;
        Ok(messages.get(session_id).cloned().unwrap_or_default())
    }

    async fn update_usage(&self, message_id: &Uuid, usage: TokenUsage) -> StoreResult<()> {
        let mut messages = self.messages.write().await;
        for records in messages.values_mut() {
            for record in records.iter_mut() {
                if record.id == *message_id {
                    record.usage = Some(usage);
                    return Ok(());
                }
            }
        }
        Err(crate::error::StorageError::NotFound {
            id: message_id.to_string(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::provider::{ContentPart, ModelName};

    fn test_session() -> Session {
        Session::new("Test Chat", ModelName::new("gpt-4"))
    }

    #[tokio::test]
    async fn memory_session_store_should_create_and_get_session() {
        let store = MemorySessionStore::new();
        let session = test_session();
        let id = session.id;

        store.create_session(session).await.unwrap();
        let loaded = store.get_session(&id).await.unwrap();

        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().title, "Test Chat");
    }

    #[tokio::test]
    async fn memory_session_store_should_list_sessions_by_updated_at() {
        let store = MemorySessionStore::new();

        let s1 = test_session();
        let s2 = test_session();
        store.create_session(s1).await.unwrap();
        store.create_session(s2).await.unwrap();

        let sessions = store.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 2);
    }

    #[tokio::test]
    async fn memory_session_store_should_delete_session_and_messages() {
        let store = MemorySessionStore::new();
        let session = test_session();
        let id = session.id;

        store.create_session(session).await.unwrap();
        store.delete_session(&id).await.unwrap();

        assert!(store.get_session(&id).await.unwrap().is_none());
        assert!(store.list_messages(&id).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn memory_session_store_should_append_and_list_messages() {
        let store = MemorySessionStore::new();
        let session = test_session();
        let session_id = session.id;
        store.create_session(session).await.unwrap();

        let msg1 = MessageRecord::new(
            session_id,
            crate::store::MessageRole::User,
            vec![ContentPart::text("Hello")],
        );
        let msg2 = MessageRecord::new(
            session_id,
            crate::store::MessageRole::Assistant,
            vec![ContentPart::text("Hi there!")],
        );

        store.append_message(msg1).await.unwrap();
        store.append_message(msg2).await.unwrap();

        let messages = store.list_messages(&session_id).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, crate::store::MessageRole::User);
        assert_eq!(messages[1].role, crate::store::MessageRole::Assistant);
    }

    #[tokio::test]
    async fn memory_session_store_should_reject_message_for_missing_session() {
        let store = MemorySessionStore::new();
        let msg = MessageRecord::new(
            Uuid::now_v7(),
            crate::store::MessageRole::User,
            vec![ContentPart::text("Hello")],
        );

        let result = store.append_message(msg).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn memory_session_store_should_update_usage() {
        let store = MemorySessionStore::new();
        let session = test_session();
        let session_id = session.id;
        store.create_session(session).await.unwrap();

        let msg = MessageRecord::new(
            session_id,
            crate::store::MessageRole::Assistant,
            vec![ContentPart::text("response")],
        );
        let msg = store.append_message(msg).await.unwrap();

        let usage = TokenUsage::new(10, 20);
        store.update_usage(&msg.id, usage).await.unwrap();

        let messages = store.list_messages(&session_id).await.unwrap();
        assert_eq!(messages[0].usage.unwrap().input_tokens, 10);
        assert_eq!(messages[0].usage.unwrap().output_tokens, 20);
    }

    #[tokio::test]
    async fn memory_session_store_should_return_not_found_for_unknown_usage() {
        let store = MemorySessionStore::new();
        let result = store
            .update_usage(&Uuid::now_v7(), TokenUsage::new(1, 1))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn memory_session_store_should_return_none_for_unknown_session() {
        let store = MemorySessionStore::new();
        let result = store.get_session(&Uuid::now_v7()).await.unwrap();
        assert!(result.is_none());
    }
}
