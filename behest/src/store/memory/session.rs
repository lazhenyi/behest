//! In-memory session store backed by `HashMap`s for sessions and messages.

use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::provider::{ModelName, TokenUsage};
use crate::store::{MessageRecord, Session, SessionStore, StoreResult};

/// In-memory session store for testing, development, and prototyping.
///
/// Sessions are stored in `HashMap<Uuid, Session>` and messages in
/// `HashMap<Uuid, Vec<MessageRecord>>`, both protected by `RwLock`.
/// A reverse index `message_id → session_id` enables O(1) message lookups
/// in [`update_usage`](SessionStore::update_usage). Data is lost when the
/// process exits. Implements [`SessionStore`].
#[derive(Default)]
pub struct MemorySessionStore {
    sessions: RwLock<HashMap<Uuid, Session>>,
    messages: RwLock<HashMap<Uuid, Vec<MessageRecord>>>,
    message_index: RwLock<HashMap<Uuid, Uuid>>,
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
        // Clean up the reverse message index for all messages in this session.
        if let Some(records) = self.messages.write().await.remove(id) {
            let mut index = self.message_index.write().await;
            for record in &records {
                index.remove(&record.id);
            }
        }
        Ok(())
    }

    async fn update_session(
        &self,
        id: &Uuid,
        title: &str,
        model: Option<&ModelName>,
    ) -> StoreResult<Session> {
        let mut sessions = self.sessions.write().await;
        let session = sessions
            .get_mut(id)
            .ok_or_else(|| crate::error::StorageError::NotFound { id: id.to_string() })?;
        title.clone_into(&mut session.title);
        session.updated_at = chrono::Utc::now();
        if let Some(m) = model {
            session.model = m.clone();
        }
        Ok(session.clone())
    }

    async fn append_message(&self, message: MessageRecord) -> StoreResult<MessageRecord> {
        let session_id = message.session_id;

        // Update session timestamp first (acquire sessions lock, release)
        {
            let mut sessions = self.sessions.write().await;
            let session = sessions.get_mut(&session_id).ok_or_else(|| {
                crate::error::StorageError::NotFound {
                    id: session_id.to_string(),
                }
            })?;
            session.updated_at = chrono::Utc::now();
        }

        // Append message (acquire messages lock, release)
        self.messages
            .write()
            .await
            .entry(session_id)
            .or_default()
            .push(message.clone());

        // Maintain the message_id → session_id reverse index for O(1) update_usage.
        self.message_index
            .write()
            .await
            .insert(message.id, session_id);

        Ok(message)
    }

    async fn list_messages(&self, session_id: &Uuid) -> StoreResult<Vec<MessageRecord>> {
        let messages = self.messages.read().await;
        Ok(messages.get(session_id).cloned().unwrap_or_default())
    }

    async fn update_usage(&self, message_id: &Uuid, usage: TokenUsage) -> StoreResult<()> {
        // O(1) lookup via the message_id → session_id reverse index.
        let session_id = {
            let index = self.message_index.read().await;
            index
                .get(message_id)
                .copied()
                .ok_or_else(|| crate::error::StorageError::NotFound {
                    id: message_id.to_string(),
                })?
        };

        let mut messages = self.messages.write().await;
        if let Some(records) = messages.get_mut(&session_id) {
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
