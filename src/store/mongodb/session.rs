//! MongoDB session store using document-per-session model.
#![allow(clippy::uninlined_format_args)]

use async_trait::async_trait;
use mongodb::bson::{self, Document, doc};
use mongodb::{Client, Collection};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::StorageError;
use crate::provider::{ContentPart, ModelName, TokenUsage, ToolCall};
use crate::store::{MessageRecord, MessageRole, Session, SessionStore, StoreResult};

/// MongoDB-backed session store using a document-per-session model.
///
/// Messages are stored in a separate collection linked by `session_id`.
pub struct MongodbSessionStore {
    sessions: Collection<SessionDoc>,
    messages: Collection<MessageDoc>,
}

impl MongodbSessionStore {
    /// Creates a MongoDB session store from a client and database name.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::ConnectionFailed`] when the connection fails.
    pub async fn new(uri: &str, database: &str) -> StoreResult<Self> {
        let client =
            Client::with_uri_str(uri)
                .await
                .map_err(|e| StorageError::ConnectionFailed {
                    backend: "mongodb".to_owned(),
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;

        let db = client.database(database);
        Ok(Self {
            sessions: db.collection("sessions"),
            messages: db.collection("messages"),
        })
    }

    /// Creates a MongoDB session store from existing collections.
    #[must_use]
    pub fn from_collections(
        sessions: Collection<SessionDoc>,
        messages: Collection<MessageDoc>,
    ) -> Self {
        Self { sessions, messages }
    }
}

/// MongoDB session document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDoc {
    /// Session identifier.
    #[serde(rename = "_id")]
    pub id: String,
    /// Session title.
    pub title: String,
    /// Model name.
    pub model: String,
    /// Metadata as a BSON document.
    pub metadata: Document,
    /// Creation timestamp.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last update timestamp.
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl SessionDoc {
    /// Converts a [`Session`] into a [`SessionDoc`] for MongoDB storage.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::SerializationFailed`] if metadata cannot be
    /// serialized to a BSON document.
    fn try_from_session(s: &Session) -> StoreResult<Self> {
        let metadata =
            bson::to_document(&s.metadata).map_err(|e| StorageError::SerializationFailed {
                message: format!("session metadata BSON serialization: {}", e),
                source: Some(Box::new(e)),
            })?;
        Ok(Self {
            id: s.id.to_string(),
            title: s.title.clone(),
            model: s.model.as_str().to_owned(),
            metadata,
            created_at: s.created_at,
            updated_at: s.updated_at,
        })
    }

    /// Converts a [`SessionDoc`] into a [`Session`].
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::DataCorruption`] if the stored UUID or metadata
    /// cannot be parsed.
    fn try_into_session(self) -> StoreResult<Session> {
        let metadata = bson::from_document::<serde_json::Value>(self.metadata).map_err(|e| {
            StorageError::DataCorruption {
                field: "session.metadata".into(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            }
        })?;
        Ok(Session {
            id: crate::store::util::parse_uuid(&self.id, "session.id")?,
            title: self.title,
            model: ModelName::new(&self.model),
            created_at: self.created_at,
            updated_at: self.updated_at,
            metadata,
        })
    }
}

/// MongoDB message document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDoc {
    /// Message identifier.
    #[serde(rename = "_id")]
    pub id: String,
    /// Session identifier.
    pub session_id: String,
    /// Message role.
    pub role: String,
    /// Content parts.
    pub content: Vec<ContentPart>,
    /// Tool calls.
    pub tool_calls: Vec<ToolCall>,
    /// Token usage.
    pub usage: Option<TokenUsage>,
    /// Creation timestamp.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl MessageDoc {
    /// Converts a [`MessageRecord`] into a [`MessageDoc`] for MongoDB storage.
    fn from_message(m: &MessageRecord) -> Self {
        Self {
            id: m.id.to_string(),
            session_id: m.session_id.to_string(),
            role: match m.role {
                MessageRole::System => "system".to_owned(),
                MessageRole::User => "user".to_owned(),
                MessageRole::Assistant => "assistant".to_owned(),
                MessageRole::Tool => "tool".to_owned(),
            },
            content: m.content.clone(),
            tool_calls: m.tool_calls.clone(),
            usage: m.usage,
            created_at: m.created_at,
        }
    }

    /// Converts a [`MessageDoc`] into a [`MessageRecord`].
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::DataCorruption`] if stored UUIDs cannot be parsed.
    fn try_into_message(self) -> StoreResult<MessageRecord> {
        let role = match self.role.as_str() {
            "system" => MessageRole::System,
            "assistant" => MessageRole::Assistant,
            "tool" => MessageRole::Tool,
            _ => MessageRole::User,
        };
        Ok(MessageRecord {
            id: crate::store::util::parse_uuid(&self.id, "message.id")?,
            session_id: crate::store::util::parse_uuid(&self.session_id, "message.session_id")?,
            role,
            content: self.content,
            tool_calls: self.tool_calls,
            tool_call_id: None,
            tool_name: None,
            usage: self.usage,
            created_at: self.created_at,
        })
    }
}

#[async_trait]
impl SessionStore for MongodbSessionStore {
    async fn create_session(&self, session: Session) -> StoreResult<Session> {
        let doc = SessionDoc::try_from_session(&session)?;
        self.sessions
            .insert_one(doc)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mongodb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;
        Ok(session)
    }

    async fn list_sessions(&self) -> StoreResult<Vec<Session>> {
        use futures_util::TryStreamExt;

        let cursor = self
            .sessions
            .find(doc! {})
            .sort(doc! { "updated_at": -1 })
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mongodb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let docs: Vec<SessionDoc> =
            cursor
                .try_collect()
                .await
                .map_err(|e| StorageError::BackendError {
                    backend: "mongodb".to_owned(),
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;

        docs.into_iter().map(SessionDoc::try_into_session).collect()
    }

    async fn get_session(&self, id: &Uuid) -> StoreResult<Option<Session>> {
        let doc = self
            .sessions
            .find_one(doc! { "_id": id.to_string() })
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mongodb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        doc.map(SessionDoc::try_into_session).transpose()
    }

    async fn delete_session(&self, id: &Uuid) -> StoreResult<()> {
        let id_str = id.to_string();
        self.sessions
            .delete_one(doc! { "_id": &id_str })
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mongodb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;
        self.messages
            .delete_many(doc! { "session_id": &id_str })
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mongodb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;
        Ok(())
    }

    async fn append_message(&self, message: MessageRecord) -> StoreResult<MessageRecord> {
        // Verify session exists
        let session_exists = self
            .sessions
            .find_one(doc! { "_id": message.session_id.to_string() })
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mongodb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?
            .is_some();

        if !session_exists {
            return Err(StorageError::NotFound {
                id: message.session_id.to_string(),
            });
        }

        let doc = MessageDoc::from_message(&message);
        self.messages
            .insert_one(doc)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mongodb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let now = bson::DateTime::now();
        self.sessions
            .update_one(
                doc! { "_id": message.session_id.to_string() },
                doc! { "$set": { "updated_at": now } },
            )
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mongodb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        Ok(message)
    }

    async fn list_messages(&self, session_id: &Uuid) -> StoreResult<Vec<MessageRecord>> {
        use futures_util::TryStreamExt;

        let cursor = self
            .messages
            .find(doc! { "session_id": session_id.to_string() })
            .sort(doc! { "created_at": 1 })
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mongodb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let docs: Vec<MessageDoc> =
            cursor
                .try_collect()
                .await
                .map_err(|e| StorageError::BackendError {
                    backend: "mongodb".to_owned(),
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;

        docs.into_iter().map(MessageDoc::try_into_message).collect()
    }

    async fn update_usage(&self, message_id: &Uuid, usage: TokenUsage) -> StoreResult<()> {
        let usage_doc =
            bson::to_document(&usage).map_err(|e| StorageError::SerializationFailed {
                message: format!("usage BSON serialization: {}", e),
                source: Some(Box::new(e)),
            })?;

        let result = self
            .messages
            .update_one(
                doc! { "_id": message_id.to_string() },
                doc! { "$set": { "usage": usage_doc } },
            )
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mongodb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        if result.matched_count == 0 {
            return Err(StorageError::NotFound {
                id: message_id.to_string(),
            });
        }
        Ok(())
    }
}
