//! SurrealDB session store using document model.
#![allow(clippy::uninlined_format_args)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::any::Any;
use uuid::Uuid;

use crate::{CompactionMeta, MessageRecord, MessageRole, Session, SessionStore, StoreResult};
use behest_core::error::StorageError;
use behest_provider::{ContentPart, ModelName, TokenUsage, ToolCall};

/// SurrealDB-backed session store using its document model.
///
/// Sessions are stored in the `sessions` table and messages in the `messages`
/// table, linked by `session_id`. Implements [`SessionStore`].
pub struct SurrealdbSessionStore {
    db: Surreal<Any>,
}

impl SurrealdbSessionStore {
    /// Creates a SurrealDB session store from an existing `Surreal<Any>` connection.
    ///
    /// The connection should already be established and the namespace/database selected.
    #[must_use]
    pub fn new(db: Surreal<Any>) -> Self {
        Self { db }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionRecord {
    title: String,
    model: String,
    metadata: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MessageStoreRecord {
    session_id: String,
    role: String,
    content: Vec<ContentPart>,
    tool_calls: Vec<ToolCall>,
    usage: Option<TokenUsage>,
    #[serde(default)]
    is_compaction: bool,
    #[serde(default)]
    is_summary: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compaction_meta: Option<CompactionMeta>,
    created_at: chrono::DateTime<chrono::Utc>,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn role_str(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

fn parse_role(s: &str) -> MessageRole {
    match s {
        "system" => MessageRole::System,
        "assistant" => MessageRole::Assistant,
        "tool" => MessageRole::Tool,
        _ => MessageRole::User,
    }
}

/// Deserializes a `serde_json::Value` into `T`, returning `DataCorruption` on failure.
fn from_value<T: serde::de::DeserializeOwned>(
    value: serde_json::Value,
    field: impl Into<String>,
) -> StoreResult<T> {
    serde_json::from_value(value).map_err(|e| StorageError::DataCorruption {
        field: field.into(),
        message: e.to_string(),
        source: Some(Box::new(e)),
    })
}

#[async_trait]
impl SessionStore for SurrealdbSessionStore {
    async fn create_session(&self, session: Session) -> StoreResult<Session> {
        let record = SessionRecord {
            title: session.title.clone(),
            model: session.model.as_str().to_owned(),
            metadata: session.metadata.clone(),
            created_at: session.created_at,
            updated_at: session.updated_at,
        };

        let content =
            serde_json::to_value(&record).map_err(|e| StorageError::SerializationFailed {
                message: format!("session serialization: {}", e),
                source: Some(Box::new(e)),
            })?;

        self.db
            .create::<Option<serde_json::Value>>(("sessions", session.id.to_string()))
            .content(content)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "surrealdb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        Ok(session)
    }

    async fn list_sessions(&self) -> StoreResult<Vec<Session>> {
        let mut result = self
            .db
            .query("SELECT * FROM sessions ORDER BY updated_at DESC")
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "surrealdb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let rows: Vec<serde_json::Value> =
            result.take(0).map_err(|e| StorageError::BackendError {
                backend: "surrealdb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        rows.into_iter()
            .map(|row| {
                // SurrealDB returns rows as { id: "...", ...fields }
                let id = row["id"]
                    .as_str()
                    .map(std::string::ToString::to_string)
                    .ok_or_else(|| StorageError::DataCorruption {
                        field: "session.id".into(),
                        message: "missing id in SurrealDB row".into(),
                        source: None,
                    })?;
                let record: SessionRecord = from_value(row, "session")?;
                Ok(Session {
                    id: crate::util::parse_uuid(&id, "session.id")?,
                    title: record.title,
                    model: ModelName::new(&record.model),
                    created_at: record.created_at,
                    updated_at: record.updated_at,
                    metadata: record.metadata,
                })
            })
            .collect::<StoreResult<Vec<_>>>()
    }

    async fn get_session(&self, id: &Uuid) -> StoreResult<Option<Session>> {
        let result: Option<serde_json::Value> = self
            .db
            .select(("sessions", id.to_string()))
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "surrealdb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        result
            .map(|value| {
                let record: SessionRecord = from_value(value, "session")?;
                Ok(Session {
                    id: *id,
                    title: record.title,
                    model: ModelName::new(&record.model),
                    created_at: record.created_at,
                    updated_at: record.updated_at,
                    metadata: record.metadata,
                })
            })
            .transpose()
    }

    async fn delete_session(&self, id: &Uuid) -> StoreResult<()> {
        self.db
            .delete::<Option<serde_json::Value>>(("sessions", id.to_string()))
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "surrealdb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        self.db
            .query("DELETE messages WHERE session_id = $sid")
            .bind(("sid", id.to_string()))
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "surrealdb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        Ok(())
    }

    async fn update_session(
        &self,
        id: &Uuid,
        title: &str,
        model: Option<&ModelName>,
    ) -> StoreResult<Session> {
        let id_str = id.to_string();
        let now = chrono::Utc::now();

        if let Some(m) = model {
            self.db
                .query("UPDATE sessions SET title = $title, model = $model, updated_at = $now WHERE id = $sid")
                .bind(("title", title.to_owned()))
                .bind(("model", m.as_str().to_owned()))
                .bind(("now", now))
                .bind(("sid", id_str.clone()))
                .await
                .map_err(|e| StorageError::BackendError {
                    backend: "surrealdb".to_owned(),
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;
        } else {
            self.db
                .query("UPDATE sessions SET title = $title, updated_at = $now WHERE id = $sid")
                .bind(("title", title.to_owned()))
                .bind(("now", now))
                .bind(("sid", id_str.clone()))
                .await
                .map_err(|e| StorageError::BackendError {
                    backend: "surrealdb".to_owned(),
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;
        }

        let result: Option<serde_json::Value> = self
            .db
            .select(("sessions", id_str))
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "surrealdb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let value = result.ok_or_else(|| StorageError::NotFound { id: id.to_string() })?;

        let record: SessionRecord = from_value(value, "session")?;
        Ok(Session {
            id: *id,
            title: record.title,
            model: ModelName::new(&record.model),
            created_at: record.created_at,
            updated_at: record.updated_at,
            metadata: record.metadata,
        })
    }

    async fn append_message(&self, message: MessageRecord) -> StoreResult<MessageRecord> {
        // Verify session exists
        let session_exists: Option<serde_json::Value> = self
            .db
            .select(("sessions", message.session_id.to_string()))
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "surrealdb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        if session_exists.is_none() {
            return Err(StorageError::NotFound {
                id: message.session_id.to_string(),
            });
        }

        let record = MessageStoreRecord {
            session_id: message.session_id.to_string(),
            role: role_str(&message.role).to_owned(),
            content: message.content.clone(),
            tool_calls: message.tool_calls.clone(),
            usage: message.usage,
            is_compaction: message.is_compaction,
            is_summary: message.is_summary,
            compaction_meta: message.compaction_meta.clone(),
            created_at: message.created_at,
        };

        let content =
            serde_json::to_value(&record).map_err(|e| StorageError::SerializationFailed {
                message: format!("message serialization: {}", e),
                source: Some(Box::new(e)),
            })?;

        self.db
            .create::<Option<serde_json::Value>>(("messages", message.id.to_string()))
            .content(content)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "surrealdb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        // Update session timestamp
        let now = chrono::Utc::now();
        self.db
            .query("UPDATE sessions SET updated_at = $now WHERE id = $sid")
            .bind(("now", now))
            .bind(("sid", message.session_id.to_string()))
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "surrealdb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        Ok(message)
    }

    async fn list_messages(&self, session_id: &Uuid) -> StoreResult<Vec<MessageRecord>> {
        let mut result = self
            .db
            .query("SELECT * FROM messages WHERE session_id = $sid ORDER BY created_at")
            .bind(("sid", session_id.to_string()))
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "surrealdb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let rows: Vec<serde_json::Value> =
            result.take(0).map_err(|e| StorageError::BackendError {
                backend: "surrealdb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        rows.into_iter()
            .map(|row| {
                // SurrealDB returns rows as { id: "messages:⟨uuid⟩", ...fields }
                let id_str = row["id"]
                    .as_str()
                    .map(std::string::ToString::to_string)
                    .ok_or_else(|| StorageError::DataCorruption {
                        field: "message.id".into(),
                        message: "missing id in SurrealDB row".into(),
                        source: None,
                    })?;
                let record: MessageStoreRecord = from_value(row, "message")?;
                Ok(MessageRecord {
                    id: crate::util::parse_uuid(&id_str, "message.id")?,
                    session_id: crate::util::parse_uuid(&record.session_id, "message.session_id")?,
                    role: parse_role(&record.role),
                    content: record.content,
                    tool_calls: record.tool_calls,
                    tool_call_id: None,
                    tool_name: None,
                    usage: record.usage,
                    created_at: record.created_at,
                    is_compaction: record.is_compaction,
                    is_summary: record.is_summary,
                    compaction_meta: record.compaction_meta,
                })
            })
            .collect::<StoreResult<Vec<_>>>()
    }

    async fn update_usage(&self, message_id: &Uuid, usage: TokenUsage) -> StoreResult<()> {
        let result: Option<serde_json::Value> = self
            .db
            .update(("messages", message_id.to_string()))
            .merge(serde_json::json!({ "usage": usage }))
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "surrealdb".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        if result.is_none() {
            return Err(StorageError::NotFound {
                id: message_id.to_string(),
            });
        }
        Ok(())
    }
}
