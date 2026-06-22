//! Redis session store using hashes and sorted sets.
#![allow(clippy::cast_precision_loss)]

use async_trait::async_trait;
use redis::AsyncCommands;
use uuid::Uuid;

use crate::error::StorageError;
use crate::provider::{ModelName, TokenUsage};
use crate::store::{MessageRecord, Session, SessionStore, StoreResult};

/// Redis-backed session store.
///
/// Sessions are stored as hashes with key pattern `session:{id}`.
/// Messages are stored as a sorted set per session with key `messages:{session_id}`,
/// scored by creation timestamp for chronological ordering.
pub struct RedisSessionStore {
    client: redis::Client,
}

impl RedisSessionStore {
    /// Creates a Redis session store from a connection URL.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::ConnectionFailed`] when the URL is invalid.
    pub fn new(url: &str) -> StoreResult<Self> {
        let client = redis::Client::open(url).map_err(|e| StorageError::ConnectionFailed {
            backend: "redis".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;
        Ok(Self { client })
    }

    /// Creates a Redis session store from an existing client.
    #[must_use]
    pub fn from_client(client: redis::Client) -> Self {
        Self { client }
    }

    async fn conn(&self) -> StoreResult<redis::aio::MultiplexedConnection> {
        self.client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| StorageError::ConnectionFailed {
                backend: "redis".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })
    }

    fn session_key(id: &Uuid) -> String {
        format!("session:{id}")
    }

    fn messages_key(session_id: &Uuid) -> String {
        format!("messages:{session_id}")
    }

    fn message_index_key(message_id: &Uuid) -> String {
        format!("message_index:{message_id}")
    }
}

#[async_trait]
impl SessionStore for RedisSessionStore {
    async fn create_session(&self, session: Session) -> StoreResult<Session> {
        let mut conn = self.conn().await?;
        let key = Self::session_key(&session.id);

        redis::pipe()
            .hset(&key, "id", session.id.to_string())
            .hset(&key, "title", &session.title)
            .hset(&key, "model", session.model.as_str())
            .hset(
                &key,
                "metadata",
                crate::store::util::to_json_string(&session.metadata, "session.metadata")?,
            )
            .hset(&key, "created_at", session.created_at.to_rfc3339())
            .hset(&key, "updated_at", session.updated_at.to_rfc3339())
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "redis".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        Ok(session)
    }

    async fn list_sessions(&self) -> StoreResult<Vec<Session>> {
        let mut conn = self.conn().await?;

        let keys: Vec<String> = redis::cmd("KEYS")
            .arg("session:*")
            .query_async(&mut conn)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "redis".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let mut sessions = Vec::new();
        for key in keys {
            if let Some(session) = load_session_from_redis(&mut conn, &key).await? {
                sessions.push(session);
            }
        }

        sessions.sort_by_key(|s| std::cmp::Reverse(s.updated_at));
        Ok(sessions)
    }

    async fn get_session(&self, id: &Uuid) -> StoreResult<Option<Session>> {
        let mut conn = self.conn().await?;
        let key = Self::session_key(id);
        load_session_from_redis(&mut conn, &key).await
    }

    async fn delete_session(&self, id: &Uuid) -> StoreResult<()> {
        let mut conn = self.conn().await?;
        let session_key = Self::session_key(id);
        let messages_key = Self::messages_key(id);

        redis::pipe()
            .del(&session_key)
            .del(&messages_key)
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "redis".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        Ok(())
    }

    async fn append_message(&self, message: MessageRecord) -> StoreResult<MessageRecord> {
        let mut conn = self.conn().await?;
        let messages_key = Self::messages_key(&message.session_id);
        let session_key = Self::session_key(&message.session_id);

        // Verify session exists
        let exists: bool = redis::cmd("EXISTS")
            .arg(&session_key)
            .query_async(&mut conn)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "redis".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        if !exists {
            return Err(StorageError::NotFound {
                id: message.session_id.to_string(),
            });
        }

        let json =
            serde_json::to_string(&message).map_err(|e| StorageError::SerializationFailed {
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let score = message.created_at.timestamp_millis() as f64;

        conn.zadd::<_, _, _, ()>(&messages_key, &json, score)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "redis".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        // Store message_id -> session_id mapping for update_usage lookup
        let index_key = Self::message_index_key(&message.id);
        let now = chrono::Utc::now();
        redis::pipe()
            .hset(&index_key, "session_id", message.session_id.to_string())
            .hset(&index_key, "score", score.to_string())
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "redis".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        // Update session's updated_at
        conn.hset::<_, _, _, ()>(&session_key, "updated_at", now.to_rfc3339())
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "redis".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        Ok(message)
    }

    async fn list_messages(&self, session_id: &Uuid) -> StoreResult<Vec<MessageRecord>> {
        let mut conn = self.conn().await?;
        let messages_key = Self::messages_key(session_id);

        let json_items: Vec<String> =
            conn.zrange(&messages_key, 0, -1)
                .await
                .map_err(|e| StorageError::BackendError {
                    backend: "redis".to_owned(),
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;

        let mut messages = Vec::new();
        for json in json_items {
            let record: MessageRecord =
                serde_json::from_str(&json).map_err(|e| StorageError::SerializationFailed {
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;
            messages.push(record);
        }

        Ok(messages)
    }

    async fn update_usage(&self, message_id: &Uuid, usage: TokenUsage) -> StoreResult<()> {
        let mut conn = self.conn().await?;
        let index_key = Self::message_index_key(message_id);

        // Look up the session_id for this message
        let fields: Vec<Option<String>> = redis::cmd("HMGET")
            .arg(&index_key)
            .arg("session_id")
            .arg("score")
            .query_async(&mut conn)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "redis".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        let session_id_str = fields[0].as_deref().ok_or_else(|| StorageError::NotFound {
            id: message_id.to_string(),
        })?;

        let session_id =
            crate::store::util::parse_uuid(session_id_str, "message_index.session_id")?;

        // Fetch all messages for the session, find the target, update usage
        let messages_key = Self::messages_key(&session_id);
        let json_items: Vec<String> =
            conn.zrange(&messages_key, 0, -1)
                .await
                .map_err(|e| StorageError::BackendError {
                    backend: "redis".to_owned(),
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;

        let mut found = false;
        for json in &json_items {
            let mut record: MessageRecord =
                serde_json::from_str(json).map_err(|e| StorageError::SerializationFailed {
                    message: e.to_string(),
                    source: Some(Box::new(e)),
                })?;

            if record.id == *message_id {
                // Remove old entry, re-insert with updated usage
                conn.zrem::<_, _, ()>(&messages_key, json)
                    .await
                    .map_err(|e| StorageError::BackendError {
                        backend: "redis".to_owned(),
                        message: e.to_string(),
                        source: Some(Box::new(e)),
                    })?;

                record.usage = Some(usage);
                let updated_json = serde_json::to_string(&record).map_err(|e| {
                    StorageError::SerializationFailed {
                        message: e.to_string(),
                        source: Some(Box::new(e)),
                    }
                })?;

                let score = record.created_at.timestamp_millis() as f64;
                conn.zadd::<_, _, _, ()>(&messages_key, &updated_json, score)
                    .await
                    .map_err(|e| StorageError::BackendError {
                        backend: "redis".to_owned(),
                        message: e.to_string(),
                        source: Some(Box::new(e)),
                    })?;

                found = true;
                break;
            }
        }

        if !found {
            return Err(StorageError::NotFound {
                id: message_id.to_string(),
            });
        }

        Ok(())
    }
}

async fn load_session_from_redis(
    conn: &mut redis::aio::MultiplexedConnection,
    key: &str,
) -> StoreResult<Option<Session>> {
    let fields: Vec<Option<String>> = redis::cmd("HMGET")
        .arg(key)
        .arg("id")
        .arg("title")
        .arg("model")
        .arg("metadata")
        .arg("created_at")
        .arg("updated_at")
        .query_async(conn)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "redis".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

    if fields.iter().all(Option::is_none) {
        return Ok(None);
    }

    let id_str = fields[0]
        .as_deref()
        .ok_or_else(|| StorageError::DataCorruption {
            field: "session.id".into(),
            message: "missing id field in Redis hash".into(),
            source: None,
        })?;
    let id = crate::store::util::parse_uuid(id_str, "session.id")?;

    let title = fields[1]
        .clone()
        .ok_or_else(|| StorageError::DataCorruption {
            field: "session.title".into(),
            message: "missing title field in Redis hash".into(),
            source: None,
        })?;

    let model = fields[2]
        .clone()
        .ok_or_else(|| StorageError::DataCorruption {
            field: "session.model".into(),
            message: "missing model field in Redis hash".into(),
            source: None,
        })?;

    let metadata_str = fields[3].as_deref().unwrap_or("{}");
    let metadata =
        serde_json::from_str(metadata_str).map_err(|e| StorageError::DataCorruption {
            field: "session.metadata".into(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

    let created_at_str = fields[4]
        .as_deref()
        .ok_or_else(|| StorageError::DataCorruption {
            field: "session.created_at".into(),
            message: "missing created_at field in Redis hash".into(),
            source: None,
        })?;
    let created_at = crate::store::util::parse_rfc3339(created_at_str, "session.created_at")?;

    let updated_at_str = fields[5]
        .as_deref()
        .ok_or_else(|| StorageError::DataCorruption {
            field: "session.updated_at".into(),
            message: "missing updated_at field in Redis hash".into(),
            source: None,
        })?;
    let updated_at = crate::store::util::parse_rfc3339(updated_at_str, "session.updated_at")?;

    Ok(Some(Session {
        id,
        title,
        model: ModelName::new(&model),
        created_at,
        updated_at,
        metadata,
    }))
}
