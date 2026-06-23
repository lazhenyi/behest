//! SQL session store implementation for PostgreSQL, MySQL, and SQLite.

use async_trait::async_trait;
use sqlx::Pool;
use uuid::Uuid;

use crate::error::StorageError;
use crate::provider::{ContentPart, TokenUsage, ToolCall};
use crate::store::{
    CompactionMeta, MessageRecord, MessageRole, Session, SessionStore, StoreResult,
};

/// SQL-backed session store supporting PostgreSQL, MySQL, and SQLite.
///
/// Uses runtime SQL queries for cross-database compatibility. The appropriate
/// database pool type is selected via Cargo feature flags.
///
/// # Migrations
///
/// Run the SQL files in `src/store/sql/migrations/{postgres,mysql,sqlite}/`
/// against your database before using this store, or use
/// [`SqlSessionStore::migrate`] to apply them programmatically.
pub struct SqlSessionStore {
    #[cfg(feature = "sqlx-postgres")]
    pool: Pool<sqlx::Postgres>,
    #[cfg(all(feature = "sqlx-mysql", not(feature = "sqlx-postgres")))]
    pool: Pool<sqlx::MySql>,
    #[cfg(all(
        feature = "sqlx-sqlite",
        not(feature = "sqlx-postgres"),
        not(feature = "sqlx-mysql")
    ))]
    pool: Pool<sqlx::Sqlite>,
}

impl SqlSessionStore {
    /// Creates a SQL session store from a PostgreSQL pool.
    #[cfg(feature = "sqlx-postgres")]
    #[must_use]
    pub fn new(pool: Pool<sqlx::Postgres>) -> Self {
        Self { pool }
    }

    /// Creates a SQL session store from a MySQL pool.
    #[cfg(all(feature = "sqlx-mysql", not(feature = "sqlx-postgres")))]
    #[must_use]
    pub fn new(pool: Pool<sqlx::MySql>) -> Self {
        Self { pool }
    }

    /// Creates a SQL session store from a SQLite pool.
    #[cfg(all(
        feature = "sqlx-sqlite",
        not(feature = "sqlx-postgres"),
        not(feature = "sqlx-mysql")
    ))]
    #[must_use]
    pub fn new(pool: Pool<sqlx::Sqlite>) -> Self {
        Self { pool }
    }

    /// Runs embedded migrations against the connected database.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::MigrationFailed`] when migrations fail.
    #[cfg(feature = "sqlx-postgres")]
    pub async fn migrate(&self) -> StoreResult<()> {
        sqlx::migrate!("src/store/sql/migrations/postgres")
            .run(&self.pool)
            .await
            .map_err(|e| StorageError::MigrationFailed {
                backend: "postgres".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })
    }

    /// Runs embedded migrations against the connected MySQL database.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::MigrationFailed`] when migrations fail.
    #[cfg(all(feature = "sqlx-mysql", not(feature = "sqlx-postgres")))]
    pub async fn migrate(&self) -> StoreResult<()> {
        sqlx::migrate!("src/store/sql/migrations/mysql")
            .run(&self.pool)
            .await
            .map_err(|e| StorageError::MigrationFailed {
                backend: "mysql".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })
    }

    /// Runs embedded migrations against the connected SQLite database.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::MigrationFailed`] when migrations fail.
    #[cfg(all(
        feature = "sqlx-sqlite",
        not(feature = "sqlx-postgres"),
        not(feature = "sqlx-mysql")
    ))]
    pub async fn migrate(&self) -> StoreResult<()> {
        sqlx::migrate!("src/store/sql/migrations/sqlite")
            .run(&self.pool)
            .await
            .map_err(|e| StorageError::MigrationFailed {
                backend: "sqlite".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })
    }
}

fn ser_content(parts: &[ContentPart]) -> StoreResult<String> {
    crate::store::util::to_json_string(parts, "message.content")
}

fn ser_tool_calls(calls: &[ToolCall]) -> StoreResult<String> {
    crate::store::util::to_json_string(calls, "message.tool_calls")
}

fn ser_usage(usage: Option<TokenUsage>) -> StoreResult<Option<String>> {
    usage
        .map(|u| crate::store::util::to_json_string(&u, "message.usage"))
        .transpose()
}

fn ser_metadata(metadata: &serde_json::Value) -> StoreResult<String> {
    crate::store::util::to_json_string(metadata, "metadata")
}

fn de_content(s: &str) -> StoreResult<Vec<ContentPart>> {
    crate::store::util::from_json_str(s, "message.content")
}

fn de_tool_calls(s: &str) -> StoreResult<Vec<ToolCall>> {
    crate::store::util::from_json_str(s, "message.tool_calls")
}

fn de_usage(s: Option<&str>) -> StoreResult<Option<TokenUsage>> {
    s.map(|v| crate::store::util::from_json_str(v, "message.usage"))
        .transpose()
}

fn de_metadata(s: &str) -> StoreResult<serde_json::Value> {
    crate::store::util::from_json_str(s, "metadata")
}

fn ser_compaction_meta(meta: Option<&CompactionMeta>) -> StoreResult<Option<String>> {
    meta.map(|m| crate::store::util::to_json_string(m, "compaction_meta"))
        .transpose()
}

fn de_compaction_meta(s: Option<&str>) -> StoreResult<Option<CompactionMeta>> {
    s.map(|v| crate::store::util::from_json_str(v, "compaction_meta"))
        .transpose()
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn role_to_str(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

fn role_from_str(s: &str) -> MessageRole {
    match s {
        "system" => MessageRole::System,
        "assistant" => MessageRole::Assistant,
        "tool" => MessageRole::Tool,
        _ => MessageRole::User,
    }
}

// --- PostgreSQL implementation ---

#[cfg(feature = "sqlx-postgres")]
#[async_trait]
impl SessionStore for SqlSessionStore {
    #[tracing::instrument(skip(self), fields(session.id = %session.id))]
    async fn create_session(&self, session: Session) -> StoreResult<Session> {
        sqlx::query(
            "INSERT INTO sessions (id, title, model, metadata, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(session.id)
        .bind(&session.title)
        .bind(session.model.as_str())
        .bind(ser_metadata(&session.metadata)?)
        .bind(session.created_at)
        .bind(session.updated_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "postgres".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;
        Ok(session)
    }

    async fn list_sessions(&self) -> StoreResult<Vec<Session>> {
        use crate::provider::ModelName;
        use chrono::{DateTime, Utc};

        let rows = sqlx::query_as::<_, (Uuid, String, String, String, DateTime<Utc>, DateTime<Utc>)>(
            "SELECT id, title, model, metadata, created_at, updated_at FROM sessions ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "postgres".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        rows.into_iter()
            .map(|(id, title, model, metadata, created_at, updated_at)| {
                Ok(Session {
                    id,
                    title,
                    model: ModelName::new(&model),
                    created_at,
                    updated_at,
                    metadata: de_metadata(&metadata)?,
                })
            })
            .collect()
    }

    async fn get_session(&self, id: &Uuid) -> StoreResult<Option<Session>> {
        use crate::provider::ModelName;
        use chrono::{DateTime, Utc};

        let row = sqlx::query_as::<_, (Uuid, String, String, String, DateTime<Utc>, DateTime<Utc>)>(
            "SELECT id, title, model, metadata, created_at, updated_at FROM sessions WHERE id = $1",
        )
        .bind(*id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "postgres".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        row.map(|(id, title, model, metadata, created_at, updated_at)| {
            Ok(Session {
                id,
                title,
                model: ModelName::new(&model),
                created_at,
                updated_at,
                metadata: de_metadata(&metadata)?,
            })
        })
        .transpose()
    }

    async fn delete_session(&self, id: &Uuid) -> StoreResult<()> {
        sqlx::query("DELETE FROM sessions WHERE id = $1")
            .bind(*id)
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "postgres".to_owned(),
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
        use crate::provider::ModelName;

        let now = chrono::Utc::now();
        let row = if let Some(m) = model {
            sqlx::query_as::<_, (Uuid, String, String, String, DateTime<Utc>, DateTime<Utc>)>(
                "UPDATE sessions SET title = $1, model = $2, updated_at = $3 WHERE id = $4 RETURNING id, title, model, metadata, created_at, updated_at",
            )
            .bind(title)
            .bind(m.as_str())
            .bind(now)
            .bind(*id)
            .fetch_optional(&self.pool)
            .await
        } else {
            sqlx::query_as::<_, (Uuid, String, String, String, DateTime<Utc>, DateTime<Utc>)>(
                "UPDATE sessions SET title = $1, updated_at = $2 WHERE id = $3 RETURNING id, title, model, metadata, created_at, updated_at",
            )
            .bind(title)
            .bind(now)
            .bind(*id)
            .fetch_optional(&self.pool)
            .await
        }
        .map_err(|e| StorageError::BackendError {
            backend: "postgres".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        row.map(|(id, title, model, metadata, created_at, updated_at)| {
            Ok(Session {
                id,
                title,
                model: ModelName::new(&model),
                created_at,
                updated_at,
                metadata: de_metadata(&metadata)?,
            })
        })
        .ok_or_else(|| StorageError::NotFound { id: id.to_string() })?
    }

    async fn append_message(&self, message: MessageRecord) -> StoreResult<MessageRecord> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "postgres".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        // Verify session exists
        let exists: Option<(Uuid,)> = sqlx::query_as("SELECT id FROM sessions WHERE id = $1")
            .bind(message.session_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "postgres".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        if exists.is_none() {
            return Err(StorageError::NotFound {
                id: message.session_id.to_string(),
            });
        }

        sqlx::query(
            "INSERT INTO messages (id, session_id, role, content, tool_calls, usage, is_compaction, is_summary, compaction_meta, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
        )
        .bind(message.id)
        .bind(message.session_id)
        .bind(role_to_str(&message.role))
        .bind(ser_content(&message.content)?)
        .bind(ser_tool_calls(&message.tool_calls)?)
        .bind(ser_usage(message.usage)?)
        .bind(message.is_compaction)
        .bind(message.is_summary)
        .bind(ser_compaction_meta(message.compaction_meta.as_ref())?)
        .bind(message.created_at)
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "postgres".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        // Update session timestamp
        let now = chrono::Utc::now();
        sqlx::query("UPDATE sessions SET updated_at = $1 WHERE id = $2")
            .bind(now)
            .bind(message.session_id)
            .execute(&mut *tx)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "postgres".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        tx.commit().await.map_err(|e| StorageError::BackendError {
            backend: "postgres".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        Ok(message)
    }

    async fn list_messages(&self, session_id: &Uuid) -> StoreResult<Vec<MessageRecord>> {
        use chrono::{DateTime, Utc};

        let rows = sqlx::query_as::<
            _,
            (
                Uuid,
                Uuid,
                String,
                String,
                String,
                Option<String>,
                bool,
                bool,
                Option<String>,
                DateTime<Utc>,
            ),
        >(
            "SELECT id, session_id, role, content, tool_calls, usage, is_compaction, is_summary, compaction_meta, created_at FROM messages WHERE session_id = $1 ORDER BY created_at",
        )
        .bind(*session_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "postgres".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        rows.into_iter()
            .map(
                |(
                    id,
                    sid,
                    role,
                    content,
                    tool_calls,
                    usage,
                    is_compaction,
                    is_summary,
                    compaction_meta,
                    created_at,
                )| {
                    Ok(MessageRecord {
                        id,
                        session_id: sid,
                        role: role_from_str(&role),
                        content: de_content(&content)?,
                        tool_calls: de_tool_calls(&tool_calls)?,
                        tool_call_id: None,
                        tool_name: None,
                        usage: de_usage(usage.as_deref())?,
                        created_at,
                        is_compaction,
                        is_summary,
                        compaction_meta: de_compaction_meta(compaction_meta.as_deref())?,
                    })
                },
            )
            .collect()
    }

    async fn update_usage(&self, message_id: &Uuid, usage: TokenUsage) -> StoreResult<()> {
        let result = sqlx::query("UPDATE messages SET usage = $1 WHERE id = $2")
            .bind(ser_usage(Some(usage))?)
            .bind(*message_id)
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "postgres".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        if result.rows_affected() == 0 {
            return Err(StorageError::NotFound {
                id: message_id.to_string(),
            });
        }
        Ok(())
    }

    async fn health_check(&self) -> StoreResult<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::ConnectionFailed {
                backend: "postgres".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;
        Ok(())
    }
}

// --- MySQL implementation ---

#[cfg(all(feature = "sqlx-mysql", not(feature = "sqlx-postgres")))]
#[async_trait]
impl SessionStore for SqlSessionStore {
    async fn create_session(&self, session: Session) -> StoreResult<Session> {
        sqlx::query(
            "INSERT INTO sessions (id, title, model, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(session.id.to_string())
        .bind(&session.title)
        .bind(session.model.as_str())
        .bind(ser_metadata(&session.metadata)?)
        .bind(session.created_at)
        .bind(session.updated_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "mysql".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;
        Ok(session)
    }

    async fn list_sessions(&self) -> StoreResult<Vec<Session>> {
        use crate::provider::ModelName;
        use chrono::{DateTime, Utc};

        let rows = sqlx::query_as::<_, (String, String, String, String, DateTime<Utc>, DateTime<Utc>)>(
            "SELECT id, title, model, metadata, created_at, updated_at FROM sessions ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "mysql".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        rows.into_iter()
            .map(|(id, title, model, metadata, created_at, updated_at)| {
                Ok(Session {
                    id: crate::store::util::parse_uuid(&id, "session.id")?,
                    title,
                    model: ModelName::new(&model),
                    created_at,
                    updated_at,
                    metadata: de_metadata(&metadata)?,
                })
            })
            .collect()
    }

    async fn get_session(&self, id: &Uuid) -> StoreResult<Option<Session>> {
        use crate::provider::ModelName;
        use chrono::{DateTime, Utc};

        let row = sqlx::query_as::<_, (String, String, String, String, DateTime<Utc>, DateTime<Utc>)>(
            "SELECT id, title, model, metadata, created_at, updated_at FROM sessions WHERE id = ?",
        )
        .bind(id.to_string())
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "mysql".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        row.map(|(id, title, model, metadata, created_at, updated_at)| {
            Ok(Session {
                id: crate::store::util::parse_uuid(&id, "session.id")?,
                title,
                model: ModelName::new(&model),
                created_at,
                updated_at,
                metadata: de_metadata(&metadata)?,
            })
        })
        .transpose()
    }

    async fn delete_session(&self, id: &Uuid) -> StoreResult<()> {
        sqlx::query("DELETE FROM sessions WHERE id = ?")
            .bind(id.to_string())
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mysql".to_owned(),
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
        use crate::provider::ModelName;

        let now = chrono::Utc::now();
        let id_str = id.to_string();

        let result = if let Some(m) = model {
            sqlx::query("UPDATE sessions SET title = ?, model = ?, updated_at = ? WHERE id = ?")
                .bind(title)
                .bind(m.as_str())
                .bind(now)
                .bind(&id_str)
                .execute(&self.pool)
                .await
        } else {
            sqlx::query("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?")
                .bind(title)
                .bind(now)
                .bind(&id_str)
                .execute(&self.pool)
                .await
        }
        .map_err(|e| StorageError::BackendError {
            backend: "mysql".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        if result.rows_affected() == 0 {
            return Err(StorageError::NotFound { id: id.to_string() });
        }

        // Read back the updated row
        let row = sqlx::query_as::<_, (String, String, String, String, DateTime<Utc>, DateTime<Utc>)>(
            "SELECT id, title, model, metadata, created_at, updated_at FROM sessions WHERE id = ?",
        )
        .bind(&id_str)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "mysql".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        let (id, title, model, metadata, created_at, updated_at) =
            row.ok_or_else(|| StorageError::NotFound { id: id_str.clone() })?;

        Ok(Session {
            id: crate::store::util::parse_uuid(&id, "session.id")?,
            title,
            model: ModelName::new(&model),
            created_at,
            updated_at,
            metadata: de_metadata(&metadata)?,
        })
    }

    async fn append_message(&self, message: MessageRecord) -> StoreResult<MessageRecord> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mysql".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        // Verify session exists
        let exists: Option<(String,)> = sqlx::query_as("SELECT id FROM sessions WHERE id = ?")
            .bind(message.session_id.to_string())
            .fetch_optional(&mut *tx)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mysql".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        if exists.is_none() {
            return Err(StorageError::NotFound {
                id: message.session_id.to_string(),
            });
        }

        sqlx::query(
            "INSERT INTO messages (id, session_id, role, content, tool_calls, `usage`, is_compaction, is_summary, compaction_meta, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(message.id.to_string())
        .bind(message.session_id.to_string())
        .bind(role_to_str(&message.role))
        .bind(ser_content(&message.content)?)
        .bind(ser_tool_calls(&message.tool_calls)?)
        .bind(ser_usage(message.usage)?)
        .bind(message.is_compaction)
        .bind(message.is_summary)
        .bind(ser_compaction_meta(message.compaction_meta.as_ref())?)
        .bind(message.created_at)
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "mysql".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        let now = chrono::Utc::now();
        sqlx::query("UPDATE sessions SET updated_at = ? WHERE id = ?")
            .bind(now)
            .bind(message.session_id.to_string())
            .execute(&mut *tx)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mysql".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        tx.commit().await.map_err(|e| StorageError::BackendError {
            backend: "mysql".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        Ok(message)
    }

    async fn list_messages(&self, session_id: &Uuid) -> StoreResult<Vec<MessageRecord>> {
        use chrono::{DateTime, Utc};

        let rows = sqlx::query_as::<_, (String, String, String, String, String, Option<String>, i8, i8, Option<String>, DateTime<Utc>)>(
            "SELECT id, session_id, role, content, tool_calls, `usage`, is_compaction, is_summary, compaction_meta, created_at FROM messages WHERE session_id = ? ORDER BY created_at",
        )
        .bind(session_id.to_string())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "mysql".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        rows.into_iter()
            .map(
                |(
                    id,
                    sid,
                    role,
                    content,
                    tool_calls,
                    usage,
                    is_compaction,
                    is_summary,
                    compaction_meta,
                    created_at,
                )| {
                    Ok(MessageRecord {
                        id: crate::store::util::parse_uuid(&id, "message.id")?,
                        session_id: crate::store::util::parse_uuid(&sid, "message.session_id")?,
                        role: role_from_str(&role),
                        content: de_content(&content)?,
                        tool_calls: de_tool_calls(&tool_calls)?,
                        tool_call_id: None,
                        tool_name: None,
                        usage: de_usage(usage.as_deref())?,
                        created_at,
                        is_compaction: is_compaction != 0,
                        is_summary: is_summary != 0,
                        compaction_meta: de_compaction_meta(compaction_meta.as_deref())?,
                    })
                },
            )
            .collect()
    }

    async fn update_usage(&self, message_id: &Uuid, usage: TokenUsage) -> StoreResult<()> {
        let result = sqlx::query("UPDATE messages SET `usage` = ? WHERE id = ?")
            .bind(ser_usage(Some(usage))?)
            .bind(message_id.to_string())
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "mysql".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        if result.rows_affected() == 0 {
            return Err(StorageError::NotFound {
                id: message_id.to_string(),
            });
        }
        Ok(())
    }

    async fn health_check(&self) -> StoreResult<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::ConnectionFailed {
                backend: "mysql".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;
        Ok(())
    }
}

// --- SQLite implementation ---

#[cfg(all(
    feature = "sqlx-sqlite",
    not(feature = "sqlx-postgres"),
    not(feature = "sqlx-mysql")
))]
#[async_trait]
impl SessionStore for SqlSessionStore {
    async fn create_session(&self, session: Session) -> StoreResult<Session> {
        sqlx::query(
            "INSERT INTO sessions (id, title, model, metadata, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )
        .bind(session.id.to_string())
        .bind(&session.title)
        .bind(session.model.as_str())
        .bind(ser_metadata(&session.metadata)?)
        .bind(session.created_at.to_rfc3339())
        .bind(session.updated_at.to_rfc3339())
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "sqlite".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;
        Ok(session)
    }

    async fn list_sessions(&self) -> StoreResult<Vec<Session>> {
        use crate::provider::ModelName;

        let rows = sqlx::query_as::<_, (String, String, String, String, String, String)>(
            "SELECT id, title, model, metadata, created_at, updated_at FROM sessions ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "sqlite".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        rows.into_iter()
            .map(|(id, title, model, metadata, created_at, updated_at)| {
                Ok(Session {
                    id: crate::store::util::parse_uuid(&id, "session.id")?,
                    title,
                    model: ModelName::new(&model),
                    created_at: crate::store::util::parse_rfc3339(
                        &created_at,
                        "session.created_at",
                    )?,
                    updated_at: crate::store::util::parse_rfc3339(
                        &updated_at,
                        "session.updated_at",
                    )?,
                    metadata: de_metadata(&metadata)?,
                })
            })
            .collect()
    }

    async fn get_session(&self, id: &Uuid) -> StoreResult<Option<Session>> {
        use crate::provider::ModelName;

        let row = sqlx::query_as::<_, (String, String, String, String, String, String)>(
            "SELECT id, title, model, metadata, created_at, updated_at FROM sessions WHERE id = ?1",
        )
        .bind(id.to_string())
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "sqlite".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        row.map(|(id, title, model, metadata, created_at, updated_at)| {
            Ok(Session {
                id: crate::store::util::parse_uuid(&id, "session.id")?,
                title,
                model: ModelName::new(&model),
                created_at: crate::store::util::parse_rfc3339(&created_at, "session.created_at")?,
                updated_at: crate::store::util::parse_rfc3339(&updated_at, "session.updated_at")?,
                metadata: de_metadata(&metadata)?,
            })
        })
        .transpose()
    }

    async fn delete_session(&self, id: &Uuid) -> StoreResult<()> {
        sqlx::query("DELETE FROM sessions WHERE id = ?1")
            .bind(id.to_string())
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "sqlite".to_owned(),
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
        use crate::provider::ModelName;

        let now = chrono::Utc::now();
        let now_str = now.to_rfc3339();
        let id_str = id.to_string();

        let result = if let Some(m) = model {
            sqlx::query("UPDATE sessions SET title = ?1, model = ?2, updated_at = ?3 WHERE id = ?4")
                .bind(title)
                .bind(m.as_str())
                .bind(&now_str)
                .bind(&id_str)
                .execute(&self.pool)
                .await
        } else {
            sqlx::query("UPDATE sessions SET title = ?1, updated_at = ?2 WHERE id = ?3")
                .bind(title)
                .bind(&now_str)
                .bind(&id_str)
                .execute(&self.pool)
                .await
        }
        .map_err(|e| StorageError::BackendError {
            backend: "sqlite".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        if result.rows_affected() == 0 {
            return Err(StorageError::NotFound { id: id.to_string() });
        }

        let row = sqlx::query_as::<_, (String, String, String, String, String, String)>(
            "SELECT id, title, model, metadata, created_at, updated_at FROM sessions WHERE id = ?1",
        )
        .bind(&id_str)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "sqlite".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        let (id, title, model, metadata, created_at, updated_at) =
            row.ok_or_else(|| StorageError::NotFound { id: id_str.clone() })?;

        Ok(Session {
            id: crate::store::util::parse_uuid(&id, "session.id")?,
            title,
            model: ModelName::new(&model),
            created_at: crate::store::util::parse_rfc3339(&created_at, "session.created_at")?,
            updated_at: crate::store::util::parse_rfc3339(&updated_at, "session.updated_at")?,
            metadata: de_metadata(&metadata)?,
        })
    }

    async fn append_message(&self, message: MessageRecord) -> StoreResult<MessageRecord> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "sqlite".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        // Verify session exists
        let exists: Option<(String,)> = sqlx::query_as("SELECT id FROM sessions WHERE id = ?1")
            .bind(message.session_id.to_string())
            .fetch_optional(&mut *tx)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "sqlite".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        if exists.is_none() {
            return Err(StorageError::NotFound {
                id: message.session_id.to_string(),
            });
        }

        sqlx::query(
            "INSERT INTO messages (id, session_id, role, content, tool_calls, usage, is_compaction, is_summary, compaction_meta, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        )
        .bind(message.id.to_string())
        .bind(message.session_id.to_string())
        .bind(role_to_str(&message.role))
        .bind(ser_content(&message.content)?)
        .bind(ser_tool_calls(&message.tool_calls)?)
        .bind(ser_usage(message.usage)?)
        .bind(message.is_compaction)
        .bind(message.is_summary)
        .bind(ser_compaction_meta(message.compaction_meta.as_ref())?)
        .bind(message.created_at.to_rfc3339())
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "sqlite".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        let now = chrono::Utc::now();
        sqlx::query("UPDATE sessions SET updated_at = ?1 WHERE id = ?2")
            .bind(now.to_rfc3339())
            .bind(message.session_id.to_string())
            .execute(&mut *tx)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "sqlite".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        tx.commit().await.map_err(|e| StorageError::BackendError {
            backend: "sqlite".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        Ok(message)
    }

    async fn list_messages(&self, session_id: &Uuid) -> StoreResult<Vec<MessageRecord>> {
        let rows = sqlx::query_as::<_, (String, String, String, String, String, Option<String>, i32, i32, Option<String>, String)>(
            "SELECT id, session_id, role, content, tool_calls, usage, is_compaction, is_summary, compaction_meta, created_at FROM messages WHERE session_id = ?1 ORDER BY created_at",
        )
        .bind(session_id.to_string())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "sqlite".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        rows.into_iter()
            .map(
                |(
                    id,
                    sid,
                    role,
                    content,
                    tool_calls,
                    usage,
                    is_compaction,
                    is_summary,
                    compaction_meta,
                    created_at,
                )| {
                    Ok(MessageRecord {
                        id: crate::store::util::parse_uuid(&id, "message.id")?,
                        session_id: crate::store::util::parse_uuid(&sid, "message.session_id")?,
                        role: role_from_str(&role),
                        content: de_content(&content)?,
                        tool_calls: de_tool_calls(&tool_calls)?,
                        tool_call_id: None,
                        tool_name: None,
                        usage: de_usage(usage.as_deref())?,
                        created_at: crate::store::util::parse_rfc3339(
                            &created_at,
                            "message.created_at",
                        )?,
                        is_compaction: is_compaction != 0,
                        is_summary: is_summary != 0,
                        compaction_meta: de_compaction_meta(compaction_meta.as_deref())?,
                    })
                },
            )
            .collect()
    }

    async fn update_usage(&self, message_id: &Uuid, usage: TokenUsage) -> StoreResult<()> {
        let result = sqlx::query("UPDATE messages SET usage = ?1 WHERE id = ?2")
            .bind(ser_usage(Some(usage))?)
            .bind(message_id.to_string())
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::BackendError {
                backend: "sqlite".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;

        if result.rows_affected() == 0 {
            return Err(StorageError::NotFound {
                id: message_id.to_string(),
            });
        }
        Ok(())
    }

    async fn health_check(&self) -> StoreResult<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::ConnectionFailed {
                backend: "sqlite".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })?;
        Ok(())
    }
}
