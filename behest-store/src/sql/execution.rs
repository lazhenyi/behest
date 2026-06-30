//! SQL execution store implementation for PostgreSQL, MySQL, and SQLite.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use async_trait::async_trait;
use sqlx::Pool;
use uuid::Uuid;

use crate::{
    ExecutionStore, SessionStats, StoreResult, ToolExecution, ToolExecutionStatus, UsageRecord,
};
use behest_core::error::StorageError;

/// SQL-backed execution store supporting PostgreSQL, MySQL, and SQLite.
///
/// The appropriate pool type is selected at compile time via Cargo feature flags.
/// Uses runtime SQL queries for cross-database compatibility. Implements
/// [`ExecutionStore`].
///
/// # Migrations
///
/// Run the SQL files in `src/sql/migrations/{postgres,mysql,sqlite}/`
/// against your database before using this store, or use
/// [`SqlExecutionStore::migrate`] to apply them programmatically.
pub struct SqlExecutionStore {
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

impl SqlExecutionStore {
    /// Creates a SQL execution store from a PostgreSQL pool.
    #[cfg(feature = "sqlx-postgres")]
    #[must_use]
    pub fn new(pool: Pool<sqlx::Postgres>) -> Self {
        Self { pool }
    }

    /// Creates a SQL execution store from a MySQL pool.
    #[cfg(all(feature = "sqlx-mysql", not(feature = "sqlx-postgres")))]
    #[must_use]
    pub fn new(pool: Pool<sqlx::MySql>) -> Self {
        Self { pool }
    }

    /// Creates a SQL execution store from a SQLite pool.
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
        sqlx::migrate!("src/sql/migrations/postgres")
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
        sqlx::migrate!("src/sql/migrations/mysql")
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
        sqlx::migrate!("src/sql/migrations/sqlite")
            .run(&self.pool)
            .await
            .map_err(|e| StorageError::MigrationFailed {
                backend: "sqlite".to_owned(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            })
    }
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

fn ser_json(value: &serde_json::Value) -> StoreResult<String> {
    crate::util::to_json_string(value, "execution.arguments")
}

fn de_json(s: &str) -> StoreResult<serde_json::Value> {
    crate::util::from_json_str(s, "execution.arguments")
}

fn de_json_opt(s: Option<&str>) -> StoreResult<Option<serde_json::Value>> {
    s.map(de_json).transpose()
}

fn status_to_str(status: ToolExecutionStatus) -> &'static str {
    match status {
        ToolExecutionStatus::Pending => "pending",
        ToolExecutionStatus::Success => "success",
        ToolExecutionStatus::Failed => "failed",
    }
}

fn status_from_str(s: &str) -> ToolExecutionStatus {
    match s {
        "success" => ToolExecutionStatus::Success,
        "failed" => ToolExecutionStatus::Failed,
        _ => ToolExecutionStatus::Pending,
    }
}

// ---------------------------------------------------------------------------
// PostgreSQL implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "sqlx-postgres")]
#[async_trait]
impl ExecutionStore for SqlExecutionStore {
    async fn record_execution(&self, execution: ToolExecution) -> StoreResult<ToolExecution> {
        sqlx::query(
            "INSERT INTO tool_executions (id, session_id, message_id, call_id, tool_name, arguments, result, status, error, duration_ms, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
        )
        .bind(execution.id)
        .bind(execution.session_id)
        .bind(execution.message_id)
        .bind(&execution.call_id)
        .bind(&execution.tool_name)
        .bind(ser_json(&execution.arguments)?)
        .bind(execution.result.as_ref().map(ser_json).transpose()?)
        .bind(status_to_str(execution.status))
        .bind(&execution.error)
        .bind(i64::try_from(execution.duration.as_millis()).unwrap_or(i64::MAX))
        .bind(execution.created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "postgres".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;
        Ok(execution)
    }

    async fn list_executions(&self, session_id: &Uuid) -> StoreResult<Vec<ToolExecution>> {
        use chrono::{DateTime, Utc};

        let rows = sqlx::query_as::<
            _,
            (
                Uuid, Uuid, Uuid, String, String, String, Option<String>,
                String, Option<String>, i64, DateTime<Utc>,
            ),
        >(
            "SELECT id, session_id, message_id, call_id, tool_name, arguments, result, status, error, duration_ms, created_at FROM tool_executions WHERE session_id = $1 ORDER BY created_at",
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
                    mid,
                    call_id,
                    tool_name,
                    arguments,
                    result,
                    status,
                    error,
                    duration_ms,
                    created_at,
                )| {
                    Ok(ToolExecution {
                        id,
                        session_id: sid,
                        message_id: mid,
                        call_id,
                        tool_name,
                        arguments: de_json(&arguments)?,
                        result: de_json_opt(result.as_deref())?,
                        status: status_from_str(&status),
                        error,
                        duration: std::time::Duration::from_millis(duration_ms as u64),
                        created_at,
                    })
                },
            )
            .collect()
    }

    async fn list_executions_by_message(
        &self,
        message_id: &Uuid,
    ) -> StoreResult<Vec<ToolExecution>> {
        use chrono::{DateTime, Utc};

        let rows = sqlx::query_as::<
            _,
            (
                Uuid, Uuid, Uuid, String, String, String, Option<String>,
                String, Option<String>, i64, DateTime<Utc>,
            ),
        >(
            "SELECT id, session_id, message_id, call_id, tool_name, arguments, result, status, error, duration_ms, created_at FROM tool_executions WHERE message_id = $1 ORDER BY created_at",
        )
        .bind(*message_id)
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
                    mid,
                    call_id,
                    tool_name,
                    arguments,
                    result,
                    status,
                    error,
                    duration_ms,
                    created_at,
                )| {
                    Ok(ToolExecution {
                        id,
                        session_id: sid,
                        message_id: mid,
                        call_id,
                        tool_name,
                        arguments: de_json(&arguments)?,
                        result: de_json_opt(result.as_deref())?,
                        status: status_from_str(&status),
                        error,
                        duration: std::time::Duration::from_millis(duration_ms as u64),
                        created_at,
                    })
                },
            )
            .collect()
    }

    async fn record_usage(&self, record: UsageRecord) -> StoreResult<UsageRecord> {
        sqlx::query(
            "INSERT INTO usage_records (id, session_id, message_id, provider, model, input_tokens, output_tokens, total_tokens, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        )
        .bind(record.id)
        .bind(record.session_id)
        .bind(record.message_id)
        .bind(&record.provider)
        .bind(&record.model)
        .bind(record.input_tokens as i64)
        .bind(record.output_tokens as i64)
        .bind(record.total_tokens as i64)
        .bind(record.created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "postgres".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;
        Ok(record)
    }

    async fn list_usage(&self, session_id: &Uuid) -> StoreResult<Vec<UsageRecord>> {
        use chrono::{DateTime, Utc};

        let rows = sqlx::query_as::<
            _,
            (
                Uuid, Uuid, Uuid, String, String, i64, i64, i64, DateTime<Utc>,
            ),
        >(
            "SELECT id, session_id, message_id, provider, model, input_tokens, output_tokens, total_tokens, created_at FROM usage_records WHERE session_id = $1 ORDER BY created_at",
        )
        .bind(*session_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "postgres".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        Ok(rows
            .into_iter()
            .map(
                |(
                    id,
                    sid,
                    mid,
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    created_at,
                )| {
                    UsageRecord {
                        id,
                        session_id: sid,
                        message_id: mid,
                        provider,
                        model,
                        input_tokens: input_tokens as u64,
                        output_tokens: output_tokens as u64,
                        total_tokens: total_tokens as u64,
                        created_at,
                    }
                },
            )
            .collect())
    }

    async fn session_stats(&self, session_id: &Uuid) -> StoreResult<SessionStats> {
        // Use subqueries to avoid cartesian products between tool_executions and usage_records
        let row = sqlx::query_as::<
            _,
            (Option<i64>, Option<i64>, Option<i64>, Option<i64>, Option<f64>, Option<i64>, Option<i64>, Option<i64>),
        >(
            "SELECT
                (SELECT COUNT(*) FROM messages WHERE session_id = $1) as message_count,
                (SELECT COUNT(*) FROM tool_executions WHERE session_id = $1) as tool_call_count,
                (SELECT COUNT(*) FROM tool_executions WHERE session_id = $1 AND status = 'success') as tool_success_count,
                (SELECT COUNT(*) FROM tool_executions WHERE session_id = $1 AND status = 'failed') as tool_failure_count,
                (SELECT AVG(duration_ms)::FLOAT8 FROM tool_executions WHERE session_id = $1) as avg_tool_duration_ms,
                (SELECT COALESCE(SUM(input_tokens), 0) FROM usage_records WHERE session_id = $1) as total_input_tokens,
                (SELECT COALESCE(SUM(output_tokens), 0) FROM usage_records WHERE session_id = $1) as total_output_tokens,
                (SELECT COALESCE(SUM(total_tokens), 0) FROM usage_records WHERE session_id = $1) as total_tokens",
        )
        .bind(*session_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "postgres".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        Ok(SessionStats {
            session_id: *session_id,
            message_count: row.0.unwrap_or(0) as u64,
            tool_call_count: row.1.unwrap_or(0) as u64,
            tool_success_count: row.2.unwrap_or(0) as u64,
            tool_failure_count: row.3.unwrap_or(0) as u64,
            avg_tool_duration_ms: row.4.unwrap_or(0.0) as u64,
            total_input_tokens: row.5.unwrap_or(0) as u64,
            total_output_tokens: row.6.unwrap_or(0) as u64,
            total_tokens: row.7.unwrap_or(0) as u64,
        })
    }
}

// ---------------------------------------------------------------------------
// MySQL implementation
// ---------------------------------------------------------------------------

#[cfg(all(feature = "sqlx-mysql", not(feature = "sqlx-postgres")))]
#[async_trait]
impl ExecutionStore for SqlExecutionStore {
    async fn record_execution(&self, execution: ToolExecution) -> StoreResult<ToolExecution> {
        sqlx::query(
            "INSERT INTO tool_executions (id, session_id, message_id, call_id, tool_name, arguments, result, status, error, duration_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(execution.id.to_string())
        .bind(execution.session_id.to_string())
        .bind(execution.message_id.to_string())
        .bind(&execution.call_id)
        .bind(&execution.tool_name)
        .bind(ser_json(&execution.arguments)?)
        .bind(execution.result.as_ref().map(ser_json).transpose()?)
        .bind(status_to_str(execution.status))
        .bind(&execution.error)
        .bind(i64::try_from(execution.duration.as_millis()).unwrap_or(i64::MAX))
        .bind(execution.created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "mysql".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;
        Ok(execution)
    }

    async fn list_executions(&self, session_id: &Uuid) -> StoreResult<Vec<ToolExecution>> {
        use chrono::{DateTime, Utc};

        let rows = sqlx::query_as::<
            _,
            (
                String, String, String, String, String, String, Option<String>,
                String, Option<String>, i64, DateTime<Utc>,
            ),
        >(
            "SELECT id, session_id, message_id, call_id, tool_name, arguments, result, status, error, duration_ms, created_at FROM tool_executions WHERE session_id = ? ORDER BY created_at",
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
                    mid,
                    call_id,
                    tool_name,
                    arguments,
                    result,
                    status,
                    error,
                    duration_ms,
                    created_at,
                )| {
                    Ok(ToolExecution {
                        id: crate::util::parse_uuid(&id, "execution.id")?,
                        session_id: crate::util::parse_uuid(&sid, "execution.session_id")?,
                        message_id: crate::util::parse_uuid(&mid, "execution.message_id")?,
                        call_id,
                        tool_name,
                        arguments: de_json(&arguments)?,
                        result: de_json_opt(result.as_deref())?,
                        status: status_from_str(&status),
                        error,
                        duration: std::time::Duration::from_millis(duration_ms as u64),
                        created_at,
                    })
                },
            )
            .collect()
    }

    async fn list_executions_by_message(
        &self,
        message_id: &Uuid,
    ) -> StoreResult<Vec<ToolExecution>> {
        use chrono::{DateTime, Utc};

        let rows = sqlx::query_as::<
            _,
            (
                String, String, String, String, String, String, Option<String>,
                String, Option<String>, i64, DateTime<Utc>,
            ),
        >(
            "SELECT id, session_id, message_id, call_id, tool_name, arguments, result, status, error, duration_ms, created_at FROM tool_executions WHERE message_id = ? ORDER BY created_at",
        )
        .bind(message_id.to_string())
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
                    mid,
                    call_id,
                    tool_name,
                    arguments,
                    result,
                    status,
                    error,
                    duration_ms,
                    created_at,
                )| {
                    Ok(ToolExecution {
                        id: crate::util::parse_uuid(&id, "execution.id")?,
                        session_id: crate::util::parse_uuid(&sid, "execution.session_id")?,
                        message_id: crate::util::parse_uuid(&mid, "execution.message_id")?,
                        call_id,
                        tool_name,
                        arguments: de_json(&arguments)?,
                        result: de_json_opt(result.as_deref())?,
                        status: status_from_str(&status),
                        error,
                        duration: std::time::Duration::from_millis(duration_ms as u64),
                        created_at,
                    })
                },
            )
            .collect()
    }

    async fn record_usage(&self, record: UsageRecord) -> StoreResult<UsageRecord> {
        sqlx::query(
            "INSERT INTO usage_records (id, session_id, message_id, provider, model, input_tokens, output_tokens, total_tokens, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(record.id.to_string())
        .bind(record.session_id.to_string())
        .bind(record.message_id.to_string())
        .bind(&record.provider)
        .bind(&record.model)
        .bind(record.input_tokens as i64)
        .bind(record.output_tokens as i64)
        .bind(record.total_tokens as i64)
        .bind(record.created_at)
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "mysql".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;
        Ok(record)
    }

    async fn list_usage(&self, session_id: &Uuid) -> StoreResult<Vec<UsageRecord>> {
        use chrono::{DateTime, Utc};

        let rows = sqlx::query_as::<
            _,
            (
                String, String, String, String, String, i64, i64, i64, DateTime<Utc>,
            ),
        >(
            "SELECT id, session_id, message_id, provider, model, input_tokens, output_tokens, total_tokens, created_at FROM usage_records WHERE session_id = ? ORDER BY created_at",
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
                    mid,
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    created_at,
                )| {
                    Ok(UsageRecord {
                        id: crate::util::parse_uuid(&id, "usage.id")?,
                        session_id: crate::util::parse_uuid(&sid, "usage.session_id")?,
                        message_id: crate::util::parse_uuid(&mid, "usage.message_id")?,
                        provider,
                        model,
                        input_tokens: input_tokens as u64,
                        output_tokens: output_tokens as u64,
                        total_tokens: total_tokens as u64,
                        created_at,
                    })
                },
            )
            .collect()
    }

    async fn session_stats(&self, session_id: &Uuid) -> StoreResult<SessionStats> {
        let sid = session_id.to_string();

        // MySQL: use separate queries since subquery approach is the same
        let row = sqlx::query_as::<
            _,
            (Option<i64>, Option<i64>, Option<i64>, Option<i64>, Option<f64>, Option<i64>, Option<i64>, Option<i64>),
        >(
            "SELECT
                (SELECT COUNT(*) FROM messages WHERE session_id = ?) as message_count,
                (SELECT COUNT(*) FROM tool_executions WHERE session_id = ?) as tool_call_count,
                (SELECT COUNT(*) FROM tool_executions WHERE session_id = ? AND status = 'success') as tool_success_count,
                (SELECT COUNT(*) FROM tool_executions WHERE session_id = ? AND status = 'failed') as tool_failure_count,
                (SELECT AVG(duration_ms) FROM tool_executions WHERE session_id = ?) as avg_tool_duration_ms,
                (SELECT COALESCE(SUM(input_tokens), 0) FROM usage_records WHERE session_id = ?) as total_input_tokens,
                (SELECT COALESCE(SUM(output_tokens), 0) FROM usage_records WHERE session_id = ?) as total_output_tokens,
                (SELECT COALESCE(SUM(total_tokens), 0) FROM usage_records WHERE session_id = ?) as total_tokens",
        )
        .bind(&sid)
        .bind(&sid)
        .bind(&sid)
        .bind(&sid)
        .bind(&sid)
        .bind(&sid)
        .bind(&sid)
        .bind(&sid)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "mysql".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        Ok(SessionStats {
            session_id: *session_id,
            message_count: row.0.unwrap_or(0) as u64,
            tool_call_count: row.1.unwrap_or(0) as u64,
            tool_success_count: row.2.unwrap_or(0) as u64,
            tool_failure_count: row.3.unwrap_or(0) as u64,
            avg_tool_duration_ms: row.4.unwrap_or(0.0) as u64,
            total_input_tokens: row.5.unwrap_or(0) as u64,
            total_output_tokens: row.6.unwrap_or(0) as u64,
            total_tokens: row.7.unwrap_or(0) as u64,
        })
    }
}

// ---------------------------------------------------------------------------
// SQLite implementation
// ---------------------------------------------------------------------------

#[cfg(all(
    feature = "sqlx-sqlite",
    not(feature = "sqlx-postgres"),
    not(feature = "sqlx-mysql")
))]
#[async_trait]
impl ExecutionStore for SqlExecutionStore {
    async fn record_execution(&self, execution: ToolExecution) -> StoreResult<ToolExecution> {
        sqlx::query(
            "INSERT INTO tool_executions (id, session_id, message_id, call_id, tool_name, arguments, result, status, error, duration_ms, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        )
        .bind(execution.id.to_string())
        .bind(execution.session_id.to_string())
        .bind(execution.message_id.to_string())
        .bind(&execution.call_id)
        .bind(&execution.tool_name)
        .bind(ser_json(&execution.arguments)?)
        .bind(execution.result.as_ref().map(ser_json).transpose()?)
        .bind(status_to_str(execution.status))
        .bind(&execution.error)
        .bind(i64::try_from(execution.duration.as_millis()).unwrap_or(i64::MAX))
        .bind(execution.created_at.to_rfc3339())
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "sqlite".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;
        Ok(execution)
    }

    async fn list_executions(&self, session_id: &Uuid) -> StoreResult<Vec<ToolExecution>> {
        let rows = sqlx::query_as::<
            _,
            (
                String, String, String, String, String, String, Option<String>,
                String, Option<String>, i64, String,
            ),
        >(
            "SELECT id, session_id, message_id, call_id, tool_name, arguments, result, status, error, duration_ms, created_at FROM tool_executions WHERE session_id = ?1 ORDER BY created_at",
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
                    mid,
                    call_id,
                    tool_name,
                    arguments,
                    result,
                    status,
                    error,
                    duration_ms,
                    created_at,
                )| {
                    Ok(ToolExecution {
                        id: crate::util::parse_uuid(&id, "execution.id")?,
                        session_id: crate::util::parse_uuid(&sid, "execution.session_id")?,
                        message_id: crate::util::parse_uuid(&mid, "execution.message_id")?,
                        call_id,
                        tool_name,
                        arguments: de_json(&arguments)?,
                        result: de_json_opt(result.as_deref())?,
                        status: status_from_str(&status),
                        error,
                        duration: std::time::Duration::from_millis(duration_ms as u64),
                        created_at: crate::util::parse_rfc3339(
                            &created_at,
                            "execution.created_at",
                        )?,
                    })
                },
            )
            .collect()
    }

    async fn list_executions_by_message(
        &self,
        message_id: &Uuid,
    ) -> StoreResult<Vec<ToolExecution>> {
        let rows = sqlx::query_as::<
            _,
            (
                String, String, String, String, String, String, Option<String>,
                String, Option<String>, i64, String,
            ),
        >(
            "SELECT id, session_id, message_id, call_id, tool_name, arguments, result, status, error, duration_ms, created_at FROM tool_executions WHERE message_id = ?1 ORDER BY created_at",
        )
        .bind(message_id.to_string())
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
                    mid,
                    call_id,
                    tool_name,
                    arguments,
                    result,
                    status,
                    error,
                    duration_ms,
                    created_at,
                )| {
                    Ok(ToolExecution {
                        id: crate::util::parse_uuid(&id, "execution.id")?,
                        session_id: crate::util::parse_uuid(&sid, "execution.session_id")?,
                        message_id: crate::util::parse_uuid(&mid, "execution.message_id")?,
                        call_id,
                        tool_name,
                        arguments: de_json(&arguments)?,
                        result: de_json_opt(result.as_deref())?,
                        status: status_from_str(&status),
                        error,
                        duration: std::time::Duration::from_millis(duration_ms as u64),
                        created_at: crate::util::parse_rfc3339(
                            &created_at,
                            "execution.created_at",
                        )?,
                    })
                },
            )
            .collect()
    }

    async fn record_usage(&self, record: UsageRecord) -> StoreResult<UsageRecord> {
        sqlx::query(
            "INSERT INTO usage_records (id, session_id, message_id, provider, model, input_tokens, output_tokens, total_tokens, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )
        .bind(record.id.to_string())
        .bind(record.session_id.to_string())
        .bind(record.message_id.to_string())
        .bind(&record.provider)
        .bind(&record.model)
        .bind(record.input_tokens as i64)
        .bind(record.output_tokens as i64)
        .bind(record.total_tokens as i64)
        .bind(record.created_at.to_rfc3339())
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "sqlite".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;
        Ok(record)
    }

    async fn list_usage(&self, session_id: &Uuid) -> StoreResult<Vec<UsageRecord>> {
        let rows = sqlx::query_as::<
            _,
            (
                String, String, String, String, String, i64, i64, i64, String,
            ),
        >(
            "SELECT id, session_id, message_id, provider, model, input_tokens, output_tokens, total_tokens, created_at FROM usage_records WHERE session_id = ?1 ORDER BY created_at",
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
                    mid,
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    created_at,
                )| {
                    Ok(UsageRecord {
                        id: crate::util::parse_uuid(&id, "usage.id")?,
                        session_id: crate::util::parse_uuid(&sid, "usage.session_id")?,
                        message_id: crate::util::parse_uuid(&mid, "usage.message_id")?,
                        provider,
                        model,
                        input_tokens: input_tokens as u64,
                        output_tokens: output_tokens as u64,
                        total_tokens: total_tokens as u64,
                        created_at: crate::util::parse_rfc3339(&created_at, "usage.created_at")?,
                    })
                },
            )
            .collect()
    }

    async fn session_stats(&self, session_id: &Uuid) -> StoreResult<SessionStats> {
        let sid = session_id.to_string();

        let row = sqlx::query_as::<
            _,
            (Option<i64>, Option<i64>, Option<i64>, Option<i64>, Option<f64>, Option<i64>, Option<i64>, Option<i64>),
        >(
            "SELECT
                (SELECT COUNT(*) FROM messages WHERE session_id = ?1) as message_count,
                (SELECT COUNT(*) FROM tool_executions WHERE session_id = ?1) as tool_call_count,
                (SELECT COUNT(*) FROM tool_executions WHERE session_id = ?1 AND status = 'success') as tool_success_count,
                (SELECT COUNT(*) FROM tool_executions WHERE session_id = ?1 AND status = 'failed') as tool_failure_count,
                (SELECT AVG(CAST(duration_ms AS REAL)) FROM tool_executions WHERE session_id = ?1) as avg_tool_duration_ms,
                (SELECT COALESCE(SUM(input_tokens), 0) FROM usage_records WHERE session_id = ?1) as total_input_tokens,
                (SELECT COALESCE(SUM(output_tokens), 0) FROM usage_records WHERE session_id = ?1) as total_output_tokens,
                (SELECT COALESCE(SUM(total_tokens), 0) FROM usage_records WHERE session_id = ?1) as total_tokens",
        )
        .bind(&sid)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| StorageError::BackendError {
            backend: "sqlite".to_owned(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })?;

        Ok(SessionStats {
            session_id: *session_id,
            message_count: row.0.unwrap_or(0) as u64,
            tool_call_count: row.1.unwrap_or(0) as u64,
            tool_success_count: row.2.unwrap_or(0) as u64,
            tool_failure_count: row.3.unwrap_or(0) as u64,
            avg_tool_duration_ms: row.4.unwrap_or(0.0) as u64,
            total_input_tokens: row.5.unwrap_or(0) as u64,
            total_output_tokens: row.6.unwrap_or(0) as u64,
            total_tokens: row.7.unwrap_or(0) as u64,
        })
    }
}
