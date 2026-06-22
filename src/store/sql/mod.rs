//! SQL-based storage backends using `sqlx`.
//!
//! Supports PostgreSQL, MySQL, and SQLite for session storage,
//! and PostgreSQL (with pgvector) for embedding storage.

pub mod session;

#[cfg(feature = "sqlx-postgres")]
pub mod embedding;

#[cfg(any(
    feature = "sqlx-postgres",
    feature = "sqlx-mysql",
    feature = "sqlx-sqlite"
))]
pub mod execution;

pub use session::SqlSessionStore;

#[cfg(feature = "sqlx-postgres")]
pub use embedding::SqlEmbeddingStore;

#[cfg(any(
    feature = "sqlx-postgres",
    feature = "sqlx-mysql",
    feature = "sqlx-sqlite"
))]
pub use execution::SqlExecutionStore;
