//! SQL-based storage backends using the `sqlx` crate.
//!
//! Supported databases:
//! - **PostgreSQL** (via feature `sqlx-postgres`): session, execution, and pgvector embedding stores
//! - **MySQL** (via feature `sqlx-mysql`): session and execution stores
//! - **SQLite** (via feature `sqlx-sqlite`): session and execution stores
//!
//! Each backend is selected at compile time via Cargo feature flags and
//! uses runtime SQL queries for cross-database compatibility.

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
