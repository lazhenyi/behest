//! Redis-backed implementation of [`SessionDataStore`].
//!
//! Session data is stored as Redis hashes with key pattern
//! `behest:session_data:{session_id}`. Each field in the hash is a JSON-
//! serialised [`Value`].

use async_trait::async_trait;
use redis::AsyncCommands;
use serde_json::Value;
use uuid::Uuid;

use super::invocation::{SessionDataError, SessionDataStore};

/// Redis-backed implementation of [`SessionDataStore`].
///
/// Each session's temporary KV data lives in a single Redis hash keyed by
/// `behest:session_data:{session_id}`. Values are stored as JSON strings.
///
/// # Example
///
/// ```rust,no_run
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// use behest::runtime::RedisSessionDataStore;
/// use behest::runtime::SessionDataStore;
/// use serde_json::json;
/// use uuid::Uuid;
///
/// let store = RedisSessionDataStore::new("redis://127.0.0.1:6379")?;
/// let sid = Uuid::new_v4();
/// store.set(sid, "key".into(), json!({"x": 1})).await?;
/// # Ok(())
/// # }
/// ```
pub struct RedisSessionDataStore {
    client: redis::Client,
}

impl RedisSessionDataStore {
    /// Creates a Redis session data store from a connection URL.
    ///
    /// # Errors
    ///
    /// Returns [`SessionDataError::Storage`] when the URL is malformed.
    pub fn new(url: &str) -> Result<Self, SessionDataError> {
        let client = redis::Client::open(url).map_err(|e| SessionDataError::Storage {
            message: format!("redis client error: {e}"),
        })?;
        Ok(Self { client })
    }

    /// Creates a Redis session data store from an existing `redis::Client`.
    #[must_use]
    pub fn from_client(client: redis::Client) -> Self {
        Self { client }
    }

    async fn conn(&self) -> Result<redis::aio::MultiplexedConnection, SessionDataError> {
        self.client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| SessionDataError::Storage {
                message: format!("redis connection error: {e}"),
            })
    }

    fn hash_key(session_id: Uuid) -> String {
        format!("behest:session_data:{session_id}")
    }
}

#[async_trait]
impl SessionDataStore for RedisSessionDataStore {
    async fn set(
        &self,
        session_id: Uuid,
        key: String,
        value: Value,
    ) -> Result<(), SessionDataError> {
        let mut conn = self.conn().await?;
        let hash = Self::hash_key(session_id);
        let json = serde_json::to_string(&value).map_err(|e| SessionDataError::Storage {
            message: format!("serialization error: {e}"),
        })?;
        conn.hset::<_, _, _, ()>(&hash, &key, &json)
            .await
            .map_err(|e| SessionDataError::Storage {
                message: format!("redis HSET error: {e}"),
            })
    }

    async fn get(&self, session_id: Uuid, key: &str) -> Result<Option<Value>, SessionDataError> {
        let mut conn = self.conn().await?;
        let hash = Self::hash_key(session_id);
        let raw: Option<String> =
            conn.hget(&hash, key)
                .await
                .map_err(|e| SessionDataError::Storage {
                    message: format!("redis HGET error: {e}"),
                })?;
        match raw {
            Some(s) => {
                let val: Value =
                    serde_json::from_str(&s).map_err(|e| SessionDataError::Storage {
                        message: format!("deserialization error: {e}"),
                    })?;
                Ok(Some(val))
            }
            None => Ok(None),
        }
    }

    async fn delete(&self, session_id: Uuid, key: &str) -> Result<(), SessionDataError> {
        let mut conn = self.conn().await?;
        let hash = Self::hash_key(session_id);
        conn.hdel::<_, _, ()>(&hash, key)
            .await
            .map_err(|e| SessionDataError::Storage {
                message: format!("redis HDEL error: {e}"),
            })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn hash_key_format() {
        let sid = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        assert_eq!(
            RedisSessionDataStore::hash_key(sid),
            "behest:session_data:550e8400-e29b-41d4-a716-446655440000"
        );
    }
}
