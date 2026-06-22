//! Shared utility functions for store backends.

use uuid::Uuid;

use crate::error::StorageError;
use crate::store::StoreResult;

/// Parses a UUID string, returning [`StorageError::DataCorruption`] on failure.
///
/// This replaces the unsafe `Uuid::parse_str(&id).unwrap_or_default()` pattern
/// that silently produces nil UUIDs when stored data is corrupted.
#[allow(dead_code)]
pub(crate) fn parse_uuid(s: &str, field: impl Into<String>) -> StoreResult<Uuid> {
    Uuid::parse_str(s).map_err(|e| StorageError::DataCorruption {
        field: field.into(),
        message: e.to_string(),
        source: Some(Box::new(e)),
    })
}

/// Parses a chrono `DateTime<Utc>` from an RFC 3339 string.
///
/// Returns [`StorageError::DataCorruption`] when the timestamp cannot be parsed.
#[allow(dead_code)]
pub(crate) fn parse_rfc3339(
    s: &str,
    field: impl Into<String>,
) -> StoreResult<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .map_err(|e| StorageError::DataCorruption {
            field: field.into(),
            message: e.to_string(),
            source: Some(Box::new(e)),
        })
}

/// Serializes a value to a JSON string, returning [`StorageError::SerializationFailed`]
/// on failure.
#[allow(dead_code)]
pub(crate) fn to_json_string<T: serde::Serialize + ?Sized>(
    value: &T,
    field: impl Into<String>,
) -> StoreResult<String> {
    serde_json::to_string(value).map_err(|e| StorageError::SerializationFailed {
        message: format!("{}: {}", field.into(), e),
        source: Some(Box::new(e)),
    })
}

/// Deserializes a JSON string to a value, returning [`StorageError::DataCorruption`]
/// on failure.
#[allow(dead_code)]
pub(crate) fn from_json_str<T: serde::de::DeserializeOwned>(
    s: &str,
    field: impl Into<String>,
) -> StoreResult<T> {
    serde_json::from_str(s).map_err(|e| StorageError::DataCorruption {
        field: field.into(),
        message: e.to_string(),
        source: Some(Box::new(e)),
    })
}
