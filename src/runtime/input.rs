//! Input admission and promotion pipeline.
//!
//! Provides event-sourced input lifecycle management: inputs are submitted,
//! validated, deduplicated, and admitted before entering the run loop.

use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for an input submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InputId(Uuid);

impl InputId {
    /// Creates a new input identifier.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Creates an input identifier from an existing UUID.
    #[must_use]
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Returns the underlying UUID.
    #[must_use]
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for InputId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for InputId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Lifecycle state of an input submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputState {
    /// Input has been submitted but not yet validated.
    Submitted,
    /// Input passed validation and is admitted for processing.
    Admitted,
    /// Input is currently being processed by a run.
    Processing,
    /// Input processing completed successfully.
    Completed,
    /// Input was rejected during validation.
    Rejected,
}

impl InputState {
    /// Returns true if the input is in a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Rejected)
    }
}

/// Persistent record of an input submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputRecord {
    /// Unique input identifier.
    pub id: InputId,
    /// Session this input belongs to.
    pub session_id: Uuid,
    /// Current state of the input.
    pub state: InputState,
    /// Input content.
    pub content: String,
    /// Fingerprint for deduplication.
    pub fingerprint: u64,
    /// Rejection reason (if rejected).
    pub rejection_reason: Option<String>,
    /// When the input was submitted.
    pub submitted_at: DateTime<Utc>,
    /// When the input was last updated.
    pub updated_at: DateTime<Utc>,
}

impl InputRecord {
    /// Creates a new input record in `Submitted` state.
    #[must_use]
    pub fn new(session_id: Uuid, content: String) -> Self {
        let fingerprint = Self::compute_fingerprint(&content);
        let now = Utc::now();
        Self {
            id: InputId::new(),
            session_id,
            state: InputState::Submitted,
            content,
            fingerprint,
            rejection_reason: None,
            submitted_at: now,
            updated_at: now,
        }
    }

    /// Computes a fingerprint for deduplication.
    fn compute_fingerprint(content: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        content.trim().hash(&mut hasher);
        hasher.finish()
    }

    /// Updates the state and timestamp.
    pub fn update_state(&mut self, state: InputState) {
        self.state = state;
        self.updated_at = Utc::now();
    }

    /// Rejects the input with a reason.
    pub fn reject(&mut self, reason: String) {
        self.state = InputState::Rejected;
        self.rejection_reason = Some(reason);
        self.updated_at = Utc::now();
    }
}

/// Event in the input lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputEvent {
    /// Input was submitted.
    Submitted {
        /// Unique input identifier.
        input_id: InputId,
        /// Session this input belongs to.
        session_id: Uuid,
        /// Raw input content.
        content: String,
        /// Content fingerprint for deduplication.
        fingerprint: u64,
        /// Submission timestamp.
        timestamp: DateTime<Utc>,
    },
    /// Input was admitted for processing.
    Admitted {
        /// Unique input identifier.
        input_id: InputId,
        /// Admission timestamp.
        timestamp: DateTime<Utc>,
    },
    /// Input was rejected.
    Rejected {
        /// Unique input identifier.
        input_id: InputId,
        /// Rejection reason.
        reason: String,
        /// Rejection timestamp.
        timestamp: DateTime<Utc>,
    },
    /// Input processing started.
    Processing {
        /// Unique input identifier.
        input_id: InputId,
        /// Processing start timestamp.
        timestamp: DateTime<Utc>,
    },
    /// Input processing completed.
    Completed {
        /// Unique input identifier.
        input_id: InputId,
        /// Processing completion timestamp.
        timestamp: DateTime<Utc>,
    },
}

impl InputEvent {
    /// Returns the input ID associated with this event.
    #[must_use]
    pub fn input_id(&self) -> InputId {
        match self {
            Self::Submitted { input_id, .. }
            | Self::Admitted { input_id, .. }
            | Self::Rejected { input_id, .. }
            | Self::Processing { input_id, .. }
            | Self::Completed { input_id, .. } => *input_id,
        }
    }
}

/// Configuration for input admission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAdmissionConfig {
    /// Whether admission is enabled.
    pub enabled: bool,
    /// Whether to reject empty inputs.
    pub reject_empty: bool,
    /// Whether to deduplicate inputs within a session.
    pub deduplicate: bool,
    /// Maximum input length (0 = unlimited).
    pub max_length: usize,
}

impl Default for InputAdmissionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reject_empty: true,
            deduplicate: true,
            max_length: 0,
        }
    }
}

impl InputAdmissionConfig {
    /// Sets whether admission is enabled.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Sets whether to reject empty inputs.
    #[must_use]
    pub fn with_reject_empty(mut self, reject_empty: bool) -> Self {
        self.reject_empty = reject_empty;
        self
    }

    /// Sets whether to deduplicate inputs.
    #[must_use]
    pub fn with_deduplicate(mut self, deduplicate: bool) -> Self {
        self.deduplicate = deduplicate;
        self
    }

    /// Sets the maximum input length.
    #[must_use]
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }
}

/// Input admission pipeline.
///
/// Validates, deduplicates, and admits inputs before they enter the run loop.
pub struct InputAdmission {
    config: InputAdmissionConfig,
    seen_fingerprints: Mutex<HashSet<u64>>,
}

impl InputAdmission {
    /// Creates a new input admission pipeline.
    #[must_use]
    pub fn new(config: InputAdmissionConfig) -> Self {
        Self {
            config,
            seen_fingerprints: Mutex::new(HashSet::new()),
        }
    }

    /// Admits an input record, returning events generated.
    ///
    /// If the input is rejected, the record is mutated and a `Rejected` event is returned.
    /// If admitted, an `Admitted` event is returned.
    ///
    /// # Errors
    ///
    /// Returns `InputAdmissionError` if the lock is poisoned.
    pub fn admit(&self, record: &mut InputRecord) -> Result<Vec<InputEvent>, InputAdmissionError> {
        if !self.config.enabled {
            record.update_state(InputState::Admitted);
            return Ok(vec![InputEvent::Admitted {
                input_id: record.id,
                timestamp: Utc::now(),
            }]);
        }

        let mut events = vec![InputEvent::Submitted {
            input_id: record.id,
            session_id: record.session_id,
            content: record.content.clone(),
            fingerprint: record.fingerprint,
            timestamp: record.submitted_at,
        }];

        // Validate: reject empty
        if self.config.reject_empty && record.content.trim().is_empty() {
            record.reject("empty input".to_string());
            events.push(InputEvent::Rejected {
                input_id: record.id,
                reason: "empty input".to_string(),
                timestamp: Utc::now(),
            });
            return Ok(events);
        }

        // Validate: max length
        if self.config.max_length > 0 && record.content.len() > self.config.max_length {
            let reason = format!("input exceeds maximum length of {}", self.config.max_length);
            record.reject(reason.clone());
            events.push(InputEvent::Rejected {
                input_id: record.id,
                reason,
                timestamp: Utc::now(),
            });
            return Ok(events);
        }

        // Deduplicate
        if self.config.deduplicate {
            let mut seen = self
                .seen_fingerprints
                .lock()
                .map_err(|_| InputAdmissionError::LockPoisoned)?;
            if !seen.insert(record.fingerprint) {
                record.reject("duplicate input".to_string());
                events.push(InputEvent::Rejected {
                    input_id: record.id,
                    reason: "duplicate input".to_string(),
                    timestamp: Utc::now(),
                });
                return Ok(events);
            }
        }

        // Admit
        record.update_state(InputState::Admitted);
        events.push(InputEvent::Admitted {
            input_id: record.id,
            timestamp: Utc::now(),
        });

        Ok(events)
    }

    /// Marks an input as processing.
    #[must_use]
    pub fn mark_processing(&self, record: &mut InputRecord) -> InputEvent {
        record.update_state(InputState::Processing);
        InputEvent::Processing {
            input_id: record.id,
            timestamp: Utc::now(),
        }
    }

    /// Marks an input as completed.
    #[must_use]
    pub fn mark_completed(&self, record: &mut InputRecord) -> InputEvent {
        record.update_state(InputState::Completed);
        InputEvent::Completed {
            input_id: record.id,
            timestamp: Utc::now(),
        }
    }

    /// Clears the deduplication cache.
    pub fn clear_cache(&self) -> Result<(), InputAdmissionError> {
        let mut seen = self
            .seen_fingerprints
            .lock()
            .map_err(|_| InputAdmissionError::LockPoisoned)?;
        seen.clear();
        Ok(())
    }
}

/// Error from input admission.
#[derive(Debug, Clone)]
pub enum InputAdmissionError {
    /// Lock was poisoned.
    LockPoisoned,
}

impl fmt::Display for InputAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LockPoisoned => write!(f, "input admission lock poisoned"),
        }
    }
}

impl std::error::Error for InputAdmissionError {}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn input_id_generation() {
        let id1 = InputId::new();
        let id2 = InputId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn input_state_terminal() {
        assert!(!InputState::Submitted.is_terminal());
        assert!(!InputState::Admitted.is_terminal());
        assert!(!InputState::Processing.is_terminal());
        assert!(InputState::Completed.is_terminal());
        assert!(InputState::Rejected.is_terminal());
    }

    #[test]
    fn input_record_creation() {
        let session_id = Uuid::new_v4();
        let record = InputRecord::new(session_id, "hello world".to_string());
        assert_eq!(record.state, InputState::Submitted);
        assert_eq!(record.content, "hello world");
        assert!(record.rejection_reason.is_none());
    }

    #[test]
    fn admission_admits_valid_input() {
        let config = InputAdmissionConfig::default();
        let admission = InputAdmission::new(config);
        let session_id = Uuid::new_v4();
        let mut record = InputRecord::new(session_id, "hello".to_string());

        let events = admission.admit(&mut record).unwrap();
        assert_eq!(record.state, InputState::Admitted);
        assert_eq!(events.len(), 2); // Submitted + Admitted
    }

    #[test]
    fn admission_rejects_empty_input() {
        let config = InputAdmissionConfig::default();
        let admission = InputAdmission::new(config);
        let session_id = Uuid::new_v4();
        let mut record = InputRecord::new(session_id, "   ".to_string());

        let events = admission.admit(&mut record).unwrap();
        assert_eq!(record.state, InputState::Rejected);
        assert_eq!(record.rejection_reason.as_deref(), Some("empty input"));
        assert_eq!(events.len(), 2); // Submitted + Rejected
    }

    #[test]
    fn admission_rejects_duplicate_input() {
        let config = InputAdmissionConfig::default();
        let admission = InputAdmission::new(config);
        let session_id = Uuid::new_v4();

        let mut record1 = InputRecord::new(session_id, "hello".to_string());
        let events1 = admission.admit(&mut record1).unwrap();
        assert_eq!(record1.state, InputState::Admitted);
        assert_eq!(events1.len(), 2);

        let mut record2 = InputRecord::new(session_id, "hello".to_string());
        let events2 = admission.admit(&mut record2).unwrap();
        assert_eq!(record2.state, InputState::Rejected);
        assert_eq!(record2.rejection_reason.as_deref(), Some("duplicate input"));
        assert_eq!(events2.len(), 2); // Submitted + Rejected
    }

    #[test]
    fn admission_rejects_over_length_input() {
        let config = InputAdmissionConfig::default().with_max_length(5);
        let admission = InputAdmission::new(config);
        let session_id = Uuid::new_v4();
        let mut record = InputRecord::new(session_id, "hello world".to_string());

        let events = admission.admit(&mut record).unwrap();
        assert_eq!(record.state, InputState::Rejected);
        assert!(
            record
                .rejection_reason
                .as_ref()
                .unwrap()
                .contains("maximum length")
        );
        assert_eq!(events.len(), 2); // Submitted + Rejected
    }

    #[test]
    fn admission_disabled_admits_all() {
        let config = InputAdmissionConfig::default().with_enabled(false);
        let admission = InputAdmission::new(config);
        let session_id = Uuid::new_v4();
        let mut record = InputRecord::new(session_id, "".to_string());

        let events = admission.admit(&mut record).unwrap();
        assert_eq!(record.state, InputState::Admitted);
        assert_eq!(events.len(), 1); // Only Admitted (no Submitted event when disabled)
    }

    #[test]
    fn admission_dedup_disabled_allows_duplicates() {
        let config = InputAdmissionConfig::default().with_deduplicate(false);
        let admission = InputAdmission::new(config);
        let session_id = Uuid::new_v4();

        let mut record1 = InputRecord::new(session_id, "hello".to_string());
        admission.admit(&mut record1).unwrap();

        let mut record2 = InputRecord::new(session_id, "hello".to_string());
        let events = admission.admit(&mut record2).unwrap();
        assert_eq!(record2.state, InputState::Admitted);
        assert_eq!(events.len(), 2); // Submitted + Admitted
    }

    #[test]
    fn mark_processing_and_completed() {
        let config = InputAdmissionConfig::default();
        let admission = InputAdmission::new(config);
        let session_id = Uuid::new_v4();
        let mut record = InputRecord::new(session_id, "hello".to_string());
        admission.admit(&mut record).unwrap();

        let event = admission.mark_processing(&mut record);
        assert_eq!(record.state, InputState::Processing);
        assert!(matches!(event, InputEvent::Processing { .. }));

        let event = admission.mark_completed(&mut record);
        assert_eq!(record.state, InputState::Completed);
        assert!(matches!(event, InputEvent::Completed { .. }));
    }

    #[test]
    fn clear_cache_allows_duplicate() {
        let config = InputAdmissionConfig::default();
        let admission = InputAdmission::new(config);
        let session_id = Uuid::new_v4();

        let mut record1 = InputRecord::new(session_id, "hello".to_string());
        admission.admit(&mut record1).unwrap();

        admission.clear_cache().unwrap();

        let mut record2 = InputRecord::new(session_id, "hello".to_string());
        let events = admission.admit(&mut record2).unwrap();
        assert_eq!(record2.state, InputState::Admitted);
        assert_eq!(events.len(), 2);
    }
}
