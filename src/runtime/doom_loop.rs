//! Doom loop detection for agent runs.
//!
//! Detects when an agent is stuck in repetitive tool call patterns:
//! - Consecutive duplicate tool calls (same tool, same arguments)
//! - Cyclic patterns (repeating sequences of tool calls)

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Configuration for doom loop detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoomLoopConfig {
    /// Enable doom loop detection. Default: `true`.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Number of consecutive identical tool calls to trigger detection. Default: `3`.
    #[serde(default = "default_consecutive_threshold")]
    pub consecutive_threshold: usize,
    /// Minimum cycle length to detect. Default: `2`.
    #[serde(default = "default_min_cycle_length")]
    pub min_cycle_length: usize,
    /// Maximum cycle length to detect. Default: `4`.
    #[serde(default = "default_max_cycle_length")]
    pub max_cycle_length: usize,
    /// Number of cycle repetitions to trigger detection. Default: `2`.
    #[serde(default = "default_cycle_repetitions")]
    pub cycle_repetitions: usize,
}

const fn default_true() -> bool {
    true
}

const fn default_consecutive_threshold() -> usize {
    3
}

const fn default_min_cycle_length() -> usize {
    2
}

const fn default_max_cycle_length() -> usize {
    4
}

const fn default_cycle_repetitions() -> usize {
    2
}

impl Default for DoomLoopConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            consecutive_threshold: 3,
            min_cycle_length: 2,
            max_cycle_length: 4,
            cycle_repetitions: 2,
        }
    }
}

impl DoomLoopConfig {
    /// Creates a new config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Disables doom loop detection.
    #[must_use]
    pub fn with_disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Sets the consecutive duplicate threshold.
    #[must_use]
    pub fn with_consecutive_threshold(mut self, threshold: usize) -> Self {
        self.consecutive_threshold = threshold;
        self
    }

    /// Sets the minimum cycle length.
    #[must_use]
    pub fn with_min_cycle_length(mut self, length: usize) -> Self {
        self.min_cycle_length = length;
        self
    }

    /// Sets the maximum cycle length.
    #[must_use]
    pub fn with_max_cycle_length(mut self, length: usize) -> Self {
        self.max_cycle_length = length;
        self
    }

    /// Sets the cycle repetition count.
    #[must_use]
    pub fn with_cycle_repetitions(mut self, repetitions: usize) -> Self {
        self.cycle_repetitions = repetitions;
        self
    }
}

/// A fingerprint representing a unique tool call.
///
/// Combines tool name and canonicalized arguments into a hash
/// for efficient comparison and pattern detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ToolCallFingerprint(u64);

impl ToolCallFingerprint {
    /// Creates a fingerprint from a tool name and arguments.
    #[must_use]
    pub fn from_tool_call(name: &str, arguments: &Value) -> Self {
        let mut hasher = DefaultHasher::new();

        name.hash(&mut hasher);

        // Canonicalize arguments by sorting object keys
        let canonical = Self::canonicalize(arguments);
        canonical.hash(&mut hasher);

        Self(hasher.finish())
    }

    /// Canonicalizes a JSON value by sorting object keys recursively.
    fn canonicalize(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut sorted: Vec<_> = map.iter().collect();
                sorted.sort_by(|a, b| a.0.cmp(b.0));
                let canonical_map: serde_json::Map<String, Value> = sorted
                    .into_iter()
                    .map(|(k, v)| (k.clone(), Self::canonicalize(v)))
                    .collect();
                Value::Object(canonical_map)
            }
            Value::Array(arr) => {
                Value::Array(arr.iter().map(Self::canonicalize).collect())
            }
            other => other.clone(),
        }
    }
}

/// The type of doom loop detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DoomLoopType {
    /// Consecutive identical tool calls.
    ConsecutiveDuplicate {
        /// The tool name being repeated.
        tool_name: String,
        /// Number of consecutive occurrences.
        count: usize,
    },
    /// A cyclic pattern of tool calls.
    Cycle {
        /// The sequence of tool names forming the cycle.
        pattern: Vec<String>,
        /// Number of times the cycle repeated.
        repetitions: usize,
    },
}

/// Detects doom loops in agent tool call sequences.
#[derive(Debug)]
pub struct DoomLoopDetector {
    config: DoomLoopConfig,
    history: Vec<(ToolCallFingerprint, String)>,
}

impl DoomLoopDetector {
    /// Creates a new detector with the given configuration.
    #[must_use]
    pub fn new(config: DoomLoopConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Records a tool call and checks for doom loops.
    ///
    /// Returns `Some(DoomLoopType)` if a doom loop is detected, `None` otherwise.
    pub fn record_and_check(&mut self, name: &str, arguments: &Value) -> Option<DoomLoopType> {
        if !self.config.enabled {
            return None;
        }

        let fingerprint = ToolCallFingerprint::from_tool_call(name, arguments);
        self.history.push((fingerprint, name.to_string()));

        // Check for consecutive duplicates
        if let Some(result) = self.check_consecutive_duplicate() {
            return Some(result);
        }

        // Check for cycles
        self.check_cycle()
    }

    /// Checks if the last N tool calls are identical.
    fn check_consecutive_duplicate(&self) -> Option<DoomLoopType> {
        let threshold = self.config.consecutive_threshold;
        if self.history.len() < threshold {
            return None;
        }

        let last = self.history.last()?;
        let all_same = self
            .history
            .iter()
            .rev()
            .take(threshold)
            .all(|(fp, _)| fp == &last.0);

        if all_same {
            Some(DoomLoopType::ConsecutiveDuplicate {
                tool_name: last.1.clone(),
                count: threshold,
            })
        } else {
            None
        }
    }

    /// Checks for cyclic patterns in the tool call history.
    fn check_cycle(&self) -> Option<DoomLoopType> {
        let min_len = self.config.min_cycle_length;
        let max_len = self.config.max_cycle_length;
        let repetitions = self.config.cycle_repetitions;

        let history_len = self.history.len();
        let min_required = min_len * (repetitions + 1);

        if history_len < min_required {
            return None;
        }

        // Try different cycle lengths
        for cycle_len in min_len..=max_len {
            let required_len = cycle_len * (repetitions + 1);
            if history_len < required_len {
                continue;
            }

            // Extract the candidate pattern from the end
            let pattern_start = history_len - required_len;
            let candidate: Vec<_> = self.history[pattern_start..pattern_start + cycle_len]
                .iter()
                .map(|(fp, _)| *fp)
                .collect();

            // Check if this pattern repeats
            let mut matches = true;
            for rep in 1..=repetitions {
                let rep_start = pattern_start + rep * cycle_len;
                for i in 0..cycle_len {
                    if self.history[rep_start + i].0 != candidate[i] {
                        matches = false;
                        break;
                    }
                }
                if !matches {
                    break;
                }
            }

            if matches {
                let pattern_names: Vec<String> = self.history[pattern_start..pattern_start + cycle_len]
                    .iter()
                    .map(|(_, name)| name.clone())
                    .collect();

                return Some(DoomLoopType::Cycle {
                    pattern: pattern_names,
                    repetitions: repetitions + 1,
                });
            }
        }

        None
    }

    /// Clears the detection history.
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Returns the current history length.
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fingerprint_same_tool_same_args() {
        let fp1 = ToolCallFingerprint::from_tool_call("read", &json!({"path": "/foo"}));
        let fp2 = ToolCallFingerprint::from_tool_call("read", &json!({"path": "/foo"}));
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn fingerprint_same_tool_different_args() {
        let fp1 = ToolCallFingerprint::from_tool_call("read", &json!({"path": "/foo"}));
        let fp2 = ToolCallFingerprint::from_tool_call("read", &json!({"path": "/bar"}));
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn fingerprint_different_tools() {
        let fp1 = ToolCallFingerprint::from_tool_call("read", &json!({"path": "/foo"}));
        let fp2 = ToolCallFingerprint::from_tool_call("write", &json!({"path": "/foo"}));
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn fingerprint_canonicalizes_object_keys() {
        let fp1 = ToolCallFingerprint::from_tool_call("test", &json!({"a": 1, "b": 2}));
        let fp2 = ToolCallFingerprint::from_tool_call("test", &json!({"b": 2, "a": 1}));
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn detect_consecutive_duplicate() {
        let config = DoomLoopConfig::new().with_consecutive_threshold(3);
        let mut detector = DoomLoopDetector::new(config);

        assert!(detector.record_and_check("read", &json!({"path": "/foo"})).is_none());
        assert!(detector.record_and_check("read", &json!({"path": "/foo"})).is_none());

        let result = detector.record_and_check("read", &json!({"path": "/foo"}));
        assert!(matches!(
            result,
            Some(DoomLoopType::ConsecutiveDuplicate { tool_name, count })
            if tool_name == "read" && count == 3
        ));
    }

    #[test]
    fn no_false_positive_on_different_args() {
        let config = DoomLoopConfig::new().with_consecutive_threshold(3);
        let mut detector = DoomLoopDetector::new(config);

        assert!(detector.record_and_check("read", &json!({"path": "/foo"})).is_none());
        assert!(detector.record_and_check("read", &json!({"path": "/bar"})).is_none());
        assert!(detector.record_and_check("read", &json!({"path": "/foo"})).is_none());
    }

    #[test]
    fn detect_cycle() {
        let config = DoomLoopConfig::new()
            .with_min_cycle_length(2)
            .with_max_cycle_length(2)
            .with_cycle_repetitions(2);
        let mut detector = DoomLoopDetector::new(config);

        // Pattern: read -> write -> read -> write -> read -> write
        assert!(detector.record_and_check("read", &json!({})).is_none());
        assert!(detector.record_and_check("write", &json!({})).is_none());
        assert!(detector.record_and_check("read", &json!({})).is_none());
        assert!(detector.record_and_check("write", &json!({})).is_none());
        assert!(detector.record_and_check("read", &json!({})).is_none());

        let result = detector.record_and_check("write", &json!({}));
        assert!(matches!(
            result,
            Some(DoomLoopType::Cycle { pattern, repetitions })
            if pattern == vec!["read", "write"] && repetitions == 3
        ));
    }

    #[test]
    fn disabled_detection_returns_none() {
        let config = DoomLoopConfig::new().with_disabled();
        let mut detector = DoomLoopDetector::new(config);

        for _ in 0..10 {
            assert!(detector.record_and_check("read", &json!({})).is_none());
        }
    }

    #[test]
    fn clear_resets_history() {
        let config = DoomLoopConfig::new();
        let mut detector = DoomLoopDetector::new(config);

        detector.record_and_check("read", &json!({}));
        detector.record_and_check("write", &json!({}));
        assert_eq!(detector.history_len(), 2);

        detector.clear();
        assert_eq!(detector.history_len(), 0);
    }
}
