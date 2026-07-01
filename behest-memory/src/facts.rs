//! Memory fact extraction — pure detection of retainable user input.
//!
//! [`detect_facts`] is a pure function: input text → list of facts. It does
//! not store anything, perform no I/O, and makes no decisions about whether
//! to persist. The caller decides which facts to keep.
//!
//! # Patterns
//!
//! Detection uses 10 regex patterns covering common "remember this" phrasings:
//!
//! | Pattern | Example |
//! |---------|---------|
//! | `please remember that ...` | "please remember that the API is at v2" |
//! | `note that ...` | "note that the port is 5432" |
//! | `for future reference: ...` | "for future reference: the DB is postgres" |
//! | `I prefer ...` / `my preferred ...` | "I prefer tabs over spaces" |
//! | `my project ...` | "my project uses Rust 1.88" |
//! | `I always ...` / `I never ...` | "I always run tests before commit" |
//! | `we use ...` | "we use tokio for async" |
//! | `our ... is/are ...` | "our deploy target is Linux" |
//! | `the api/server/database/port/host/endpoint ...` | "the database is on port 5432" |
//!
//! # Categories
//!
//! Each detected fact is categorized by keyword matching:
//! - [`Preference`](FactCategory::Preference) — user tastes/habits
//! - [`Environment`](FactCategory::Environment) — ports, hosts, config, URLs
//! - [`Solution`](FactCategory::Solution) — fixes, workarounds, bugs
//! - [`Pattern`](FactCategory::Pattern) — conventions, standards, practices
//! - [`Other`](FactCategory::Other) — fallback
//!
//! # Example
//!
//! ```ignore
//! use behest_memory::facts::detect_facts;
//!
//! let input = "please remember that I prefer tabs, and note that the port is 5432";
//! let facts = detect_facts(input);
//! assert_eq!(facts.len(), 2);
//! assert!(facts.iter().any(|f| f.category == behest_memory::facts::FactCategory::Preference));
//! assert!(facts.iter().any(|f| f.category == behest_memory::facts::FactCategory::Environment));
//! ```

// Static regex patterns below are compile-time-validated constants; their
// `unwrap()` calls cannot fail at runtime.
#![allow(clippy::unwrap_used)]

use serde::{Deserialize, Serialize};

/// Category assigned to a detected fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactCategory {
    /// User tastes, habits, preferences ("I prefer ...").
    Preference,
    /// Ports, hosts, configs, URLs, endpoints.
    Environment,
    /// Fixes, workarounds, bugs, solutions.
    Solution,
    /// Conventions, standards, practices.
    Pattern,
    /// Anything else that matched a "remember" pattern but doesn't fit above.
    Other,
}

impl std::fmt::Display for FactCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preference => write!(f, "preference"),
            Self::Environment => write!(f, "environment"),
            Self::Solution => write!(f, "solution"),
            Self::Pattern => write!(f, "pattern"),
            Self::Other => write!(f, "other"),
        }
    }
}

/// A single detected retainable fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedFact {
    /// The fact text, trimmed and normalized.
    pub content: String,
    /// Category assigned by keyword matching.
    pub category: FactCategory,
    /// Confidence score (0.0–1.0). Higher = stronger pattern match.
    pub confidence: f32,
    /// Which detection pattern triggered (human-readable).
    pub matched_pattern: String,
}

/// Minimum and maximum length for a retained fact.
const MIN_FACT_LEN: usize = 8;
const MAX_FACT_LEN: usize = 240;

/// Detects retainable facts in the given user input.
///
/// Pure function: no I/O, no storage, no side effects. Returns a list of
/// [`ExtractedFact`] values. The caller decides which (if any) to persist.
///
/// Overlapping matches are resolved by priority: more specific patterns
/// (like "please remember that") take precedence over shorter ones
/// (like "remember that"). After a match is accepted, any later match
/// whose span overlaps it is skipped.
#[must_use]
pub fn detect_facts(text: &str) -> Vec<ExtractedFact> {
    // Collect all captures with their byte spans.
    // Each pattern has one capture group: the fact body after the trigger.
    // We also keep the full match text (with trigger) for categorization,
    // since the trigger word itself ("prefer", "always", "port") is a strong
    // category signal.
    let mut candidates: Vec<(usize, usize, String, String, &'static str)> = Vec::new();

    for (pattern_name, regex) in DETECTION_PATTERNS {
        for caps in regex.captures_iter(text) {
            if let Some(group) = caps.get(1) {
                candidates.push((
                    group.start(),
                    group.end(),
                    caps.get(0).unwrap_or(group).as_str().to_string(),
                    group.as_str().to_string(),
                    pattern_name,
                ));
            }
        }
    }

    // Sort by (start asc, length desc) so earlier, longer matches win.
    candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));

    let mut accepted_spans: Vec<(usize, usize)> = Vec::new();
    let mut facts: Vec<ExtractedFact> = Vec::new();

    for (start, end, full_match, body, pattern_name) in candidates {
        // Skip if this match overlaps any accepted span
        let overlaps = accepted_spans.iter().any(|(s, e)| start < *e && end > *s);
        if overlaps {
            continue;
        }

        let content = body.trim().trim_end_matches('.').trim().to_string();

        let len = content.chars().count();
        if (MIN_FACT_LEN..=MAX_FACT_LEN).contains(&len) {
            // Categorize using the full match (includes trigger words like
            // "prefer", "always", "port") for stronger signals.
            let category = categorize(&full_match);
            let confidence = confidence_for(pattern_name, &content);
            facts.push(ExtractedFact {
                content,
                category,
                confidence,
                matched_pattern: pattern_name.to_string(),
            });
            accepted_spans.push((start, end));
        }
    }

    // Deduplicate by content (case-insensitive)
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    facts.retain(|f| {
        let key = f.content.to_lowercase();
        seen.insert(key)
    });

    facts
}

/// Categorizes a fact by keyword matching.
///
/// Categories are checked in order of specificity: Solution and Pattern
/// before Environment, because phrases like "fix the server" contain both
/// a solution keyword ("fix") and an environment keyword ("server").
fn categorize(content: &str) -> FactCategory {
    let lower = content.to_lowercase();
    // Preference first — explicit user-taste triggers.
    if lower.contains("prefer")
        || lower.contains("like")
        || lower.contains("always")
        || lower.contains("never")
        || lower.contains("favorite")
    {
        return FactCategory::Preference;
    }
    // Solution before Environment — "fix the server" is a solution, not env.
    if lower.contains("fix")
        || lower.contains("solve")
        || lower.contains("error")
        || lower.contains("bug")
        || lower.contains("workaround")
    {
        return FactCategory::Solution;
    }
    // Pattern before Environment — "convention for the database" is a pattern.
    if lower.contains("pattern")
        || lower.contains("convention")
        || lower.contains("standard")
        || lower.contains("practice")
    {
        return FactCategory::Pattern;
    }
    if lower.contains("port")
        || lower.contains("host")
        || lower.contains("server")
        || lower.contains("database")
        || lower.contains("db")
        || lower.contains("config")
        || lower.contains("url")
        || lower.contains("token")
        || lower.contains("endpoint")
        || lower.contains("api")
    {
        return FactCategory::Environment;
    }
    FactCategory::Other
}

/// Confidence score based on the pattern and content.
fn confidence_for(pattern_name: &str, content: &str) -> f32 {
    // Explicit "remember" patterns are high confidence.
    let base = match pattern_name {
        "please_remember" | "remember_that" | "for_future_reference" => 0.95,
        "note_that" => 0.85,
        "i_prefer" | "my_preferred" | "i_always" | "i_never" => 0.80,
        "my_project" | "we_use" | "our_is" => 0.70,
        "the_x" => 0.60,
        _ => 0.50,
    };
    // Longer, more specific facts are slightly more confident.
    let len_bonus = (content.chars().count() as f32 / 240.0).min(0.05);
    base + len_bonus
}

// ── Detection patterns ──
//
// Each pattern is a static regex compiled once on first use. Pattern strings
// are compile-time-validated constants, so the lazy init cannot fail.

use regex::Regex;
use std::sync::LazyLock;

static DETECTION_PATTERNS: &[(&str, &LazyLock<Regex>)] = &[
    ("please_remember", &PLEASE_REMEMBER),
    ("remember_that", &REMEMBER_THAT),
    ("note_that", &NOTE_THAT),
    ("for_future_reference", &FOR_FUTURE_REFERENCE),
    ("i_prefer", &I_PREFER),
    ("my_preferred", &MY_PREFERRED),
    ("my_project", &MY_PROJECT),
    ("i_always", &I_ALWAYS),
    ("i_never", &I_NEVER),
    ("we_use", &WE_USE),
    ("our_is", &OUR_IS),
    ("the_x", &THE_X),
];

static PLEASE_REMEMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)please\s+remember\s+that\s+([^.!?]+)").unwrap());
static REMEMBER_THAT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bremember\s+that\s+([^.!?]+)").unwrap());
static NOTE_THAT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bnote\s+that\s+([^.!?]+)").unwrap());
static FOR_FUTURE_REFERENCE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)for\s+future\s+reference\s*:?\s*([^.!?]+)").unwrap());
static I_PREFER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bI\s+prefer\s+([^.!?]+)").unwrap());
static MY_PREFERRED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bmy\s+preferred\s+([^.!?]+)").unwrap());
static MY_PROJECT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bmy\s+project\s+([^.!?]+)").unwrap());
static I_ALWAYS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bI\s+always\s+([^.!?]+)").unwrap());
static I_NEVER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bI\s+never\s+([^.!?]+)").unwrap());
static WE_USE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bwe\s+use\s+([^.!?]+)").unwrap());
static OUR_IS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bour\s+\w+\s+(?:is|are)\s+([^.!?]+)").unwrap());
static THE_X: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\bthe\s+(?:api|server|database|port|host|endpoint)\s+(?:is|are)?\s*:?\s*([^.!?]+)",
    )
    .unwrap()
});

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn detects_please_remember() {
        let facts = detect_facts("please remember that the API is at v2");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].category, FactCategory::Environment);
        assert!(facts[0].confidence >= 0.95);
        assert_eq!(facts[0].matched_pattern, "please_remember");
    }

    #[test]
    fn detects_note_that() {
        let facts = detect_facts("note that the port is 5432");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].category, FactCategory::Environment);
        assert!(facts[0].confidence >= 0.85);
    }

    #[test]
    fn detects_for_future_reference() {
        let facts = detect_facts("for future reference: the DB is postgres");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].category, FactCategory::Environment);
    }

    #[test]
    fn detects_i_prefer() {
        let facts = detect_facts("I prefer tabs over spaces");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].category, FactCategory::Preference);
    }

    #[test]
    fn detects_i_always_and_i_never() {
        let always = detect_facts("I always run tests before commit");
        assert!(
            always
                .iter()
                .any(|f| f.category == FactCategory::Preference)
        );

        let never = detect_facts("I never push to main directly");
        assert!(never.iter().any(|f| f.category == FactCategory::Preference));
    }

    #[test]
    fn detects_we_use() {
        let facts = detect_facts("we use tokio for async runtime");
        assert_eq!(facts.len(), 1);
        // "use" doesn't match any category keyword → Other
        assert_eq!(facts[0].category, FactCategory::Other);
    }

    #[test]
    fn detects_the_x_pattern() {
        let facts = detect_facts("the database is on port 5432");
        assert!(!facts.is_empty());
        assert!(
            facts
                .iter()
                .any(|f| f.category == FactCategory::Environment)
        );
    }

    #[test]
    fn detects_multiple_facts_in_one_input() {
        // Two separate sentences so each pattern captures independently.
        let input = "please remember that I prefer tabs. note that the port is 5432";
        let facts = detect_facts(input);
        assert_eq!(facts.len(), 2);
        assert!(facts.iter().any(|f| f.category == FactCategory::Preference));
        assert!(
            facts
                .iter()
                .any(|f| f.category == FactCategory::Environment)
        );
    }

    #[test]
    fn deduplicates_case_insensitive() {
        let input = "I prefer tabs. I prefer tabs.";
        // Note: the regex captures to the first sentence end, so we get one
        // fact from "I prefer tabs" — the second occurrence is a separate
        // match but should be deduplicated.
        let facts = detect_facts(input);
        let contents: Vec<&str> = facts.iter().map(|f| f.content.as_str()).collect();
        // At most one "I prefer tabs" fact
        let tab_facts: Vec<_> = contents
            .into_iter()
            .filter(|c| c.to_lowercase().contains("tabs"))
            .collect();
        assert!(tab_facts.len() <= 1);
    }

    #[test]
    fn no_facts_for_plain_text() {
        let facts = detect_facts("hello world, how are you today?");
        assert!(facts.is_empty());
    }

    #[test]
    fn rejects_too_short_facts() {
        // "I prefer a" is too short (content "a" < MIN_FACT_LEN=8)
        let facts = detect_facts("I prefer a");
        assert!(facts.is_empty());
    }

    #[test]
    fn rejects_too_long_facts() {
        let long_body = "x".repeat(300);
        let input = format!("please remember that {long_body}");
        let facts = detect_facts(&input);
        assert!(facts.is_empty());
    }

    #[test]
    fn categorizes_solution_keywords() {
        let facts = detect_facts("please remember that the fix is to restart the server");
        assert!(facts.iter().any(|f| f.category == FactCategory::Solution));
    }

    #[test]
    fn categorizes_pattern_keywords() {
        let facts = detect_facts("please remember that our convention is to use snake_case");
        assert!(facts.iter().any(|f| f.category == FactCategory::Pattern));
    }

    #[test]
    fn confidence_increases_with_length() {
        let short = detect_facts("I prefer tabs over spaces");
        let long = detect_facts("I prefer using tabs over spaces for all my rust code projects");
        assert!(!short.is_empty());
        assert!(!long.is_empty());
        // Longer fact should have >= confidence due to length bonus
        assert!(long[0].confidence >= short[0].confidence);
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(detect_facts("").is_empty());
    }

    #[test]
    fn category_display() {
        assert_eq!(FactCategory::Preference.to_string(), "preference");
        assert_eq!(FactCategory::Environment.to_string(), "environment");
        assert_eq!(FactCategory::Solution.to_string(), "solution");
        assert_eq!(FactCategory::Pattern.to_string(), "pattern");
        assert_eq!(FactCategory::Other.to_string(), "other");
    }
}
