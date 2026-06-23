//! Tool output truncation with head+tail sampling.
//!
//! When tool output exceeds configured limits, the content is truncated
//! using a head+tail sampling strategy: the first half of allowed lines
//! and the last half are preserved, with a truncation marker in between.
//! Full output is written to a file for later inspection.
//!
//! Ported from OpenCode V2's `ToolOutputStore.bound()`.

use std::fmt::Write;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Maximum lines before truncation applies (default).
pub const DEFAULT_MAX_LINES: usize = 2_000;
/// Maximum bytes before truncation applies (default).
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024; // 50 KB

/// Configuration for tool output truncation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutputConfig {
    /// Maximum lines before truncation. `None` disables line-based truncation.
    pub max_lines: Option<usize>,
    /// Maximum bytes before truncation. `None` disables byte-based truncation.
    pub max_bytes: Option<usize>,
    /// Directory to save full truncated output files. `None` disables file saving.
    pub output_dir: Option<PathBuf>,
}

impl Default for ToolOutputConfig {
    fn default() -> Self {
        Self {
            max_lines: Some(DEFAULT_MAX_LINES),
            max_bytes: Some(DEFAULT_MAX_BYTES),
            output_dir: None,
        }
    }
}

/// Result of tool output truncation.
#[derive(Debug, Clone)]
pub struct TruncationResult {
    /// The truncated output text (or original if within limits).
    pub text: String,
    /// Whether truncation was applied.
    pub was_truncated: bool,
    /// Number of original lines (0 if unknown).
    pub original_lines: usize,
    /// Number of original bytes.
    pub original_bytes: usize,
    /// Path to the saved full output file, if any.
    pub saved_path: Option<PathBuf>,
}

/// Applies head+tail sampling truncation to tool output.
///
/// Strategy: preserves the first half and the last half of the allowed
/// line count, inserting a truncation marker in between. Falls back to
/// byte-based prefix/suffix if byte limits are stricter.
#[must_use]
pub fn truncate_output(
    output: &str,
    config: &ToolOutputConfig,
    identifier: Option<&str>,
) -> TruncationResult {
    let original_bytes = output.len();
    let lines: Vec<&str> = output.lines().collect();
    let original_lines = lines.len();

    let max_lines = config.max_lines.unwrap_or(usize::MAX);
    let max_bytes = config.max_bytes.unwrap_or(usize::MAX);

    // Check if truncation is needed
    if original_lines <= max_lines && original_bytes <= max_bytes {
        return TruncationResult {
            text: output.to_owned(),
            was_truncated: false,
            original_lines,
            original_bytes,
            saved_path: None,
        };
    }

    // Determine line budget
    let line_budget = max_lines.min(original_lines);
    // Minimum 4 lines to apply head+tail sampling
    let result_text = if line_budget > 4 {
        let head_lines = line_budget.div_ceil(2);
        let tail_lines = line_budget / 2; // floor division

        let head: Vec<&str> = lines.iter().take(head_lines).copied().collect();
        let tail: Vec<&str> = lines
            .iter()
            .rev()
            .take(tail_lines)
            .copied()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let marker = truncation_marker(
            original_lines,
            original_bytes,
            head_lines + 1..original_lines - tail_lines,
            identifier,
            config.output_dir.as_ref(),
        );

        let mut sampled = head.join("\n");
        sampled.push('\n');
        sampled.push_str(&marker);
        if !tail.is_empty() {
            sampled.push('\n');
            sampled.push_str(&tail.join("\n"));
        }
        sampled
    } else {
        // Too few lines for head+tail — just truncate
        let prefix: String = output.chars().take(max_bytes.min(original_bytes)).collect();
        let marker = truncation_marker(
            original_lines,
            original_bytes,
            0..original_lines,
            identifier,
            config.output_dir.as_ref(),
        );
        format!("{prefix}\n{marker}")
    };

    // Apply byte limit to the result
    let result_text = if result_text.len() > max_bytes {
        let head_bytes = max_bytes * 3 / 4;
        let tail_bytes = max_bytes - head_bytes;
        let head: String = result_text.chars().take(head_bytes).collect();
        let tail: String = result_text
            .chars()
            .rev()
            .take(tail_bytes)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        let marker = format!(
            "\n... output truncated ({original_lines} lines, {original_bytes} bytes total) ...\n"
        );
        format!("{head}{marker}{tail}")
    } else {
        result_text
    };

    // Save full output to file
    let saved_path = if let (Some(dir), Some(id)) = (&config.output_dir, identifier) {
        let path = dir.join(format!("tool_{id}"));
        if let Err(e) = std::fs::create_dir_all(dir) {
            tracing::warn!(error = %e, "failed to create tool output directory");
            None
        } else if let Err(e) = std::fs::write(&path, output) {
            tracing::warn!(error = %e, "failed to write full tool output");
            None
        } else {
            Some(path)
        }
    } else {
        None
    };

    TruncationResult {
        text: result_text,
        was_truncated: true,
        original_lines,
        original_bytes,
        saved_path,
    }
}

/// Generates the truncation marker text.
fn truncation_marker(
    total_lines: usize,
    total_bytes: usize,
    _omitted_range: std::ops::Range<usize>,
    identifier: Option<&str>,
    output_dir: Option<&PathBuf>,
) -> String {
    let omitted_lines = total_lines.saturating_sub(if total_lines > 4 {
        total_lines.min(DEFAULT_MAX_LINES).div_ceil(2) * 2
    } else {
        total_lines.min(1)
    });
    let mut marker = format!(
        "\n... output truncated ({omitted_lines} lines omitted, {total_bytes} bytes total) ..."
    );

    if let (Some(dir), Some(id)) = (output_dir, identifier) {
        let path = dir.join(format!("tool_{id}"));
        let _ = write!(
            marker,
            "\nFull output saved to: {path}",
            path = path.display()
        );
    }

    marker
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn no_truncation_when_within_limits() {
        let config = ToolOutputConfig::default();
        let output = "line1\nline2\nline3";
        let result = truncate_output(output, &config, None);
        assert!(!result.was_truncated);
        assert_eq!(result.text, output);
    }

    #[test]
    fn truncates_by_lines() {
        let config = ToolOutputConfig {
            max_lines: Some(6),
            max_bytes: None,
            output_dir: None,
        };
        let lines: Vec<String> = (0..100).map(|i| format!("line{i}")).collect();
        let output = lines.join("\n");
        let result = truncate_output(&output, &config, None);
        assert!(result.was_truncated);

        // Should have head+tail with marker
        let result_lines: Vec<&str> = result.text.lines().collect();
        assert!(result_lines.len() < 20, "should be significantly truncated");

        // First and last lines should be preserved
        assert!(result.text.contains("line0"));
        assert!(result.text.contains("line99"));
    }

    #[test]
    fn truncates_by_bytes() {
        let config = ToolOutputConfig {
            max_lines: None,
            max_bytes: Some(100),
            output_dir: None,
        };
        let output = "x".repeat(10_000);
        let result = truncate_output(&output, &config, None);
        assert!(result.was_truncated);
        assert!(
            result.text.len() <= 200,
            "result should be near the byte limit"
        );
    }

    #[test]
    fn includes_truncation_marker() {
        let config = ToolOutputConfig {
            max_lines: Some(6),
            max_bytes: None,
            output_dir: None,
        };
        let lines: Vec<String> = (0..100).map(|i| format!("line{i}")).collect();
        let output = lines.join("\n");
        let result = truncate_output(&output, &config, None);
        assert!(result.text.contains("output truncated"));
    }

    #[test]
    fn empty_output_passes_through() {
        let config = ToolOutputConfig::default();
        let result = truncate_output("", &config, None);
        assert!(!result.was_truncated);
        assert!(result.text.is_empty());
    }

    #[test]
    fn single_line_within_limit() {
        let config = ToolOutputConfig::default();
        let output = "single line";
        let result = truncate_output(output, &config, None);
        assert!(!result.was_truncated);
        assert_eq!(result.text, output);
    }
}
