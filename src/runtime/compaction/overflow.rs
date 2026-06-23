//! Compact overflow detection.
//!
//! Determines when the conversation context is approaching or exceeding
//! the model's context window, triggering compaction before or after
//! a provider turn.
//!
//! Ported from OpenCode V1: `packages/opencode/src/session/overflow.ts`.

/// Token headroom reserved between context limit and compaction trigger.
///
/// This ensures the compaction LLM call itself (prompt + response) fits
/// without overflowing. OpenCode uses 20,000 tokens.
pub const COMPACTION_BUFFER: usize = 20_000;

/// Computes the number of tokens usable for conversation history.
///
/// The usable space is the model's context window minus:
/// 1. The maximum output tokens the model can generate
/// 2. Optional reserved headroom (defaults to `min(COMPACTION_BUFFER, max_output)`)
///
/// When `model_context` is 0 (unlimited), returns `usize::MAX`.
#[must_use]
pub fn usable_tokens(model_context: u32, max_output: u32, reserved: Option<usize>) -> usize {
    if model_context == 0 {
        return usize::MAX;
    }

    let reserved = reserved.unwrap_or_else(|| COMPACTION_BUFFER.min(max_output as usize));
    let context = model_context as usize;

    context
        .saturating_sub(max_output as usize)
        .saturating_sub(reserved)
}

/// Returns `true` when the conversation history exceeds the usable context window.
///
/// Compaction is skipped when:
/// - `auto_enabled` is `false`
/// - `model_context` is 0 (no context limit)
///
/// # Arguments
/// * `total_tokens` - Estimated tokens in the full message history.
/// * `model_context` - The model's maximum context window (0 = unlimited).
/// * `max_output` - The model's maximum output tokens.
/// * `auto_enabled` - Whether automatic compaction is enabled.
#[must_use]
pub fn is_overflow(
    total_tokens: usize,
    model_context: u32,
    max_output: u32,
    auto_enabled: bool,
) -> bool {
    if !auto_enabled || model_context == 0 {
        return false;
    }

    let usable = usable_tokens(model_context, max_output, None);
    total_tokens >= usable
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn usable_with_typical_gpt4o() {
        // gpt-4o: 128K context, 16K output
        let usable = usable_tokens(128_000, 16_384, None);
        // 128_000 - 16_384 - min(20_000, 16_384) = 128_000 - 16_384 - 16_384 = 95_232
        assert_eq!(usable, 95_232);
    }

    #[test]
    fn usable_with_large_output() {
        // Model with 200K context, 32K output
        let usable = usable_tokens(200_000, 32_000, None);
        // 200_000 - 32_000 - min(20_000, 32_000) = 200_000 - 32_000 - 20_000 = 148_000
        assert_eq!(usable, 148_000);
    }

    #[test]
    fn usable_with_small_output() {
        // Model with 8K context, 4K output
        let usable = usable_tokens(8_000, 4_000, None);
        // 8_000 - 4_000 - min(20_000, 4_000) = 8_000 - 4_000 - 4_000 = 0
        assert_eq!(usable, 0);
    }

    #[test]
    fn usable_unlimited_context() {
        assert_eq!(usable_tokens(0, 16_384, None), usize::MAX);
    }

    #[test]
    fn usable_with_explicit_reserved() {
        let usable = usable_tokens(128_000, 16_384, Some(10_000));
        // 128_000 - 16_384 - 10_000 = 101_616
        assert_eq!(usable, 101_616);
    }

    #[test]
    fn overflow_detected() {
        // gpt-4o: usable = 95_232
        assert!(!is_overflow(90_000, 128_000, 16_384, true));
        assert!(is_overflow(96_000, 128_000, 16_384, true));
    }

    #[test]
    fn overflow_skipped_when_auto_disabled() {
        assert!(!is_overflow(200_000, 128_000, 16_384, false));
    }

    #[test]
    fn overflow_skipped_when_unlimited_context() {
        assert!(!is_overflow(1_000_000, 0, 16_384, true));
    }
}
