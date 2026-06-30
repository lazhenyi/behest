//! Context compaction — LLM-driven conversation summarisation.
//!
//! When a conversation approaches the model's context window, older
//! messages are summarised by a smaller/cheaper "compaction" LLM call
//! into a structured anchored summary. This frees context tokens while
//! preserving the semantic continuity of the conversation.
//!
//! # Sub-modules
//!
//! - [`select`] — Turn-based message selection (head vs tail).
//! - [`prompt`] — Anchored summary prompt template.
//! - [`overflow`] — Context overflow detection.
//! - [`prune`] — Old tool output pruning.
//!
//! # Architecture
//!
//! ```text
//! compact_if_needed() / compact_after_overflow()
//!   ├── overflow::is_overflow()      — detect if compaction is needed
//!   ├── select::select()             — split messages into head/tail
//!   ├── prompt::build_prompt()       — construct anchored summary prompt
//!   └── run_compaction()             — call compaction LLM
//!       └── store compaction result  — persist compaction messages
//! ```
//!
//! Ported from OpenCode V1/V2 compaction infrastructure.

pub mod overflow;
pub mod prompt;
pub mod prune;
pub mod select;

use std::sync::Arc;

use uuid::Uuid;

use crate::provider::{ChatProvider, ChatRequest, Message, ModelName};
use crate::store::{CompactionMeta, MessageRecord, MessageRole, SessionStore};
use crate::token::estimate_records_tokens;

use super::error::{RuntimeError, RuntimeResult};

/// Circuit breaker that gates compaction calls after repeated failures.
///
/// When consecutive compaction failures reach the configured threshold,
/// the breaker opens and all proactive compaction is skipped for the
/// remainder of the run. Reactive compaction (triggered by a provider
/// context overflow error) is still attempted.
#[derive(Debug, Clone)]
pub struct CompactionCircuitBreaker {
    consecutive_failures: u32,
    threshold: u32,
    state: BreakerState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BreakerState {
    Closed,
    Open {
        opened_at: chrono::DateTime<chrono::Utc>,
    },
}

impl CompactionCircuitBreaker {
    /// Creates a new breaker with the given failure threshold.
    #[must_use]
    pub fn new(threshold: u32) -> Self {
        Self {
            consecutive_failures: 0,
            threshold,
            state: BreakerState::Closed,
        }
    }

    /// Records a successful compaction, resetting the failure count and
    /// closing the breaker.
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.state = BreakerState::Closed;
    }

    /// Records a failed compaction. If the failure count reaches the
    /// threshold, the breaker opens. Returns `true` when the breaker
    /// transitions from closed to open (so the caller can emit an event).
    pub fn record_failure(&mut self) -> bool {
        if self.is_open() {
            return false;
        }
        self.consecutive_failures += 1;
        if self.consecutive_failures >= self.threshold {
            self.state = BreakerState::Open {
                opened_at: chrono::Utc::now(),
            };
            return true;
        }
        false
    }

    /// Returns `true` when the breaker is open and proactive compaction
    /// should be skipped.
    #[must_use]
    pub fn is_open(&self) -> bool {
        matches!(self.state, BreakerState::Open { .. })
    }

    /// Returns the number of consecutive failures recorded so far.
    #[must_use]
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }
}

/// Service for LLM-driven conversational context compaction.
///
/// When conversation history approaches a model's context window, this
/// service summarises older messages via a smaller/cheaper "compaction"
/// LLM call. The resulting anchored summary preserves semantic continuity
/// while freeing tokens for the main provider.
///
/// Supports both proactive compaction (before each turn) and reactive
/// compaction (after a provider context-overflow error). A circuit breaker
/// gates repeated failures to avoid cascading costs.
pub struct CompactionService {
    /// Provider registry for model access.
    providers: crate::provider::ProviderRegistry,
    /// Compaction configuration.
    config: crate::runtime::policy::CompactionConfig,
}

/// Result of a successful compaction.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// The LLM-generated summary text.
    pub summary_text: String,
    /// Message ID of the compaction user message (stored in session).
    pub compaction_user_id: Uuid,
    /// Message ID of the compaction assistant message (contains the summary).
    pub summary_message_id: Uuid,
    /// First retained message ID after the compaction boundary.
    pub tail_start_id: Uuid,
    /// Estimated tokens freed by this compaction.
    pub tokens_saved: usize,
}

impl CompactionService {
    /// Creates a new compaction service.
    #[must_use]
    pub fn new(
        providers: crate::provider::ProviderRegistry,
        config: crate::runtime::policy::CompactionConfig,
    ) -> Self {
        Self { providers, config }
    }

    /// Returns the compaction configuration.
    #[must_use]
    pub fn config(&self) -> &crate::runtime::policy::CompactionConfig {
        &self.config
    }

    /// Proactive compaction: called before each provider turn to check
    /// whether the conversation will overflow the context window.
    ///
    /// Returns `Ok(None)` when compaction is not needed. Returns
    /// `Ok(Some(result))` when compaction was performed successfully.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] when the compaction LLM call or storage fails.
    pub async fn compact_if_needed(
        &self,
        messages: &[MessageRecord],
        model_context: u32,
        max_output: u32,
        store: &dyn SessionStore,
        session_id: Uuid,
    ) -> RuntimeResult<Option<CompactionResult>> {
        let total_tokens = estimate_records_tokens(messages);

        if !overflow::is_overflow(total_tokens, model_context, max_output, self.config.auto) {
            return Ok(None);
        }

        let (provider, model) = self.resolve_compaction_provider()?;
        self.compact_impl(messages, provider, model, store, session_id, None)
            .await
            .map(Some)
    }

    /// Reactive compaction: called after the provider returns a context
    /// overflow error. This always performs compaction (does not check
    /// `is_overflow` first).
    ///
    /// Uses `model_context` to compute a safe [`crate::runtime::policy::CompactionConfig::keep_tokens`]
    /// override so the compaction prompt itself does not exceed the compaction
    /// model's context window.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] when the compaction LLM call or storage fails.
    pub async fn compact_after_overflow(
        &self,
        messages: &[MessageRecord],
        model_context: u32,
        max_output: u32,
        store: &dyn SessionStore,
        session_id: Uuid,
    ) -> RuntimeResult<CompactionResult> {
        let _ = max_output;

        let (provider, model) = self.resolve_compaction_provider()?;

        // The compaction model may have a smaller context window than the run
        // model. Reserve half the context for the prompt and head material,
        // the other half for the tail we want to keep.
        let safe_keep = (model_context as usize)
            .saturating_div(2)
            .max(1_024)
            .min(self.config.keep_tokens);

        self.compact_impl(
            messages,
            provider,
            model,
            store,
            session_id,
            Some(safe_keep),
        )
        .await
    }

    /// Internal compaction implementation.
    ///
    /// `keep_tokens_override` allows callers to override the configured
    /// [`CompactionConfig::keep_tokens`] — used by [`compact_after_overflow`]
    /// to ensure the compaction prompt fits within the overflow model's context.
    async fn compact_impl(
        &self,
        messages: &[MessageRecord],
        provider: Arc<dyn ChatProvider>,
        model: ModelName,
        store: &dyn SessionStore,
        session_id: Uuid,
        keep_tokens_override: Option<usize>,
    ) -> RuntimeResult<CompactionResult> {
        let effective_keep = keep_tokens_override.unwrap_or(self.config.keep_tokens);

        let selection = select::select(messages, self.config.tail_turns, effective_keep);

        if selection.head.is_empty() {
            let dummy_id = Uuid::nil();
            return Ok(CompactionResult {
                summary_text: String::new(),
                compaction_user_id: dummy_id,
                summary_message_id: dummy_id,
                tail_start_id: dummy_id,
                tokens_saved: 0,
            });
        }

        let tail_start_id = selection.tail_start_id.unwrap_or_else(Uuid::nil);

        let previous_summary = store
            .get_latest_compaction(&session_id)
            .await
            .map_err(RuntimeError::Storage)?
            .and_then(|m| m.compaction_meta)
            .and_then(|meta| meta.summary_text);

        let prompt_text = prompt::build_prompt(&selection.head, previous_summary.as_deref());

        let summary_text = self
            .run_compaction_llm(&*provider, &model, &prompt_text)
            .await?;

        let compaction_user_id = Uuid::now_v7();
        let compaction_user = MessageRecord {
            id: compaction_user_id,
            session_id,
            role: MessageRole::User,
            content: vec![crate::provider::ContentPart::text("[compaction]")],
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            usage: None,
            created_at: chrono::Utc::now(),
            is_compaction: true,
            is_summary: false,
            compaction_meta: Some(CompactionMeta::new(tail_start_id)),
        };
        store
            .append_message(compaction_user)
            .await
            .map_err(RuntimeError::Storage)?;

        let summary_message_id = Uuid::now_v7();
        let summary_message = MessageRecord {
            id: summary_message_id,
            session_id,
            role: MessageRole::Assistant,
            content: vec![crate::provider::ContentPart::text(&summary_text)],
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            usage: None,
            created_at: chrono::Utc::now(),
            is_compaction: false,
            is_summary: true,
            compaction_meta: Some(
                CompactionMeta::new(tail_start_id).with_summary(summary_text.clone()),
            ),
        };
        store
            .append_message(summary_message)
            .await
            .map_err(RuntimeError::Storage)?;

        let head_tokens = estimate_records_tokens(&selection.head);
        let summary_tokens = crate::token::estimate_tokens(&summary_text);

        Ok(CompactionResult {
            summary_text,
            compaction_user_id,
            summary_message_id,
            tail_start_id,
            tokens_saved: head_tokens.saturating_sub(summary_tokens),
        })
    }

    /// Calls the compaction LLM with the summarisation prompt.
    async fn run_compaction_llm(
        &self,
        provider: &dyn ChatProvider,
        model: &ModelName,
        prompt: &str,
    ) -> RuntimeResult<String> {
        let request = ChatRequest::new(model.clone()).with_message(Message::user_text(prompt));

        let response = provider
            .complete(request)
            .await
            .map_err(RuntimeError::Provider)?;

        let summary = extract_text_content(&response.message);

        if summary.is_empty() {
            return Err(RuntimeError::Provider(
                crate::error::ProviderError::Decode {
                    provider: provider.id(),
                    message: "compaction LLM returned empty response".to_owned(),
                },
            ));
        }

        Ok(summary)
    }

    /// Resolves the provider and model to use for compaction.
    ///
    /// Order of precedence:
    /// 1. `config.provider` / `config.model` (explicit compaction provider/model)
    /// 2. The default provider/model available in the registry
    fn resolve_compaction_provider(&self) -> RuntimeResult<(Arc<dyn ChatProvider>, ModelName)> {
        let provider_id = self
            .config
            .provider
            .clone()
            .or_else(|| self.providers.chat_ids().first().cloned())
            .ok_or_else(|| {
                RuntimeError::ProviderNotFound("no provider configured for compaction".to_owned())
            })?;

        let provider = self
            .providers
            .chat(&provider_id)
            .ok_or_else(|| RuntimeError::ProviderNotFound(provider_id.to_string()))?;

        let model = self
            .config
            .model
            .clone()
            .unwrap_or_else(|| ModelName::new("gpt-4o-mini"));

        Ok((provider, model))
    }
}

/// Extracts plain text content from an assistant message.
fn extract_text_content(message: &Message) -> String {
    match message {
        Message::Assistant { content, .. }
        | Message::System { content }
        | Message::User { content }
        | Message::Tool { content, .. } => {
            let mut text = String::new();
            for part in content {
                if let crate::provider::ContentPart::Text { text: t, .. } = part {
                    text.push_str(t);
                }
            }
            text
        }
        _ => String::new(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::store::{MessageRecord, MessageRole};
    use crate::token::estimate_records_tokens;
    use uuid::Uuid;

    fn make_user(text: &str) -> MessageRecord {
        MessageRecord::new(
            Uuid::now_v7(),
            MessageRole::User,
            vec![crate::provider::ContentPart::text(text)],
        )
    }

    fn make_assistant(text: &str) -> MessageRecord {
        MessageRecord::new(
            Uuid::now_v7(),
            MessageRole::Assistant,
            vec![crate::provider::ContentPart::text(text)],
        )
    }

    #[test]
    fn compact_if_needed_skips_when_no_overflow() {
        let messages = vec![make_user("hi"), make_assistant("hello")];
        let total = estimate_records_tokens(&messages);
        // With gpt-4o's 128K context, these few messages won't overflow
        assert!(!overflow::is_overflow(total, 128_000, 16_384, true,));
    }

    #[test]
    fn overflow_detected_on_large_context() {
        // Simulate a large conversation
        let large_text = "x".repeat(100_000); // ~25K tokens
        let messages = vec![
            make_user(&large_text),
            make_assistant(&large_text),
            make_user(&large_text),
            make_assistant(&large_text),
            make_user(&large_text),
            make_assistant(&large_text),
        ];
        let total = estimate_records_tokens(&messages);
        // With a 32K context model, this should overflow
        assert!(overflow::is_overflow(total, 32_000, 4_096, true,));
    }

    #[test]
    fn extract_text_from_assistant() {
        let msg = Message::assistant_text("summary text");
        assert_eq!(extract_text_content(&msg), "summary text");
    }

    #[test]
    fn breaker_starts_closed() {
        let breaker = CompactionCircuitBreaker::new(3);
        assert!(!breaker.is_open());
        assert_eq!(breaker.consecutive_failures(), 0);
    }

    #[test]
    fn breaker_opens_after_threshold_failures() {
        let mut breaker = CompactionCircuitBreaker::new(3);
        assert!(!breaker.record_failure());
        assert!(!breaker.is_open());
        assert!(!breaker.record_failure());
        assert!(!breaker.is_open());
        assert!(breaker.record_failure());
        assert!(breaker.is_open());
        assert_eq!(breaker.consecutive_failures(), 3);
    }

    #[test]
    fn breaker_resets_on_success() {
        let mut breaker = CompactionCircuitBreaker::new(3);
        breaker.record_failure();
        breaker.record_failure();
        assert!(!breaker.is_open());

        breaker.record_success();
        assert!(!breaker.is_open());
        assert_eq!(breaker.consecutive_failures(), 0);

        // Should need 3 fresh failures to open again
        breaker.record_failure();
        breaker.record_failure();
        assert!(!breaker.is_open());
        breaker.record_failure();
        assert!(breaker.is_open());
    }

    #[test]
    fn breaker_after_open_does_not_count_further() {
        let mut breaker = CompactionCircuitBreaker::new(2);
        breaker.record_failure();
        assert!(breaker.record_failure());
        assert!(breaker.is_open());

        // Further failures don't re-trigger
        assert!(!breaker.record_failure());
        assert!(breaker.is_open());
        assert_eq!(breaker.consecutive_failures(), 2);
    }

    #[test]
    fn breaker_threshold_zero_opens_immediately() {
        let mut breaker = CompactionCircuitBreaker::new(0);
        assert!(breaker.record_failure());
        assert!(breaker.is_open());
    }
}
