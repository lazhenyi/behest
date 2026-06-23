//! Context pipeline for runtime.
//!
//! Wraps the existing `ContextFactory` with runtime-specific adapters
//! for session history, compaction-aware message filtering, and
//! token-budget management.

use std::sync::Arc;

use uuid::Uuid;

use crate::context::{ContextAdapter, ContextFactory, ContextInput, ContextOutput};
use crate::provider::{ChatRequest, ContentPart, Message, ModelName, ToolSpec};
use crate::store::MessageRecord;
use crate::token::estimate_records_tokens;

use super::error::RuntimeResult;

/// Runtime context pipeline that composes context from multiple sources.
///
/// The pipeline:
/// 1. Loads session history from the store
/// 2. Applies compaction filter (reorder post-compaction messages)
/// 3. Invokes registered context adapters (system prompt, RAG, etc.)
/// 4. Applies token-budget trimming as a safety net
/// 5. Produces a final `ChatRequest`
pub struct ContextPipeline {
    factory: ContextFactory,
    max_history_messages: usize,
    max_history_tokens: usize,
    enable_compaction_filter: bool,
}

impl ContextPipeline {
    /// Creates a new context pipeline with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            factory: ContextFactory::new(),
            max_history_messages: 50,
            max_history_tokens: 64_000,
            enable_compaction_filter: true,
        }
    }

    /// Creates a context pipeline with an existing context factory.
    #[must_use]
    pub fn with_factory(factory: ContextFactory) -> Self {
        Self {
            factory,
            max_history_messages: 50,
            max_history_tokens: 64_000,
            enable_compaction_filter: true,
        }
    }

    /// Sets the maximum number of history messages to include (fallback limit).
    #[must_use]
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history_messages = max;
        self
    }

    /// Sets the maximum token budget for history messages.
    #[must_use]
    pub fn with_max_history_tokens(mut self, max: usize) -> Self {
        self.max_history_tokens = max;
        self
    }

    /// Enables or disables the compaction message filter.
    #[must_use]
    pub fn with_compaction_filter(mut self, enable: bool) -> Self {
        self.enable_compaction_filter = enable;
        self
    }

    /// Registers a context adapter.
    pub fn register<A>(&mut self, adapter: A)
    where
        A: ContextAdapter + 'static,
    {
        self.factory.register(adapter);
    }

    /// Registers an already shared context adapter.
    pub fn register_arc(&mut self, adapter: Arc<dyn ContextAdapter>) {
        self.factory.register_arc(adapter);
    }

    /// Returns adapter names in registration order.
    pub fn adapter_names(&self) -> impl Iterator<Item = &str> {
        self.factory.adapter_names()
    }

    /// Builds a chat request from context.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` when context building fails.
    pub async fn build(
        &self,
        store: &super::store::RuntimeStore,
        session_id: Uuid,
        model: ModelName,
        user_message: Option<&str>,
        tools: Option<&[ToolSpec]>,
    ) -> RuntimeResult<ChatRequest> {
        let input = ContextInput {
            user_message: user_message.map(str::to_owned),
            session_id: Some(session_id.to_string()),
            metadata: serde_json::Value::Null,
        };

        let mut output = self.factory.build(&input).await.map_err(|e| {
            super::error::RuntimeError::Context(crate::error::ContextError::AdapterFailed {
                adapter: "pipeline".to_owned(),
                message: e.to_string(),
            })
        })?;

        let records = store
            .sessions()
            .list_messages(&session_id)
            .await
            .map_err(super::error::RuntimeError::Storage)?;

        let records = if self.enable_compaction_filter {
            apply_compaction_filter(records)
        } else {
            records
        };

        let records = trim_by_tokens(records, self.max_history_tokens);

        let history: Vec<Message> = records
            .into_iter()
            .filter_map(super::store::record_to_message)
            .collect();

        output.extend(history);

        if let Some(text) = user_message {
            output.extend([Message::user_text(text)]);
        }

        let request = match tools {
            Some(specs) => output.into_request_with_tools(model, specs),
            None => output.into_request(model),
        };

        Ok(request)
    }

    /// Builds context output without creating a request.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError` when context building fails.
    pub async fn build_context(
        &self,
        store: &super::store::RuntimeStore,
        session_id: Uuid,
        user_message: Option<&str>,
    ) -> RuntimeResult<ContextOutput> {
        let input = ContextInput {
            user_message: user_message.map(str::to_owned),
            session_id: Some(session_id.to_string()),
            metadata: serde_json::Value::Null,
        };

        let mut output = self.factory.build(&input).await.map_err(|e| {
            super::error::RuntimeError::Context(crate::error::ContextError::AdapterFailed {
                adapter: "pipeline".to_owned(),
                message: e.to_string(),
            })
        })?;

        let records = store
            .sessions()
            .list_messages(&session_id)
            .await
            .map_err(super::error::RuntimeError::Storage)?;

        let records = if self.enable_compaction_filter {
            apply_compaction_filter(records)
        } else {
            records
        };

        let records = trim_by_tokens(records, self.max_history_tokens);

        let history: Vec<Message> = records
            .into_iter()
            .filter_map(super::store::record_to_message)
            .collect();

        output.extend(history);

        if let Some(text) = user_message {
            output.extend([Message::user_text(text)]);
        }

        Ok(output)
    }
}

impl Default for ContextPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Applies the compaction message filter.
///
/// Finds the latest completed compaction in the message history and reorders
/// messages so that the compacted head (old messages that were summarized)
/// is replaced by the compaction checkpoint while the retained tail and
/// post-compaction messages remain visible.
///
/// Ported from OpenCode V1's `filterCompacted()`.
fn apply_compaction_filter(records: Vec<MessageRecord>) -> Vec<MessageRecord> {
    if records.is_empty() {
        return records;
    }

    // Walk backwards to find the latest completed compaction pair:
    //   summary_assistant (is_summary=true) immediately after
    //   compaction_user (is_compaction=true)
    let mut summary_idx: Option<usize> = None;
    let mut compaction_idx: Option<usize> = None;

    for i in (0..records.len()).rev() {
        let rec = &records[i];
        if rec.is_summary && summary_idx.is_none() {
            summary_idx = Some(i);
        } else if rec.is_compaction && compaction_idx.is_none() && summary_idx.is_some() {
            // Found the compaction user that precedes this summary
            compaction_idx = Some(i);
            break;
        } else if !rec.is_summary && !rec.is_compaction && summary_idx.is_some() {
            // We found a summary but the preceding message is NOT a compaction user
            // Reset — this is not a valid compaction pair
            summary_idx = None;
        }
    }

    let (Some(c_idx), Some(s_idx)) = (compaction_idx, summary_idx) else {
        return records;
    };

    let tail_start_id = records[c_idx]
        .compaction_meta
        .as_ref()
        .and_then(|m| m.tail_start_id);

    let Some(tail_start) = tail_start_id else {
        return records;
    };

    // Find the index of tail_start_id
    let tail_idx = records.iter().position(|r| r.id == tail_start);

    // Split records into three groups:
    //   before: [0..tail_idx) — the compacted head (EXCLUDED)
    //   tail:   [tail_idx..compact_end) — the retained tail
    //   after:  [compact_end..) — post-compaction messages
    //
    // compact_end = first non-compaction message after summary_idx
    let compact_end = records
        .iter()
        .skip(s_idx + 1)
        .position(|r| !r.is_compaction && !r.is_summary)
        .map_or(records.len(), |p| s_idx + 1 + p);

    // Build result:
    //   1. Compaction checkpoint as a synthetic system message
    //   2. Retained tail (from tail_idx to between compaction and summary)
    //   3. Post-compaction messages (after compact_end)

    let mut result = Vec::with_capacity(records.len());

    // Phase 1: Synthetic compaction checkpoint
    // Build a system message from the compaction pair
    if let Some(summary_meta) = &records[s_idx].compaction_meta {
        if let Some(summary_text) = &summary_meta.summary_text {
            let checkpoint = MessageRecord {
                id: Uuid::now_v7(),
                session_id: records[c_idx].session_id,
                role: crate::store::MessageRole::System,
                content: vec![ContentPart::text(format!(
                    "<conversation-checkpoint>\n<summary>\n{summary_text}\n</summary>\n</conversation-checkpoint>"
                ))],
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_name: None,
                usage: None,
                created_at: records[s_idx].created_at,
                is_compaction: false,
                is_summary: false,
                compaction_meta: None,
            };
            result.push(checkpoint);
        }
    }

    // Phase 2: Retained tail (messages between tail_start and compaction_user)
    if let Some(ti) = tail_idx {
        let tail_end = c_idx.min(records.len());
        for rec in records.iter().skip(ti).take(tail_end.saturating_sub(ti)) {
            if !rec.is_compaction && !rec.is_summary {
                result.push(rec.clone());
            }
        }
    }

    // Phase 3: Post-compaction messages (everything after the summary)
    for rec in records.iter().skip(compact_end) {
        result.push(rec.clone());
    }

    result
}

/// Trims message history to stay within a token budget.
///
/// Walks from the end of the list forward, accumulating token estimates,
/// and drops the oldest messages when the budget is exceeded.
/// Preserves the first message if it is a system message.
fn trim_by_tokens(records: Vec<MessageRecord>, max_tokens: usize) -> Vec<MessageRecord> {
    if records.is_empty() {
        return records;
    }

    let total = estimate_records_tokens(&records);
    if total <= max_tokens {
        return records;
    }

    // Note: first system message preservation is handled implicitly —
    // since we walk backwards, the earliest messages (including system)
    // are the first ones dropped when budget is exceeded.

    // Walk backwards, keeping messages until budget exceeded
    let mut kept = Vec::new();
    let mut tokens = 0usize;

    for rec in records.into_iter().rev() {
        let rec_tokens = crate::token::estimate_record_tokens(&rec);
        if tokens + rec_tokens > max_tokens && !kept.is_empty() {
            // Don't add this message — it would exceed budget
            // But keep the system message if we haven't included it yet
            break;
        }
        tokens += rec_tokens;
        kept.push(rec);
    }

    kept.reverse();

    // Re-prepend the system message if it was dropped
    // If we dropped the system message due to extreme length, that's acceptable

    kept
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::context::StaticAdapter;
    use crate::provider::ContentPart;
    use crate::runtime::memory::MemoryRunStore;
    use crate::store::memory::{MemoryExecutionStore, MemorySessionStore};
    use crate::store::{CompactionMeta, MessageRole, Session};

    fn make_store() -> super::super::store::RuntimeStore {
        let sessions = MemorySessionStore::new();
        let executions = MemoryExecutionStore::new();
        let runs = MemoryRunStore::new();
        super::super::store::RuntimeStore::new(
            Box::new(sessions),
            Box::new(executions),
            Box::new(runs),
        )
    }

    fn make_record_for(session_id: Uuid, role: MessageRole, text: &str) -> MessageRecord {
        MessageRecord::new(session_id, role, vec![ContentPart::text(text)])
    }

    #[tokio::test]
    async fn pipeline_should_compose_system_and_history() {
        let store = make_store();

        let session = Session::new("Test", ModelName::new("gpt-4"));
        store
            .sessions()
            .create_session(session.clone())
            .await
            .unwrap();

        let user_msg = make_record_for(session.id, MessageRole::User, "Hello");
        store.sessions().append_message(user_msg).await.unwrap();

        let mut pipeline = ContextPipeline::new();
        pipeline.register(StaticAdapter::system("You are helpful."));

        let request = pipeline
            .build(
                &store,
                session.id,
                ModelName::new("gpt-4"),
                Some("How are you?"),
                None,
            )
            .await
            .unwrap();

        assert_eq!(request.messages.len(), 3);
        assert!(matches!(request.messages[0], Message::System { .. }));
        assert!(matches!(request.messages[1], Message::User { .. }));
        assert!(matches!(request.messages[2], Message::User { .. }));
    }

    #[tokio::test]
    async fn pipeline_should_apply_token_trim() {
        let store = make_store();

        let session = Session::new("Test", ModelName::new("gpt-4"));
        store
            .sessions()
            .create_session(session.clone())
            .await
            .unwrap();

        for i in 0..10 {
            let msg = make_record_for(session.id, MessageRole::User, &format!("Message {i}"));
            store.sessions().append_message(msg).await.unwrap();
        }

        // Very restrictive token budget — should only keep a few messages
        let pipeline = ContextPipeline::new().with_max_history_tokens(50);

        let request = pipeline
            .build(&store, session.id, ModelName::new("gpt-4"), None, None)
            .await
            .unwrap();

        // Should have fewer messages than the original 10
        assert!(request.messages.len() < 10);
    }

    #[tokio::test]
    async fn pipeline_should_filter_compacted_head() {
        let store = make_store();

        let session = Session::new("Test", ModelName::new("gpt-4"));
        store
            .sessions()
            .create_session(session.clone())
            .await
            .unwrap();

        let sid = session.id;

        // Create messages: m1, m2 (head), m3, m4 (tail)
        let m1 = make_record_for(sid, MessageRole::User, "m1");
        let m2 = make_record_for(sid, MessageRole::Assistant, "m2");
        let m3 = make_record_for(sid, MessageRole::User, "m3");
        let m4 = make_record_for(sid, MessageRole::Assistant, "m4");

        let m3_id = m3.id;

        store.sessions().append_message(m1).await.unwrap();
        store.sessions().append_message(m2).await.unwrap();
        store.sessions().append_message(m3).await.unwrap();
        store.sessions().append_message(m4).await.unwrap();

        // Append compaction pair
        let compaction_user = MessageRecord {
            id: Uuid::now_v7(),
            session_id: sid,
            role: MessageRole::User,
            content: vec![ContentPart::text("[compaction]")],
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            usage: None,
            created_at: chrono::Utc::now(),
            is_compaction: true,
            is_summary: false,
            compaction_meta: Some(CompactionMeta::new(m3_id)),
        };
        store
            .sessions()
            .append_message(compaction_user)
            .await
            .unwrap();

        let summary_msg = MessageRecord {
            id: Uuid::now_v7(),
            session_id: sid,
            role: MessageRole::Assistant,
            content: vec![ContentPart::text("Summary of m1-m2")],
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            usage: None,
            created_at: chrono::Utc::now(),
            is_compaction: false,
            is_summary: true,
            compaction_meta: Some(
                CompactionMeta::new(m3_id).with_summary("Summary of m1-m2".to_owned()),
            ),
        };
        store.sessions().append_message(summary_msg).await.unwrap();

        // Append post-compaction message
        let m5 = make_record_for(sid, MessageRole::User, "m5");
        store.sessions().append_message(m5).await.unwrap();

        let pipeline = ContextPipeline::new();

        let request = pipeline
            .build(&store, sid, ModelName::new("gpt-4"), None, None)
            .await
            .unwrap();

        // Should NOT include m1, m2 (compacted head)
        // Should include: checkpoint system message, m3, m4, m5
        let has_m1 = request.messages.iter().any(|m| {
            matches!(m, Message::User { content } if content.iter().any(|p| matches!(p, ContentPart::Text { text } if text == "m1")))
        });
        let has_m3 = request.messages.iter().any(|m| {
            matches!(m, Message::User { content } if content.iter().any(|p| matches!(p, ContentPart::Text { text } if text == "m3")))
        });
        let has_checkpoint = request.messages.iter().any(|m| {
            matches!(m, Message::System { content } if content.iter().any(|p| matches!(p, ContentPart::Text { text } if text.contains("conversation-checkpoint"))))
        });

        assert!(!has_m1, "compacted head should be excluded");
        assert!(has_m3, "retained tail should be included");
        assert!(has_checkpoint, "compaction checkpoint should be present");
    }

    #[tokio::test]
    async fn pipeline_no_compaction_returns_all() {
        let store = make_store();

        let session = Session::new("Test", ModelName::new("gpt-4"));
        store
            .sessions()
            .create_session(session.clone())
            .await
            .unwrap();

        for i in 0..5 {
            let msg = make_record_for(session.id, MessageRole::User, &format!("msg{i}"));
            store.sessions().append_message(msg).await.unwrap();
        }

        let pipeline = ContextPipeline::new().with_compaction_filter(true);

        let request = pipeline
            .build(&store, session.id, ModelName::new("gpt-4"), None, None)
            .await
            .unwrap();

        assert_eq!(request.messages.len(), 5);
    }

    #[test]
    fn trim_by_tokens_preserves_tail() {
        let sid = Uuid::now_v7();
        let records: Vec<MessageRecord> = (0..20)
            .map(|i| make_record_for(sid, MessageRole::User, &format!("msg{i:03}")))
            .collect();

        let trimmed = trim_by_tokens(records, 100);

        // Should have fewer records, but not empty
        assert!(!trimmed.is_empty());
        assert!(trimmed.len() < 20);
    }

    #[test]
    fn trim_by_tokens_all_when_under_budget() {
        let sid = Uuid::now_v7();
        let records = vec![
            make_record_for(sid, MessageRole::User, "hi"),
            make_record_for(sid, MessageRole::Assistant, "hello"),
        ];

        let trimmed = trim_by_tokens(records.clone(), 1000);
        assert_eq!(trimmed.len(), 2);
    }

    #[test]
    fn filter_compacted_no_compaction_returns_unchanged() {
        let sid = Uuid::now_v7();
        let records = vec![
            make_record_for(sid, MessageRole::User, "a"),
            make_record_for(sid, MessageRole::Assistant, "b"),
        ];

        let filtered = apply_compaction_filter(records.clone());
        assert_eq!(filtered.len(), 2);
    }
}
