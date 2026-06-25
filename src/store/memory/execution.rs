//! In-memory execution store for tool calls and token usage tracking.

use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::store::{
    ExecutionStore, SessionStats, StoreResult, ToolExecution, ToolExecutionStatus, UsageRecord,
};

/// In-memory execution store for testing, development, and prototyping.
///
/// Tracks tool executions and token usage records in memory-backed
/// `HashMap`s protected by `RwLock`. Data is lost when the process exits.
/// Implements [`ExecutionStore`].
#[derive(Default)]
pub struct MemoryExecutionStore {
    executions: RwLock<HashMap<Uuid, ToolExecution>>,
    usage_records: RwLock<HashMap<Uuid, UsageRecord>>,
    message_counts: RwLock<HashMap<Uuid, u64>>,
}

impl MemoryExecutionStore {
    /// Creates an empty in-memory execution store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the message count for a session, used by [`session_stats`](ExecutionStore::session_stats).
    ///
    /// In a real backend this would be computed from a messages table;
    /// the in-memory store requires the caller to explicitly provide it
    /// since messages are maintained in the session store, not here.
    pub async fn set_message_count(&self, session_id: Uuid, count: u64) {
        self.message_counts.write().await.insert(session_id, count);
    }
}

#[async_trait]
impl ExecutionStore for MemoryExecutionStore {
    async fn record_execution(&self, execution: ToolExecution) -> StoreResult<ToolExecution> {
        let id = execution.id;
        self.executions.write().await.insert(id, execution.clone());
        Ok(execution)
    }

    async fn list_executions(&self, session_id: &Uuid) -> StoreResult<Vec<ToolExecution>> {
        let executions = self.executions.read().await;
        let mut result: Vec<ToolExecution> = executions
            .values()
            .filter(|e| e.session_id == *session_id)
            .cloned()
            .collect();
        result.sort_by_key(|e| e.created_at);
        Ok(result)
    }

    async fn list_executions_by_message(
        &self,
        message_id: &Uuid,
    ) -> StoreResult<Vec<ToolExecution>> {
        let executions = self.executions.read().await;
        let mut result: Vec<ToolExecution> = executions
            .values()
            .filter(|e| e.message_id == *message_id)
            .cloned()
            .collect();
        result.sort_by_key(|e| e.created_at);
        Ok(result)
    }

    async fn record_usage(&self, record: UsageRecord) -> StoreResult<UsageRecord> {
        let id = record.id;
        self.usage_records.write().await.insert(id, record.clone());
        Ok(record)
    }

    async fn list_usage(&self, session_id: &Uuid) -> StoreResult<Vec<UsageRecord>> {
        let records = self.usage_records.read().await;
        let mut result: Vec<UsageRecord> = records
            .values()
            .filter(|r| r.session_id == *session_id)
            .cloned()
            .collect();
        result.sort_by_key(|r| r.created_at);
        Ok(result)
    }

    async fn session_stats(&self, session_id: &Uuid) -> StoreResult<SessionStats> {
        let executions = self.executions.read().await;
        let usage_records = self.usage_records.read().await;
        let message_counts = self.message_counts.read().await;

        let session_execs: Vec<&ToolExecution> = executions
            .values()
            .filter(|e| e.session_id == *session_id)
            .collect();

        let session_usage: Vec<&UsageRecord> = usage_records
            .values()
            .filter(|r| r.session_id == *session_id)
            .collect();

        let tool_call_count = session_execs.len() as u64;
        let tool_success_count = session_execs
            .iter()
            .filter(|e| e.status == ToolExecutionStatus::Success)
            .count() as u64;
        let tool_failure_count = session_execs
            .iter()
            .filter(|e| e.status == ToolExecutionStatus::Failed)
            .count() as u64;

        let total_duration_ms: u64 = session_execs
            .iter()
            .map(|e| u64::try_from(e.duration.as_millis()).unwrap_or(u64::MAX))
            .sum();

        let avg_tool_duration_ms = total_duration_ms.checked_div(tool_call_count).unwrap_or(0);

        let total_input_tokens: u64 = session_usage.iter().map(|r| r.input_tokens).sum();
        let total_output_tokens: u64 = session_usage.iter().map(|r| r.output_tokens).sum();
        let total_tokens: u64 = session_usage.iter().map(|r| r.total_tokens).sum();

        let message_count = message_counts.get(session_id).copied().unwrap_or(0);

        Ok(SessionStats {
            session_id: *session_id,
            message_count,
            tool_call_count,
            tool_success_count,
            tool_failure_count,
            total_input_tokens,
            total_output_tokens,
            total_tokens,
            avg_tool_duration_ms,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::provider::TokenUsage;
    use serde_json::json;
    use std::time::Duration;

    fn test_session_id() -> Uuid {
        Uuid::now_v7()
    }

    fn test_message_id() -> Uuid {
        Uuid::now_v7()
    }

    #[tokio::test]
    async fn memory_execution_store_should_record_and_list_executions() {
        let store = MemoryExecutionStore::new();
        let session_id = test_session_id();
        let message_id = test_message_id();

        let exec = ToolExecution::new(
            session_id,
            message_id,
            "call_1",
            "get_weather",
            json!({"city": "London"}),
        )
        .with_success(json!({"temp": 22}), Duration::from_millis(150));

        store.record_execution(exec).await.unwrap();

        let execs = store.list_executions(&session_id).await.unwrap();
        assert_eq!(execs.len(), 1);
        assert_eq!(execs[0].tool_name, "get_weather");
        assert_eq!(execs[0].status, ToolExecutionStatus::Success);
        assert_eq!(execs[0].duration, Duration::from_millis(150));
    }

    #[tokio::test]
    async fn memory_execution_store_should_list_by_message() {
        let store = MemoryExecutionStore::new();
        let session_id = test_session_id();
        let msg1 = test_message_id();
        let msg2 = test_message_id();

        store
            .record_execution(ToolExecution::new(
                session_id,
                msg1,
                "call_1",
                "tool_a",
                json!({}),
            ))
            .await
            .unwrap();
        store
            .record_execution(ToolExecution::new(
                session_id,
                msg2,
                "call_2",
                "tool_b",
                json!({}),
            ))
            .await
            .unwrap();

        let by_msg1 = store.list_executions_by_message(&msg1).await.unwrap();
        assert_eq!(by_msg1.len(), 1);
        assert_eq!(by_msg1[0].tool_name, "tool_a");
    }

    #[tokio::test]
    async fn memory_execution_store_should_record_failed_execution() {
        let store = MemoryExecutionStore::new();
        let session_id = test_session_id();
        let message_id = test_message_id();

        let exec = ToolExecution::new(session_id, message_id, "call_1", "broken_tool", json!({}))
            .with_failure("tool crashed", Duration::from_millis(50));

        store.record_execution(exec).await.unwrap();

        let execs = store.list_executions(&session_id).await.unwrap();
        assert_eq!(execs[0].status, ToolExecutionStatus::Failed);
        assert_eq!(execs[0].error.as_deref(), Some("tool crashed"));
    }

    #[tokio::test]
    async fn memory_execution_store_should_record_and_list_usage() {
        let store = MemoryExecutionStore::new();
        let session_id = test_session_id();
        let message_id = test_message_id();

        let record = UsageRecord::new(
            session_id,
            message_id,
            "openai",
            "gpt-4",
            TokenUsage::new(100, 50),
        );

        store.record_usage(record).await.unwrap();

        let usage = store.list_usage(&session_id).await.unwrap();
        assert_eq!(usage.len(), 1);
        assert_eq!(usage[0].provider, "openai");
        assert_eq!(usage[0].model, "gpt-4");
        assert_eq!(usage[0].input_tokens, 100);
        assert_eq!(usage[0].output_tokens, 50);
        assert_eq!(usage[0].total_tokens, 150);
    }

    #[tokio::test]
    async fn memory_execution_store_should_compute_session_stats() {
        let store = MemoryExecutionStore::new();
        let session_id = test_session_id();
        let msg = test_message_id();

        store.set_message_count(session_id, 10).await;

        // 3 tool executions: 2 success, 1 failure
        store
            .record_execution(
                ToolExecution::new(session_id, msg, "c1", "t1", json!({}))
                    .with_success(json!(null), Duration::from_millis(100)),
            )
            .await
            .unwrap();
        store
            .record_execution(
                ToolExecution::new(session_id, msg, "c2", "t2", json!({}))
                    .with_success(json!(null), Duration::from_millis(200)),
            )
            .await
            .unwrap();
        store
            .record_execution(
                ToolExecution::new(session_id, msg, "c3", "t3", json!({}))
                    .with_failure("err", Duration::from_millis(300)),
            )
            .await
            .unwrap();

        // 2 usage records
        store
            .record_usage(UsageRecord::new(
                session_id,
                msg,
                "openai",
                "gpt-4",
                TokenUsage::new(100, 50),
            ))
            .await
            .unwrap();
        store
            .record_usage(UsageRecord::new(
                session_id,
                msg,
                "openai",
                "gpt-4",
                TokenUsage::new(200, 80),
            ))
            .await
            .unwrap();

        let stats = store.session_stats(&session_id).await.unwrap();

        assert_eq!(stats.message_count, 10);
        assert_eq!(stats.tool_call_count, 3);
        assert_eq!(stats.tool_success_count, 2);
        assert_eq!(stats.tool_failure_count, 1);
        assert_eq!(stats.total_input_tokens, 300);
        assert_eq!(stats.total_output_tokens, 130);
        assert_eq!(stats.total_tokens, 430);
        assert_eq!(stats.avg_tool_duration_ms, 200); // (100+200+300)/3
    }

    #[tokio::test]
    async fn memory_execution_store_should_return_empty_stats() {
        let store = MemoryExecutionStore::new();
        let stats = store.session_stats(&test_session_id()).await.unwrap();

        assert_eq!(stats.message_count, 0);
        assert_eq!(stats.tool_call_count, 0);
        assert_eq!(stats.total_tokens, 0);
    }
}
