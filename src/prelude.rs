//! Convenient imports for application and integration code.

pub use crate::agent::{
    AgentDefinition, AgentMode, AgentRegistry, PermissionEffect, PermissionRule,
};
pub use crate::config::{
    AgentConfig, AgentConfigBuilder, ConfigLoader, ProviderConfig, ProviderType, RuntimeConfig,
    RuntimePolicyConfig, StoreBackend, StoreConfig,
};
pub use crate::context::{
    ContextAdapter, ContextFactory, ContextInput, ContextOutput, ContextResult, FunctionAdapter,
    StaticAdapter,
};
pub use crate::error::{ContextError, ProviderError, Result, StorageError, ToolError};
pub use crate::provider::{
    ChatProvider, ChatRequest, ChatResponse, ChatStream, ChatStreamEvent, ContentPart, Embedding,
    EmbeddingInput, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, FinishReason, Message,
    ModelName, ProviderCapabilities, ProviderHttpConfig, ProviderId, ProviderRegistry,
    ProviderResult, ResponseFormat, TokenUsage, ToolCall, ToolChoice, ToolSpec,
};
pub use crate::runtime::{
    AgentEvent, AgentRuntime, CompactionConfig, CompactionResult, CompactionService,
    ContextPipeline, Control, EmitRequest, EventKind, FileSnapshotStore, InvocationError,
    InvocationEvent, InvocationHandle, ModelRouter, RunId, RunOutput, RunRequest, RunStatus,
    RuntimeError, RuntimeInvocation, RuntimePolicy, RuntimeStore, SessionContext, SessionGate,
    SessionGuard, Snapshot, SnapshotStore, ToolRuntime,
};
pub use crate::store::memory::{
    MemoryArtifactStore, MemoryEmbeddingStore, MemoryExecutionStore, MemorySessionStore,
};
pub use crate::store::{
    Artifact, ArtifactStore, CompactionMeta, EmbeddingRecord, EmbeddingStore, ExecutionStore,
    MessageRecord, MessageRole, ScoredEmbedding, Session, SessionStats, SessionStore, StoreResult,
    ToolExecution, ToolExecutionStatus, UsageRecord,
};
pub use crate::token::{
    estimate_content_part_tokens, estimate_message_tokens, estimate_messages_tokens,
    estimate_record_tokens, estimate_records_tokens, estimate_tokens,
};
pub use crate::tool::{ExternalTool, FunctionTool, Tool, ToolOutput, ToolRegistry, ToolResult};
pub use crate::tool_output::{ToolOutputConfig, TruncationResult, truncate_output};

#[cfg(feature = "openai")]
pub use crate::adapt::openai::{OpenAiChatAdapter, OpenAiEmbeddingAdapter};

#[cfg(feature = "anthropic")]
pub use crate::adapt::anthropic::AnthropicChatAdapter;

#[cfg(any(
    feature = "sqlx-postgres",
    feature = "sqlx-mysql",
    feature = "sqlx-sqlite"
))]
pub use crate::store::sql::SqlSessionStore;

#[cfg(feature = "sqlx-postgres")]
pub use crate::store::sql::SqlEmbeddingStore;

#[cfg(feature = "mongodb")]
pub use crate::store::mongodb::MongodbSessionStore;

#[cfg(feature = "surrealdb")]
pub use crate::store::surrealdb::SurrealdbSessionStore;

#[cfg(feature = "redis")]
pub use crate::store::redis::RedisSessionStore;

#[cfg(feature = "qdrant")]
pub use crate::store::qdrant::QdrantEmbeddingStore;

#[cfg(feature = "object_store")]
pub use crate::store::object::{DiskArtifactStore, S3ArtifactStore};

#[cfg(feature = "rag")]
pub use crate::rag::RagContextAdapter;

#[cfg(feature = "queue")]
pub use crate::queue::{EventPublisher, NoOpPublisher, QueueError, QueueResult};

#[cfg(all(feature = "queue", feature = "nats"))]
pub use crate::queue::NatsEventPublisher;

#[cfg(all(feature = "queue", feature = "redis"))]
pub use crate::queue::RedisStreamsPublisher;

#[cfg(feature = "rag")]
pub use crate::config::RagConfig;

#[cfg(feature = "queue")]
pub use crate::config::{QueueBackend, QueueConfig};
