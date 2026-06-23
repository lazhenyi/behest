# Extending the Runtime

## Provider Adapter 开发

新增 model provider 时，按以下步骤操作：

### 1. 目录结构

```
src/adapt/<provider>/
├── mod.rs      # pub mod chat; pub mod convert; pub mod types; (pub mod embed;)
├── chat.rs     # ChatProvider 实现
├── convert.rs  # 请求构建 + 响应解析 + 错误映射
├── embed.rs    # EmbeddingProvider 实现（可选，仅支持 embedding 的 provider）
├── types.rs    # provider 原生 API 类型
```

### 2. 实现 ChatProvider

`chat.rs` 实现 `ChatProvider` trait：

```rust
#[async_trait]
impl ChatProvider for MyProvider {
    fn id(&self) -> ProviderId { ... }
    fn capabilities(&self) -> ProviderCapabilities { ... }
    async fn complete(&self, request: ChatRequest) -> ProviderResult<ChatResponse> { ... }
    async fn stream(&self, request: ChatRequest) -> ProviderResult<ChatStream> { ... }
}
```

### 3. 共享基础设施

- HTTP client 通过 `adapt/http.rs` 的 `HttpClientConfig` 构建
- SSE 解析使用 `adapt/sse.rs` 的 `SseParser`
- 异步 trait 使用 `#[async_trait]`

### 4. 错误转换

`convert.rs` 中实现 `From<reqwest::Error>` → `ProviderError::Transport`、HTTP status → `ProviderError::Authentication` / `RateLimited` 等映射。禁止在 chat.rs 中内联错误转换逻辑。

### 5. Feature gate

在 `Cargo.toml` 新增 feature（小写 provider 名），在 `adapt/mod.rs` 中 conditional compile：

```rust
#[cfg(feature = "myprovider")]
pub mod myprovider;
```

## Embedding Provider

与 ChatProvider 平行。`embed.rs` 实现 `EmbeddingProvider` trait，`convert.rs` 处理 embedding 请求/响应。

## Runtime 扩展

### 新增 AgentEvent variant

在 `src/runtime/event.rs` 中新增 variant 时，同步更新：

1. `AgentEvent::run_id()` — 新增 variant 的 match arm
2. `AgentEvent` 的 `Debug` / `Clone` / `Serialize` / `Deserialize` derive 确认覆盖

### 新增 TurnState

在 `src/runtime/turn.rs` 中新增状态时，同步更新：

1. `TurnTransition::resolve()` — 新增状态的 match arm（包含所有合法 `TurnOutcome` → `TurnAction` 映射）
2. `TurnTransition::status_for()` — 新增状态 → `RunStatus` 映射
3. `AgentRuntime::run_loop_inner()` — 新增执行分支 + snapshot save 调用
4. 更新 ASCII FSM 注释

### 新增 Store backend

在 `src/store/` 下新增目录。必须实现对应的 store trait（`SessionStore` / `EmbeddingStore` / `ExecutionStore`）。新增 Cargo feature 命名使用 backend 名（小写）。SQL backend 的 migration 放在 `src/store/sql/migrations/<backend>/`。

### 新增 Context adapter

实现 `ContextAdapter` trait，在 `ContextPipeline` 中通过 `register()` 注册。adapter 在 context build 时按注册顺序执行。

### 新增 RunStore backend

实现 `RunStore` trait（在 `src/runtime/store.rs` 中定义）：

```rust
#[async_trait]
impl RunStore for MyRunStore {
    async fn create_run(&self, record: RunRecord) -> RuntimeResult<()>;
    async fn get_run(&self, run_id: RunId) -> RuntimeResult<Option<RunRecord>>;
    async fn update_run_status(&self, run_id: RunId, status: RunStatus) -> RuntimeResult<()>;
    async fn append_event(&self, record: RunEventRecord) -> RuntimeResult<()>;
    async fn list_events(&self, run_id: RunId) -> RuntimeResult<Vec<RunEventRecord>>;
    async fn list_runs(&self, session_id: Uuid) -> RuntimeResult<Vec<RunRecord>>;
    async fn delete_run(&self, run_id: RunId) -> RuntimeResult<()>;
    async fn health_check(&self) -> RuntimeResult<()>;
}
```

关键设计：
- `get_run_state()` 有默认实现（折叠 `get_run()` + `list_events()` 为 `RunState`），backend 可 override 为原生投影
- `list_runs_filtered()` 有默认实现但 backend 应 override 为原生查询
- `append_event()` 必须原子更新 run 投影（status + total_usage + iteration + last_finish）
- 可参考 `src/runtime/memory.rs` 的 `MemoryRunStore` 实现

### 新增 ArtifactStore backend

实现 `ArtifactStore` trait（在 `src/store/mod.rs` 中定义）：

```rust
#[async_trait]
impl ArtifactStore for MyArtifactStore {
    async fn put(&self, artifact: Artifact) -> StoreResult<Artifact>;
    async fn get(&self, id: &Uuid) -> StoreResult<Option<Artifact>>;
    async fn delete(&self, id: &Uuid) -> StoreResult<()>;
    async fn list_by_session(&self, session_id: &Uuid) -> StoreResult<Vec<Artifact>>;
}
```

已有实现：`MemoryArtifactStore`（内存）、`DiskArtifactStore` / `S3ArtifactStore`（object_store feature）。

### 新增 InputAdmission 规则

在 `src/runtime/input.rs` 的 `InputAdmission::admit()` 中添加验证逻辑：

1. 在 `admit()` 方法中添加新的 `if` 分支（在现有 validate/dedup 之后）
2. 失败时调用 `record.reject(reason)` 并返回 `InputEvent::Rejected`
3. 对应的 `InputAdmissionConfig` 添加开关字段
4. 更新 `InputAdmissionConfig` 的 `Default` 实现

### 新增 BackgroundJobPool 作业类型

在 `src/runtime/job.rs` 的 `JobType` enum 中新增 variant：

1. 在 `JobType` enum 中添加新 variant（含必要 payload）
2. 在 `BackgroundJobPool::execute_job()` 中添加对应执行分支
3. 如果 variant 依赖外部 service（如 queue），使用 `#[cfg(feature = "...")]` gated

### 新增 EventPublisher backend

在 `src/queue/` 下新增文件，实现 `EventPublisher` trait：

```rust
#[async_trait]
pub trait EventPublisher: Send + Sync {
    async fn publish(&self, event: AgentEvent) -> QueueResult<()>;
}
```

1. 新增 Cargo feature（backend 名）
2. 在 `src/queue/mod.rs` 中 conditional compile
3. 在 `src/prelude.rs` 中 feature-gated re-export
4. 已有实现：`NatsEventPublisher`（nats feature）、`RedisStreamsPublisher`（redis feature）

### 新增 Compaction 阶段

Compaction pipeline 在 `src/runtime/compaction/` 下分为：

- `overflow.rs` — 检测是否需要 compaction（token overflow / message count overflow）
- `prompt.rs` — 构建 compaction LLM prompt（含 incremental summary 支持）
- `select.rs` — 选择 compact 的 head 消息和保留的 tail 消息
- `prune.rs` — 处理 post-compaction 的消息裁剪

新增阶段时：
1. 在 `compaction/mod.rs` 的 `CompactionService` 主流程中接入
2. 阶段必须 idempotent，避免在失败时产生半截 compaction
3. 关键状态变更记录到 `AgentEvent::CompactionCircuitOpened`
