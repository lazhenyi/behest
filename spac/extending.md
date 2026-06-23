# Extending the Runtime

## Provider Adapter 开发

新增 model provider 时，按以下步骤操作：

### 1. 目录结构

```
src/adapt/<provider>/
├── mod.rs      # pub mod chat; pub mod convert; pub mod types;
├── chat.rs     # ChatProvider 实现
├── convert.rs  # 请求构建 + 响应解析 + 错误映射
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
