# Architecture

## 概览

`agents` 是 Rust-native agent runtime library crate。采用 Hexagonal Architecture：provider trait 作为 port，adapt 层作为 adapter，runtime 作为 application core，store 作为 persistence port。

```
                                    ┌─────────────┐
                                    │  adapt/     │──→ OpenAI, Anthropic HTTP adapters
                                    └──────┬──────┘
                                           │ implements
              ┌────────────────────────────┼────────────────────────────┐
              │                   provider/                             │
              │   traits.rs ← ChatProvider / EmbeddingProvider          │
              │   registry.rs ← ProviderRegistry                        │
              │   message.rs, tool.rs, events.rs, capabilities.rs       │
              │   config.rs ← ProviderConfig, ProviderHttpConfig        │
              │   id.rs ← ProviderId, ModelName newtype                  │
              │   embedding.rs ← EmbeddingProvider trait                │
              └────────────────────────────┬────────────────────────────┘
                                           │ used by
              ┌────────────────────────────┼────────────────────────────┐
              │                    runtime/                              │
              │   agent.rs ← AgentRuntime (streaming-first loop)        │
              │   router.rs ← ModelRouter (cap check, retry, fallback)  │
              │   context.rs ← ContextPipeline (history + compaction)   │
              │   compaction/ ← CompactionService                       │
              │   turn.rs ← FSM (TurnState → TurnTransition → TurnAction)│
              │   session_gate.rs ← per-session lock                    │
              │   snapshot.rs ← FSM crash recovery                      │
              │   doom_loop.rs ← duplicate / cycle detection            │
              │   policy.rs ← RuntimePolicy (limits, budget, timeout)   │
              │   tool.rs   ← ToolRuntime (validation, timeout, concur) │
              │   event.rs  ← AgentEvent enum                           │
              │   store.rs  ← RuntimeStore + RunStore trait             │
              │   state.rs  ← RunState (event-sourced projection)       │
              │   run.rs    ← RunId, RunRequest, RunStatus              │
              │   input.rs  ← InputAdmission (validate, dedup, admit)   │
              │   job.rs    ← BackgroundJobPool (async event persist)   │
              │   accumulator.rs ← StreamAccumulator (text + tool call) │
              │   memory.rs ← MemoryRunStore                            │
              │   error.rs  ← RuntimeError enum                         │
              │   invocation.rs ← RuntimeInvocation (emit/on facade)    │
              │   event_store.rs ← RuntimeEventStore (replay source)    │
              │   stream.rs ← RuntimeEventEnvelope, RuntimeRoom         │
              │   stream_adapter.rs ← RuntimeStreamAdapter (live fanout)│
              │   subscription.rs ← RuntimeSubscriptionHub (replay+live)│
              └────────────────────────────┬────────────────────────────┘
                                           │ persists via
              ┌────────────────────────────┼────────────────────────────┐
              │                     store/                               │
              │   memory/, redis/, sql/, mongodb/, surrealdb/, qdrant/,  │
              │   object/ (S3 + disk artifact store)                    │
              │   util.rs ← shared store helpers                       │
              └─────────────────────────────────────────────────────────┘
                                           │ publishes via
              ┌────────────────────────────┼────────────────────────────┐
              │                     queue/  ──→ NATS, Redis Streams     │
              └─────────────────────────────────────────────────────────┘
              ┌────────────────────────────┼────────────────────────────┐
              │                     rag/    ──→ Qdrant, Tantivy          │
              └─────────────────────────────────────────────────────────┘
              ┌────────────────────────────┼────────────────────────────┐
              │                     config/                              │
              │   mod.rs ← AgentConfig + AgentConfigBuilder             │
              │   loader.rs ← file + env loading                        │
              │   provider.rs, runtime.rs, store.rs, rag.rs, queue.rs   │
              └─────────────────────────────────────────────────────────┘
              ┌────────────────────────────┼────────────────────────────┐
              │   context.rs ← ContextAdapter + ContextFactory           │
              │   tool.rs    ← Tool trait + ToolRegistry                 │
              │   tool_output.rs ← TruncationConfig + truncate_output    │
              │   tool_scope.rs  ← ScopedToolRegistry (LIFO shadow)      │
              │   token.rs  ← Token estimation heuristics               │
              │   error.rs  ← Public Error + ProviderError + ...         │
              │   agent/    ← AgentDefinition + AgentRegistry            │
              └─────────────────────────────────────────────────────────┘
```

依赖方向：`adapt → provider ← runtime → store`。禁止反向依赖。provider 层不得引用 adapt 或 runtime。

## 模块边界

| 模块 | 职责 | 对外暴露 |
|------|------|---------|
| `provider/` | 模型无关 trait、类型、注册表 | `ChatProvider`, `EmbeddingProvider`, `ProviderRegistry`, message/tool 类型, `ProviderConfig`, `ProviderId`, `ModelName` |
| `runtime/` | agent 执行内核：状态机、事件、策略、compaction、输入准入、后台作业、调用门面、流基础设施 | `AgentRuntime`, `RunOutput`, `AgentEvent`, `RuntimePolicy`, `ModelRouter`, `RunState`, `InputAdmission`, `BackgroundJobPool`, `ContextPipeline`, `ToolRuntime`, `RuntimeInvocation`, `EmitRequest`, `EventKind`, `Control`, `RuntimeEventStore`, `RuntimeStreamAdapter`, `RuntimeSubscriptionHub` |
| `agent/` | agent 定义（primary/subagent）和权限 | `AgentDefinition`, `AgentRegistry`, `AgentMode`, `PermissionRule`, `PermissionEffect` |
| `adapt/` | HTTP adapter 实现（OpenAI, Anthropic） | feature-gated，不暴露到 prelude |
| `store/` | 持久化抽象 | `SessionStore`, `EmbeddingStore`, `ExecutionStore`, `ArtifactStore`, `CompactionMeta`, `SessionStats`，feature-gated backends（memory/redis/sql/mongodb/surrealdb/qdrant/object） |
| `config/` | 配置加载（file + env + builder） | `AgentConfig`, `AgentConfigBuilder`, `ConfigLoader`, `ProviderConfig`, `RuntimeConfig`, `StoreConfig`, feature-gated `RagConfig`/`QueueConfig` |
| `context.rs` | 上下文组合管道 | `ContextFactory`, `ContextAdapter`, `ContextInput`, `ContextOutput`, `StaticAdapter`, `FunctionAdapter` |
| `tool.rs` | 工具 trait 与注册表 | `Tool`, `FunctionTool`, `ExternalTool`, `ToolRegistry`, `ToolOutput`, `ToolResult` |
| `tool_output.rs` | 工具输出截断（head+tail） | `ToolOutputConfig`, `TruncationResult`, `truncate_output()` |
| `tool_scope.rs` | LIFO 作用域工具注册表 | `ScopedToolRegistry`（shadow-stack: turn → run → agent → base） |
| `error.rs` | 公共错误类型 | `Error`, `ProviderError`, `ToolError`, `StorageError`, `ContextError` |
| `token.rs` | Token 估算（chars/4 启发式） | `estimate_tokens()`, `estimate_message_tokens()`, `estimate_records_tokens()` |
| `grpc/` | gRPC server（feature = `server`） | 不暴露到 prelude |
| `rag/` | RAG context adapter（feature = `rag`） | feature-gated：`RagContextAdapter`，backend: qdrant, tantivy |
| `queue/` | 外部事件发布（feature = `queue`） | feature-gated：`EventPublisher`, `NatsEventPublisher`, `RedisStreamsPublisher` |

`lib.rs` 只做两件事：声明模块 + 集中 re-export。`prelude.rs` 聚合最常用类型。

## Runtime 内核设计

### AgentRuntime 主循环

```
InputAdmission → CheckingPolicy → BuildingContext →    CallingModel
                                                        (via ModelRouter)
                                                            │
                                                            ▼
                                                    ProcessingResponse
                                                            │
                                           ┌────────────────┴────────────────┐
                                           │ tool calls?                    │
                                           ▼                                ▼
                                    ExecutingTools (via ToolRuntime)   [break loop]
                                           │
                                           ▼
                                       Persisting ─────────→ back to InputAdmission
```

### Turn FSM

每个 turn 经历 6 个 `TurnState`：
1. `CheckingPolicy` — 检查 iteration / token 预算
2. `BuildingContext` — 构建 `ChatRequest`（含 proactive compaction）
3. `CallingModel` — 通过 `ModelRouter` 路由到 provider（含 capability check + retry + fallback）
4. `ProcessingResponse` — 判断 finish_reason 是否 tool_calls
5. `ExecutingTools` — 批量执行工具（`ToolRuntime` 含 schema 校验 + timeout + 并发安全分区），含 doom loop 检测
6. `Persisting` — 写 session + 异步事件持久化到 `BackgroundJobPool`

`TurnTransition::resolve()` 决定每个状态的下一步：`Continue` / `BreakLoop` / `CompactAndRetry` / `Fail`。

### 关键组件

- **InputAdmission** — 输入生命周期管理：验证（空检查、长度限制）、去重（content fingerprint）、准入状态追踪（Submitted → Admitted → Processing → Completed/Rejected）
- **ModelRouter** — provider 路由层，封装 `ProviderRegistry`：capability 检查 + 指数退避重试 + fallback chain。同时处理 chat 和 embedding 路由
- **SessionGate** — per-session `tokio::sync::Mutex`，防止并发 run 交错写入同一 session
- **SnapshotStore** — FSM 级别的 crash recovery。每个状态转换前保存 snapshot，run 结束时删除。`AgentRuntime::resume()` 从 snapshot 恢复
- **RunState** — 事件溯源状态投影。折叠 `AgentEvent` 序列重建 run 的完整状态（status、iteration、total_usage、last_finish、last_error）
- **CompactionService** — proactive（BuildingContext 阶段）和 reactive（provider 返回 context_overflow 时）两种模式。compaction pipeline 分为 overflow 检测 → prompt 构建 → LLM summary → prune → select
- **ContextPipeline** — 运行级上下文管道：组合 `ContextFactory` adapter + session history 加载 + compaction filter + token-budget 裁剪
- **DoomLoopDetector** — 检测连续重复 tool call 和循环模式
- **ToolRuntime** — 工具执行运行时：JSON schema 校验 + per-tool timeout + 并发/排他分区（`is_concurrency_safe`）+ `ExecutionStore` 记录
- **StreamAccumulator** — 流式响应累积器：增量拼接 text delta + tool call arguments，最终输出完整 `Message::Assistant`
- **BackgroundJobPool** — 优先级感知的异步作业池：事件持久化 + 外部发布（NATS/Redis），含指数退避重试 + graceful shutdown + 磁盘持久化
- **RuntimeStore** — 运行时 store facade：组合 `SessionStore` + `ExecutionStore` + `RunStore`，提供统一的持久化接口
- **RunStore** — run 生命周期管理 trait：create_run / append_event / list_events / get_run_state（事件溯源投影）。内存实现 `MemoryRunStore`

## Store 架构

四种持久化 trait + 对应 backend：

| Trait | 职责 | 主要方法 | Backend |
|-------|------|---------|---------|
| `SessionStore` | session CRUD + 消息历史 | `create_session`, `append_message`, `list_messages`, `get_latest_compaction` | memory, sqlx (pg/mysql/sqlite), mongodb, surrealdb, redis |
| `EmbeddingStore` | 向量持久化 + 近邻搜索 | `upsert`, `search`, `delete_by_session` | memory, qdrant |
| `ExecutionStore` | 工具执行记录 + token 用量 + session 统计 | `record_execution`, `record_usage`, `session_stats` | memory |
| `ArtifactStore` | 二进制文件/附件存储 | `put`, `get`, `list_by_session`, `delete_by_session` | memory, disk, S3 (object_store) |

### 关键类型

- **`MessageRecord`** — 持久化消息记录，含 `is_compaction` / `is_summary` 标记 + `compaction_meta`（tail_start_id + previous_compaction_id + summary_text）
- **`CompactionMeta`** — 压缩元数据：tail_start_id（保留尾巴起点）、previous_compaction_id（增量摘要）、summary_text（LLM 生成的摘要文本）
- **`ToolExecution`** — 工具执行记录（session/message/call_id/tool_name/arguments/result/status/duration）
- **`UsageRecord`** — token 用量记录（session/message/provider/model/input_tokens/output_tokens）
- **`SessionStats`** — session 聚合统计（message_count/tool_call_count/total_tokens/avg_tool_duration）
- **`Artifact`** — 二进制 artifact（name/content_type/data base64）

## Tool 基础设施

三层 tool 注册与执行架构：

```
Tool trait (tool.rs)              ← 抽象：name + description + parameters_schema + execute
    │
    ├── FunctionTool<F>            ← 闭包工具：ReadOnly / ConcurrencySafe 标记
    └── ExternalTool               ← 外部工具：schema-only，execute 返回 NotImplemented

ToolRegistry (tool.rs)            ← 全局工具注册表：注册/查找/specs/执行

ToolRuntime (runtime/tool.rs)     ← 运行级工具执行：
    │                              ←   · JSON schema 校验（jsonschema crate）
    │                              ←   · per-tool timeout
    │                              ←   · 并发/排他分区（is_concurrency_safe）
    │                              ←   · ExecutionStore 记录
    │
ScopedToolRegistry (tool_scope.rs) ← LIFO shadow-stack 作用域
    │                              ←   Turn scope → Run scope → Agent scope → Base registry
    └── ToolOutputConfig (tool_output.rs) ← 输出截断：head+tail 采样 + 文件保存
```
