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
              └────────────────────────────┬────────────────────────────┘
                                           │ used by
              ┌────────────────────────────┼────────────────────────────┐
              │                    runtime/                              │
              │   agent.rs ← AgentRuntime (streaming-first loop)        │
              │   run.rs   ← RunId, RunRequest, RunStatus               │
              │   context.rs ← ContextPipeline                          │
              │   compaction/ ← CompactionService                       │
              │   turn.rs ← FSM (TurnState → TurnTransition → TurnAction)│
              │   session_gate.rs ← per-session lock                    │
              │   snapshot.rs ← FSM crash recovery                      │
              │   doom_loop.rs ← duplicate / cycle detection            │
              │   policy.rs ← RuntimePolicy (limits, budget, timeout)   │
              │   tool.rs   ← ToolRuntime                               │
              │   event.rs  ← AgentEvent enum                           │
              │   store.rs  ← RuntimeStore (session + execution + run)  │
              └────────────────────────────┬────────────────────────────┘
                                           │ persists via
              ┌────────────────────────────┼────────────────────────────┐
              │                     store/                               │
              │   memory/, redis/, sql/, mongodb/, surrealdb/, qdrant/   │
              └─────────────────────────────────────────────────────────┘
```

依赖方向：`adapt → provider ← runtime → store`。禁止反向依赖。provider 层不得引用 adapt 或 runtime。

## 模块边界

| 模块 | 职责 | 对外暴露 |
|------|------|---------|
| `provider/` | 模型无关 trait、类型、注册表 | `ChatProvider`, `EmbeddingProvider`, `ProviderRegistry`, message/tool 类型 |
| `runtime/` | agent 执行内核：状态机、事件、策略、compaction | `AgentRuntime`, `RunOutput`, `AgentEvent`, `RuntimePolicy` |
| `agent/` | agent 定义（primary/subagent）和权限 | `AgentDefinition`, `AgentRegistry`, `PermissionRule` |
| `adapt/` | HTTP adapter 实现（OpenAI, Anthropic） | feature-gated，不暴露到 prelude |
| `store/` | 持久化抽象 | `SessionStore`, `EmbeddingStore`, `ExecutionStore`，feature-gated backends |
| `config/` | 配置加载 | `AgentConfig`, per-domain config structs |
| `context/` | 上下文组合管道 | `ContextFactory`, `ContextAdapter`, `ContextInput` |
| `tool/` | 工具注册与执行 | `ToolRegistry`, `FunctionTool`, `ToolResult` |
| `error.rs` | 公共错误类型 | `Error`, `ProviderError`, `ToolError`, `StorageError`, `ContextError` |
| `grpc/` | gRPC server（feature = `server`） | 不暴露到 prelude |
| `rag/` | RAG adapter（feature = `rag`） | feature-gated |
| `queue/` | 外部事件发布（feature = `queue`） | feature-gated |

`lib.rs` 只做两件事：声明模块 + 集中 re-export。`prelude.rs` 聚合最常用类型。

## Runtime 内核设计

### AgentRuntime 主循环

```
CheckingPolicy → BuildingContext → CallingModel → ProcessingResponse
                                                     │
                                        ┌────────────┴────────────┐
                                        │ tool calls?              │
                                        ▼                         ▼
                                   ExecutingTools             [break loop]
                                        │
                                        ▼
                                    Persisting ─────────→ back to CheckingPolicy
```

### Turn FSM

每个 turn 经历 6 个 `TurnState`：
1. `CheckingPolicy` — 检查 iteration / token 预算
2. `BuildingContext` — 构建 `ChatRequest`（含 proactive compaction）
3. `CallingModel` — streaming 优先，失败 fallback 到 non-streaming
4. `ProcessingResponse` — 判断 finish_reason 是否 tool_calls
5. `ExecutingTools` — 批量执行工具，含 doom loop 检测
6. `Persisting` — 写 session

`TurnTransition::resolve()` 决定每个状态的下一步：`Continue` / `BreakLoop` / `CompactAndRetry` / `Fail`。

### 关键组件

- **SessionGate** — per-session `tokio::sync::Mutex`，防止并发 run 交错写入同一 session
- **SnapshotStore** — FSM 级别的 crash recovery。每个状态转换前保存 snapshot，run 结束时删除。`AgentRuntime::resume()` 从 snapshot 恢复
- **CompactionService** — proactive（BuildingContext 阶段）和 reactive（provider 返回 context_overflow 时）两种模式
- **DoomLoopDetector** — 检测连续重复 tool call 和循环模式
- **BackgroundJobPool** — 异步事件持久化和外部发布
