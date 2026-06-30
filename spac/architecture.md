# Architecture

## 概览

`behest` 是 Rust-native agent runtime workspace。当前组织采用 facade + domain crate 的结构：

```text
behest-core
  ├─ behest-provider
  │    ├─ behest-adapter-openai
  │    └─ behest-adapter-anthropic
  ├─ behest-store
  ├─ behest-context
  ├─ behest-tool
  ├─ behest-event
  ├─ behest-memory
  └─ behest-approval

behest-runtime

behest
  └─ facade: re-export + config + high-level runtime surface
```

核心依赖方向：

```text
adapter -> provider -> core
store   -> provider + core
runtime -> provider + store + tool + context + event + memory + approval + core
behest  -> facade over domain crates
```

禁止反向依赖：

- `behest-provider` 不得依赖 `behest-runtime` 或 `behest`。
- `behest-store` 不得依赖 `behest`。
- adapter crate 不得依赖 runtime。
- `behest-runtime` 不得依赖 facade crate `behest`。

## Workspace 边界

| Crate | 职责 |
|---|---|
| `behest-core` | 基础类型：message、error、id、token、sans-IO run state machine |
| `behest-provider` | Provider-neutral traits、request/response、stream event、provider registry、HTTP provider config |
| `behest-store` | Session/embedding/artifact/execution store traits 与 feature-gated backends |
| `behest-adapter-openai` | OpenAI-compatible chat 与 embedding adapter |
| `behest-adapter-anthropic` | Anthropic chat adapter |
| `behest-context` | 分层 runtime context traits |
| `behest-tool` | Tool trait、registry、execution strategy |
| `behest-event` | Agent event、hook、event actions |
| `behest-memory` | Conversation memory、active window、compaction contracts |
| `behest-approval` | Human-in-the-loop tool approval gate |
| `behest-runtime` | 不依赖 facade 的 runtime glue crate |
| `behest` | 对外 facade：稳定旧路径、config、high-level runtime modules、prelude |

## Facade 策略

`behest` 保留旧路径以降低 breaking 面：

```text
behest::provider -> behest-provider
behest::store    -> behest-store
behest::adapt    -> behest-adapter-openai / behest-adapter-anthropic
```

这些 facade module 只做 re-export，不再保存第二套实现。

## Feature Flags

Feature 从 facade 向下转发：

- `openai` -> `behest-adapter-openai`
- `anthropic` -> `behest-adapter-anthropic`
- `redis` / `sqlx-*` / `mongodb` / `qdrant` / `object_store` -> `behest-store`
- `tls-rustls` / `tls-native` -> provider adapters 与 store backends

当前没有 `server` / gRPC feature。文档不得声明不存在的 transport 或 binary。

## Runtime 说明

`behest` 仍保留 high-level runtime surface，包括：

- `AgentRuntime`
- `ManagedRuntime`
- `RuntimeInvocation`
- `RuntimeEventStore`
- `RuntimeStreamAdapter`
- `RuntimeSubscriptionHub`
- `RuntimePolicy`
- `ToolRuntime`

`behest-runtime` 是不依赖 facade 的 runtime glue crate。后续如果继续收敛，应逐步把 `behest/src/runtime` 的高层实现下沉到 `behest-runtime`，再让 `behest` 只 re-export runtime API。

## Store 架构

`behest-store` 提供四类持久化 trait：

| Trait | 职责 |
|---|---|
| `SessionStore` | session CRUD 与消息历史 |
| `EmbeddingStore` | 向量持久化与近邻搜索 |
| `ArtifactStore` | 文件、图片、附件等二进制 artifact |
| `ExecutionStore` | tool execution、usage、session stats |

Backends：

- always available: memory
- feature-gated: Redis, SQLx PostgreSQL/MySQL/SQLite, MongoDB, Qdrant, object store

## Adapter 架构

Adapter crate 只依赖 `behest-provider` 和底层 HTTP 依赖，不依赖 runtime。

```text
behest-adapter-openai     -> implements ChatProvider + EmbeddingProvider
behest-adapter-anthropic  -> implements ChatProvider
```

## 质量边界

- public enum 使用 `#[non_exhaustive]` 后，跨 crate match 必须有 wildcard fallback。
- workspace check 必须覆盖 `--all-targets --all-features`。
- facade 目录不得保留 dead implementation 文件。
