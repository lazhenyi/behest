<div align="center">

# behest

**Rust 原生的生产级 AI Agent 运行时构建库**

<img src="assets/banner.webp" alt="behest — Rust 原生 Agent 运行时" width="100%">

[![CI](https://github.com/lazhenyi/behest/actions/workflows/ci.yml/badge.svg)](https://github.com/lazhenyi/behest/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

[English](README.md) · **简体中文** · [繁體中文](README.zh-TW.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---

## 项目简介

`behest` 提供 provider-neutral 的契约，涵盖对话、流式传输、工具调用、嵌入、运行时执行、存储、队列、RAG、可观测性，以及可选的 gRPC 服务。

它为需要显式控制模型提供商、工具执行、持久化和运维边界而设计——而非不透明的「agent 框架」魔法。

> 状态：早期基础 crate。公共 API 刻意保持紧凑、强类型、有文档。

## 为什么叫 behest

**behest** /bɪˈhest/ — *名词* 一个人的命令或指令。

> At the **behest** of the user, the agent acts.

Agent 运行时的核心不是「自主意识」，而是受控的委托执行：用户下达意图，系统在明确边界内组合上下文、调用模型、执行工具、持久化状态、发布事件——可审计、可恢复、可限制、可替换。

`behest` 这个名字刻意避开 "brain / cognition / intelligence" 这类膨胀隐喻。它只陈述一个工程事实：

> tool-calling, streaming, memory, queue, RAG, snapshot — 所有机制的存在，都是因为有人下达了命令。

## 设计目标

- **Rust 原生优先**：类型化 API、显式错误、无隐藏运行时假设。
- **Provider-neutral 核心**：OpenAI、Anthropic、本地模型、代理或内部 provider 均可实现相同契约。
- **流式优先运行时**：agent 循环围绕流式模型事件设计，非流式作为降级方案。
- **类型化工具边界**：工具通过 JSON Schema 描述，通过显式注册表执行。
- **可插拔持久化**：默认内存，外部存储通过 feature flag 启用。
- **运维表面**：事件发布、快照、会话门控、压缩、重试策略、可选 gRPC 服务。
- **精简公共 API**：基础原语优于框架膨胀。

## 功能概览

| 领域 | 能力 |
|---|---|
| Provider 契约 | `ChatProvider`、`EmbeddingProvider`、请求/响应模型、流事件、provider 能力 |
| Provider 注册表 | 对话和嵌入 provider 的内存路由 |
| 对话模型类型 | 消息、内容部件、工具调用、响应格式、token 用量、结束原因 |
| 工具运行时 | `Tool`、`FunctionTool`、`ExternalTool`、`ToolRegistry`、schema 生成、执行分发 |
| Agent 运行时 | 上下文构建、模型调用、工具循环、会话持久化、事件发射 |
| 运行时调用 | `RuntimeInvocation`、`EmitRequest`、`EventKind`、`Control`，传输中立的 emit/on 门面 |
| 运行时流 | `RuntimeEventStore`、`RuntimeStreamAdapter`、`RuntimeSubscriptionHub`，重放 + 实时广播 |
| 推理图 | `ReasoningGraph`、`ReasoningOperator`、`ControlKind`，基于 DAG 的推理策略 |
| 运行时安全 | 会话门控、运行时策略、输入准入、死循环检测、工具输出截断 |
| 存储 | 内存存储、Redis、SQLx、MongoDB、SurrealDB、对象存储、Qdrant 嵌入 |
| 上下文与 RAG | 上下文适配器、静态/函数适配器、可选 RAG 适配器 |
| 队列 | 通过 NATS 或 Redis Streams 的可选事件发布 |
| 配置 | 构建器、基于文件的配置、环境变量加载、secret 间接引用 |
| 服务 | `server` feature 下的可选 gRPC 服务二进制文件 |
| 可观测性 | tracing 和可选 OpenTelemetry 集成 |

## 快速开始

```toml
[dependencies]
behest = "0.2"
```

创建一个 provider-neutral 的对话请求：

```rust
use behest::prelude::*;

let request = ChatRequest::new(ModelName::new("example-model"))
    .with_message(Message::system_text("You are concise."))
    .with_user_text("Summarize this project in one sentence.");
```

在注册表中注册 provider 并路由请求：

```rust
use behest::prelude::*;

let registry = ProviderRegistry::new();
let provider_id = ProviderId::new("my-provider");

// 先注册一个 ChatProvider 实现。
// registry.register_chat(my_provider);

// 然后通过中性注册表路由。
// let response = registry.complete(&provider_id, request).await?;
```

更多示例见 [`examples/`](examples/)。

## 实现自定义 Provider

`behest` 不强制将某个厂商 SDK 置于核心。为任何模型后端、网关、本地推理服务或内部 provider 实现 `ChatProvider`。

```rust
use async_trait::async_trait;
use behest::prelude::*;

struct EchoProvider {
    id: ProviderId,
}

#[async_trait]
impl ChatProvider for EchoProvider {
    fn id(&self) -> ProviderId {
        self.id.clone()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::chat()
    }

    async fn complete(&self, request: ChatRequest) -> ProviderResult<ChatResponse> {
        Ok(ChatResponse {
            provider: self.id.clone(),
            model: request.model,
            message: Message::assistant_text("echo"),
            finish_reason: FinishReason::Stop,
            usage: None,
            raw: None,
        })
    }
}
```

流式 provider 可覆写 `stream`。

## 定义和执行工具

工具是显式的运行时对象。每个工具暴露稳定的名称、人类可读的描述和 JSON Schema 参数契约。

```rust
use behest::prelude::*;
use serde_json::{json, Value};

let tool = FunctionTool::new(
    "echo",
    "Echoes the input message.",
    json!({
        "type": "object",
        "properties": {
            "message": { "type": "string" }
        },
        "required": ["message"]
    }),
    |args: Value| async move {
        Ok(args.get("message").cloned().unwrap_or(Value::Null))
    },
)
.read_only()
.concurrency_safe();

let registry = ToolRegistry::new();
registry.register(tool);
```

Provider 返回的工具调用可通过注册表执行：

```rust
use behest::prelude::*;
use serde_json::json;

let call = ToolCall::new("call_1", "echo", json!({ "message": "hello" }));
let output = registry.execute(&call).await?;
```

## 运行时模型

在运行时层，`AgentRuntime` 编排完整的 agent 循环：

```text
RunRequest
  -> 加载或创建会话
  -> 准入输入
  -> 构建上下文
  -> 调用模型 provider
  -> 流式/持久化助手输出
  -> 执行工具调用
  -> 追加工具结果
  -> 重复直到完成、限制或错误
  -> 发射 AgentEvent
```

运行时整合：

- `ProviderRegistry`
- `ContextPipeline`
- `ToolRuntime`
- `RuntimeStore`
- `RuntimePolicy`
- `CompactionService`
- `SessionGate`
- 可选事件发布器
- 可选快照存储
- 可选后台任务池

## 配置

`AgentConfig` 支持分层配置：

1. 默认值
2. 文件源
3. 环境变量
4. 手动构建器设置

```rust
use behest::prelude::*;

let config = AgentConfig::builder()
    .with_file("behest.toml")?
    .with_env("BEHEST")?
    .build()?;

let runtime = config.into_runtime().await?;
```

Secret 可通过 `env:VAR_NAME` 间接加载：

```toml
[providers.openai]
api_key = "env:OPENAI_API_KEY"
```

完整配置结构见 [`behest.toml` 示例](examples/hello_config.rs)。

## Provider 适配器

具体 provider 适配器通过 feature gate 启用。

| Feature | 适配器 | Chat | Stream | Embeddings | Tools |
|---|---|---:|---:|---:|---:|
| `openai` | `OpenAiChatAdapter`、`OpenAiEmbeddingAdapter` | 是 | 是 | 是 | 是 |
| `anthropic` | `AnthropicChatAdapter` | 是 | 是 | 否 | 是 |

启用适配器：

```toml
[dependencies]
behest = { version = "0.2", features = ["openai", "anthropic"] }
```

## Feature Flags

<details>
<summary>点击展开完整 feature 列表</summary>

**默认：**

| Feature | 说明 |
|---|---|
| `tls-rustls` | 使用 rustls 的默认 TLS 栈 |

**Provider 适配器：**

| Feature | 说明 |
|---|---|
| `openai` | OpenAI 兼容的对话和嵌入适配器 |
| `anthropic` | Anthropic 兼容的对话适配器 |

**TLS：**

| Feature | 说明 |
|---|---|
| `tls-rustls` | 为 HTTP/已启用后端启用 rustls TLS 集成 |
| `tls-native` | 为 HTTP/已启用后端启用 native TLS 集成 |

**存储：**

| Feature | 说明 |
|---|---|
| `redis` | Redis 存储支持和 Redis Streams 原语 |
| `redis-cluster` | Redis Cluster 支持；隐含 `redis` |
| `sqlx-postgres` | SQLx PostgreSQL 存储支持 |
| `sqlx-mysql` | SQLx MySQL 存储支持 |
| `sqlx-sqlite` | SQLx SQLite 存储支持 |
| `mongodb` | MongoDB 会话存储支持 |
| `surrealdb` | SurrealDB 会话存储支持 |
| `object_store` | 对象存储支持，包括 AWS S3 |
| `storage-all` | Redis、PostgreSQL、MySQL、SQLite、MongoDB 和 SurrealDB 存储 feature |

**RAG：**

| Feature | 说明 |
|---|---|
| `rag` | 核心 RAG 上下文适配器 |
| `qdrant` | Qdrant 嵌入存储后端 |
| `tantivy` | Tantivy 后端支持 |
| `rag-all` | 启用 `rag`、`qdrant` 和 `tantivy` |

**队列：**

| Feature | 说明 |
|---|---|
| `queue` | 核心事件发布器 trait |
| `nats` | NATS 事件发布器 |
| `queue-all` | 启用 `queue`、`nats` 和 `redis` |

**服务与可观测性：**

| Feature | 说明 |
|---|---|
| `server` | gRPC 服务二进制文件和 protobuf 服务层 |
| `otel` | OpenTelemetry tracing 集成 |

**便捷 profile：**

| Feature | 说明 |
|---|---|
| `full` | 开箱即用的完整运行时 profile：OpenAI、Anthropic、Redis、Redis Cluster、NATS、PostgreSQL、MongoDB、SurrealDB、OpenTelemetry、所有 RAG 后端、所有队列后端和对象存储。刻意不启用 `server`、`sqlx-mysql` 或 `sqlx-sqlite`。 |

</details>

使用选定 feature 的示例：

```toml
[dependencies]
behest = {
    version = "0.2",
    default-features = false,
    features = ["tls-rustls", "openai", "anthropic", "redis", "queue", "nats"]
}
```

## 错误模型

`behest` 暴露类型化错误类别，而非字符串化的框架失败：

- `ProviderError`
- `ToolError`
- `StorageError`
- `ContextError`
- `RuntimeError`
- 顶层 `Error`
- crate 级 `Result<T>`

Provider 错误区分不支持的能力、可重试失败、传输失败、无效响应和适配器特定错误。

工具错误区分缺失工具、无效参数、执行失败、超时和未实现的外部工具。

## Lint 策略

crate 刻意严格：

- `unsafe_code = "forbid"`
- `missing_docs = "deny"`
- `unreachable_pub = "deny"`
- `clippy::all = "deny"`
- `dbg_macro = "deny"`
- `expect_used = "deny"`
- `todo = "deny"`
- `unimplemented = "deny"`
- `unwrap_used = "deny"`

本项目将公共 API 清晰度和失败路径卫生视为运行时契约的一部分。

## 开发

```bash
# 格式化
cargo fmt --all --check

# 检查所有目标和 feature
cargo check --all-targets --all-features --locked

# Lint
cargo clippy --all-targets --all-features --locked -- -D warnings

# 测试
cargo test --all-features --locked

# 构建文档
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

运行完整本地验证集：

```bash
cargo fmt --all --check && \
cargo check --all-targets --all-features --locked && \
cargo clippy --all-targets --all-features --locked -- -D warnings && \
cargo test --all-features --locked && \
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

## 许可证

以下任一许可：

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

由您选择。
