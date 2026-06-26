<div align="center">

# behest

**Rust 原生的生產級 AI Agent 運行時構建庫**

<img src="assets/banner.webp" alt="behest — Rust 原生 Agent 運行時" width="100%">

[![CI](https://github.com/lazhenyi/behest/actions/workflows/ci.yml/badge.svg)](https://github.com/lazhenyi/behest/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

[English](README.md) · [简体中文](README.zh-CN.md) · **繁體中文** · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---

## 專案簡介

`behest` 提供 provider-neutral 的契約，涵蓋對話、串流傳輸、工具呼叫、嵌入、運行時執行、儲存、佇列、RAG、可觀測性，以及可選的 gRPC 服務。

它為需要明確控制模型供應商、工具執行、持久化和營運邊界而設計——而非不透明的「agent 框架」魔法。

> 狀態：早期基礎 crate。公共 API 刻意保持緊湊、強型別、有文件。

## 為什麼叫 behest

**behest** /bɪˈhest/ — *名詞* 一個人的命令或指令。

> At the **behest** of the user, the agent acts.

Agent 運行時的核心不是「自主意識」，而是受控的委託執行：使用者下達意圖，系統在明確邊界內組合上下文、呼叫模型、執行工具、持久化狀態、發佈事件——可稽核、可恢復、可限制、可替換。

`behest` 這個名字刻意避開 "brain / cognition / intelligence" 這類膨脹隱喻。它只陳述一個工程事實：

> tool-calling, streaming, memory, queue, RAG, snapshot — 所有機制的存在，都是因為有人下達了命令。

## 設計目標

- **Rust 原生優先**：型別化 API、明確錯誤、無隱藏運行時假設。
- **Provider-neutral 核心**：OpenAI、Anthropic、本地模型、代理或內部 provider 均可實現相同契約。
- **串流優先運行時**：agent 迴圈圍繞串流模型事件設計，非串流作為降級方案。
- **型別化工具邊界**：工具透過 JSON Schema 描述，透過明確註冊表執行。
- **可插拔持久化**：預設記憶體，外部儲存透過 feature flag 啟用。
- **營運表面**：事件發佈、快照、閘道壓縮、重試策略、可選 gRPC 服務。
- **精簡公共 API**：基礎原語優於框架膨脹。

## 功能概覽

| 領域 | 能力 |
|---|---|
| Provider 契約 | `ChatProvider`、`EmbeddingProvider`、請求/回應模型、串流事件、provider 能力 |
| Provider 註冊表 | 對話和嵌入 provider 的記憶體路由 |
| 對話模型型別 | 訊息、內容部件、工具呼叫、回應格式、token 用量、結束原因 |
| 工具運行時 | `Tool`、`FunctionTool`、`ExternalTool`、`ToolRegistry`、schema 生成、執行分發 |
| Agent 運行時 | 上下文建構、模型呼叫、工具迴圈、會話持久化、事件發射 |
| 運行時調用 | `RuntimeInvocation`、`EmitRequest`、`EventKind`、`Control`，傳輸中立的 emit/on 門面 |
| 運行時串流 | `RuntimeEventStore`、`RuntimeStreamAdapter`、`RuntimeSubscriptionHub`，重放 + 即時廣播 |
| 推理圖 | `ReasoningGraph`、`ReasoningOperator`、`ControlKind`，基於 DAG 的推理策略 |
| 運行時安全 | 閘道壓縮、運行時策略、輸入准入、死迴圈偵測、工具輸出截斷 |
| 儲存 | 記憶體儲存、Redis、SQLx、MongoDB、SurrealDB、物件儲存、Qdrant 嵌入 |
| 上下文與 RAG | 上下文配接器、靜態/函式配接器、可選 RAG 配接器 |
| 佇列 | 透過 NATS 或 Redis Streams 的可選事件發佈 |
| 配置 | 建構器、基於檔案的配置、環境變數載入、secret 間接引用 |
| 服務 | `server` feature 下的可選 gRPC 服務二進位檔 |
| 可觀測性 | tracing 和可選 OpenTelemetry 整合 |

## 快速開始

```toml
[dependencies]
behest = "0.2"
```

建立一個 provider-neutral 的對話請求：

```rust
use behest::prelude::*;

let request = ChatRequest::new(ModelName::new("example-model"))
    .with_message(Message::system_text("You are concise."))
    .with_user_text("Summarize this project in one sentence.");
```

在註冊表中註冊 provider 並路由請求：

```rust
use behest::prelude::*;

let registry = ProviderRegistry::new();
let provider_id = ProviderId::new("my-provider");

// 先註冊一個 ChatProvider 實現。
// registry.register_chat(my_provider);

// 然後透過中性註冊表路由。
// let response = registry.complete(&provider_id, request).await?;
```

更多範例見 [`examples/`](examples/)。

## 實現自訂 Provider

`behest` 不強制將某個廠商 SDK 置於核心。為任何模型後端、閘道、本地推理服務或內部 provider 實現 `ChatProvider`。

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

串流 provider 可覆寫 `stream`。

## 定義和執行工具

工具是明確的運行時物件。每個工具暴露穩定的名稱、人類可讀的描述和 JSON Schema 參數契約。

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

Provider 回傳的工具呼叫可透過註冊表執行：

```rust
use behest::prelude::*;
use serde_json::json;

let call = ToolCall::new("call_1", "echo", json!({ "message": "hello" }));
let output = registry.execute(&call).await?;
```

## 運行時模型

在運行時層，`AgentRuntime` 編排完整的 agent 迴圈：

```text
RunRequest
  -> 載入或建立會話
  -> 准入輸入
  -> 建構上下文
  -> 呼叫模型 provider
  -> 串流/持久化助手輸出
  -> 執行工具呼叫
  -> 追加工具結果
  -> 重複直到完成、限制或錯誤
  -> 發射 AgentEvent
```

運行時整合：

- `ProviderRegistry`
- `ContextPipeline`
- `ToolRuntime`
- `RuntimeStore`
- `RuntimePolicy`
- `CompactionService`
- `SessionGate`
- 可選事件發佈器
- 可選快照儲存
- 可選背景任務池

## 配置

`AgentConfig` 支援分層配置：

1. 預設值
2. 檔案來源
3. 環境變數
4. 手動建構器設定

```rust
use behest::prelude::*;

let config = AgentConfig::builder()
    .with_file("behest.toml")?
    .with_env("BEHEST")?
    .build()?;

let runtime = config.into_runtime().await?;
```

Secret 可透過 `env:VAR_NAME` 間接載入：

```toml
[providers.openai]
api_key = "env:OPENAI_API_KEY"
```

完整配置結構見 [`behest.toml` 範例](examples/hello_config.rs)。

## Provider 配接器

具體 provider 配接器透過 feature gate 啟用。

| Feature | 配接器 | Chat | Stream | Embeddings | Tools |
|---|---|---:|---:|---:|---:|
| `openai` | `OpenAiChatAdapter`、`OpenAiEmbeddingAdapter` | 是 | 是 | 是 | 是 |
| `anthropic` | `AnthropicChatAdapter` | 是 | 是 | 否 | 是 |

啟用配接器：

```toml
[dependencies]
behest = { version = "0.2", features = ["openai", "anthropic"] }
```

## Feature Flags

<details>
<summary>點擊展開完整 feature 列表</summary>

**預設：**

| Feature | 說明 |
|---|---|
| `tls-rustls` | 使用 rustls 的預設 TLS 棧 |

**Provider 配接器：**

| Feature | 說明 |
|---|---|
| `openai` | OpenAI 相容的對話和嵌入配接器 |
| `anthropic` | Anthropic 相容的對話配接器 |

**TLS：**

| Feature | 說明 |
|---|---|
| `tls-rustls` | 為 HTTP/已啟用後端啟用 rustls TLS 整合 |
| `tls-native` | 為 HTTP/已啟用後端啟用 native TLS 整合 |

**儲存：**

| Feature | 說明 |
|---|---|
| `redis` | Redis 儲存支援和 Redis Streams 原語 |
| `redis-cluster` | Redis Cluster 支援；隱含 `redis` |
| `sqlx-postgres` | SQLx PostgreSQL 儲存支援 |
| `sqlx-mysql` | SQLx MySQL 儲存支援 |
| `sqlx-sqlite` | SQLx SQLite 儲存支援 |
| `mongodb` | MongoDB 會話儲存支援 |
| `surrealdb` | SurrealDB 會話儲存支援 |
| `object_store` | 物件儲存支援，包括 AWS S3 |
| `storage-all` | Redis、PostgreSQL、MySQL、SQLite、MongoDB 和 SurrealDB 儲存 feature |

**RAG：**

| Feature | 說明 |
|---|---|
| `rag` | 核心 RAG 上下文配接器 |
| `qdrant` | Qdrant 嵌入儲存後端 |
| `tantivy` | Tantivy 後端支援 |
| `rag-all` | 啟用 `rag`、`qdrant` 和 `tantivy` |

**佇列：**

| Feature | 說明 |
|---|---|
| `queue` | 核心事件發佈器 trait |
| `nats` | NATS 事件發佈器 |
| `queue-all` | 啟用 `queue`、`nats` 和 `redis` |

**服務與可觀測性：**

| Feature | 說明 |
|---|---|
| `server` | gRPC 服務二進位檔和 protobuf 服務層 |
| `otel` | OpenTelemetry tracing 整合 |

**便捷 profile：**

| Feature | 說明 |
|---|---|
| `full` | 開箱即用的完整運行時 profile：OpenAI、Anthropic、Redis、Redis Cluster、NATS、PostgreSQL、MongoDB、SurrealDB、OpenTelemetry、所有 RAG 後端、所有佇列後端和物件儲存。刻意不啟用 `server`、`sqlx-mysql` 或 `sqlx-sqlite`。 |

</details>

使用選定 feature 的範例：

```toml
[dependencies]
behest = {
    version = "0.2",
    default-features = false,
    features = ["tls-rustls", "openai", "anthropic", "redis", "queue", "nats"]
}
```

## 錯誤模型

`behest` 暴露型別化錯誤類別，而非字串化的框架失敗：

- `ProviderError`
- `ToolError`
- `StorageError`
- `ContextError`
- `RuntimeError`
- 頂層 `Error`
- crate 級 `Result<T>`

Provider 錯誤區分不支援的能力、可重試失敗、傳輸失敗、無效回應和配接器特定錯誤。

工具錯誤區分缺失工具、無效參數、執行失敗、逾時和未實現的外部工具。

## Lint 策略

crate 刻意嚴格：

- `unsafe_code = "forbid"`
- `missing_docs = "deny"`
- `unreachable_pub = "deny"`
- `clippy::all = "deny"`
- `dbg_macro = "deny"`
- `expect_used = "deny"`
- `todo = "deny"`
- `unimplemented = "deny"`
- `unwrap_used = "deny"`

本專案將公共 API 清晰度和失敗路徑衛生視為運行時契約的一部分。

## 開發

```bash
# 格式化
cargo fmt --all --check

# 檢查所有目標和 feature
cargo check --all-targets --all-features --locked

# Lint
cargo clippy --all-targets --all-features --locked -- -D warnings

# 測試
cargo test --all-features --locked

# 建構文件
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

執行完整本機驗證集：

```bash
cargo fmt --all --check && \
cargo check --all-targets --all-features --locked && \
cargo clippy --all-targets --all-features --locked -- -D warnings && \
cargo test --all-features --locked && \
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

## 授權條款

以下任一授權：

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

由您選擇。
