<div align="center">

# behest

**プロダクション AI Agent ランタイムのための Rust ネイティブビルディングブロック**

<img src="assets/banner.webp" alt="behest — Rust ネイティブ Agent ランタイム" width="100%">

[![CI](https://github.com/lazhenyi/behest/actions/workflows/ci.yml/badge.svg)](https://github.com/lazhenyi/behest/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [Français](README.fr.md) · **日本語** · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---

## 概要

`behest` は、チャット、ストリーミング、ツール呼び出し、埋め込み、ランタイム実行、ストレージ、キュー、RAG、オブザーバビリティ、およびオプションの gRPC サービスのためのプロバイダー中立のコントラクトを提供します。

不透明な「エージェントフレームワーク」の魔法ではなく、モデルプロバイダー、ツール実行、永続化、運用境界を明示的に制御する必要があるシステム向けに設計されています。

> ステータス：初期基盤クレート。公開 API は意図的にコンパクトで、強く型付けされ、文書化されています。

## behest という名前

**behest** /bɪˈhest/ — *名詞* ある人の命令や指示。

> At the **behest** of the user, the agent acts.

エージェントランタイムの核心は「自律意識」ではなく、制御された委譲です：ユーザーが意図を出し、システムが明示的な境界内でコンテキストを構成し、モデルを呼び出し、ツールを実行し、状態を永続化し、イベントを発行します — 監査可能、回復可能、制約可能、置換可能。

`behest` という名前は、「brain / cognition / intelligence」のような膨張した隠喩を意図的に避けます。エンジニアリングの事実だけを述べます：

> tool-calling, streaming, memory, queue, RAG, snapshot — すべてのメカニズムは、誰かが命令を出したから存在します。

## 設計目標

- **Rust ネイティブ優先**：型付けされた API、明示的なエラー、隠れたランタイム仮定なし。
- **プロバイダー中立コア**：OpenAI、Anthropic、ローカルモデル、プロキシ、または内部プロバイダーが同じコントラクトを実装可能。
- **ストリーミングファーストランタイム**：エージェントループはストリームモデルイベントを中心に設計、非ストリーミングはフォールバック。
- **型付けツール境界**：ツールは JSON Schema で記述され、明示的なレジストリを通じて実行。
- **プラガブル永続化**：デフォルトはメモリ、外部ストレージは feature flag で有効化。
- **運用サーフェス**：イベント発行、スナップショット、セッションゲート、圧縮、リトライポリシー、オプションの gRPC サーバー。
- **小さな公開 API**：フレームワークの拡張よりも基盤プリミティブ優先。

## 機能概要

| 領域 | 機能 |
|---|---|
| プロバイダー契約 | `ChatProvider`、`EmbeddingProvider`、リクエスト/レスポンスモデル、ストリームイベント、プロバイダー能力 |
| プロバイダーレジストリ | チャットおよび埋め込みプロバイダーのインメモリルーティング |
| チャットモデルタイプ | メッセージ、コンテンツパーツ、ツール呼び出し、レスポンスフォーマット、トークン使用量、終了理由 |
| ツールランタイム | `Tool`、`FunctionTool`、`ExternalTool`、`ToolRegistry`、スキーマ生成、実行ディスパッチ |
| エージェントランタイム | コンテキスト構築、モデル呼び出し、ツールループ、セッション永続化、イベント発行 |
| ランタイム呼び出し | `RuntimeInvocation`、`EmitRequest`、`EventKind`、`Control`、トランスポート中立な emit/on ファサード |
| ランタイムストリーム | `RuntimeEventStore`、`RuntimeStreamAdapter`、`RuntimeSubscriptionHub`、リプレイ + ライブファナウト |
| ランタイムセキュリティ | セッションゲート、ランタイムポリシー、入力准入、ドゥームループ検出、ツール出力切り詰め |
| ストレージ | メモリストレージ、Redis、SQLx、MongoDB、SurrealDB、オブジェクトストレージ、Qdrant 埋め込み |
| コンテキストと RAG | コンテキストアダプター、静的/関数アダプター、オプション RAG アダプター |
| キュー | NATS または Redis Streams 経由のオプションイベント発行 |
| 設定 | ビルダー、ファイルベース設定、環境変数ロード、シークレット間接参照 |
| サーバー | `server` feature のオプション gRPC サーバーバイナリ |
| オブザーバビリティ | tracing とオプションの OpenTelemetry 統合 |

## クイックスタート

```toml
[dependencies]
behest = "0.2"
```

プロバイダー中立のチャットリクエストを作成：

```rust
use behest::prelude::*;

let request = ChatRequest::new(ModelName::new("example-model"))
    .with_message(Message::system_text("You are concise."))
    .with_user_text("Summarize this project in one sentence.");
```

レジストリにプロバイダーを登録し、リクエストをルーティング：

```rust
use behest::prelude::*;

let registry = ProviderRegistry::new();
let provider_id = ProviderId::new("my-provider");

// まず ChatProvider 実装を登録。
// registry.register_chat(my_provider);

// その後、中立レジストリを通じてルーティング。
// let response = registry.complete(&provider_id, request).await?;
```

詳細な例は [`examples/`](examples/) を参照。

## カスタムプロバイダーの実装

`behest` は特定のベンダー SDK をコアに強制しません。任意のモデルバックエンド、ゲートウェイ、ローカル推論サービス、または内部プロバイダーに `ChatProvider` を実装できます。

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

ストリーミングプロバイダーは `stream` をオーバーライドできます。

## ツールの定義と実行

ツールは明示的なランタイムオブジェクトです。各ツールは安定した名前、人間が読める説明、JSON Schema 引数コントラクトを公開します。

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

プロバイダーから返されたツール呼び出しはレジストリを通じて実行可能：

```rust
use behest::prelude::*;
use serde_json::json;

let call = ToolCall::new("call_1", "echo", json!({ "message": "hello" }));
let output = registry.execute(&call).await?;
```

## ランタイムモデル

ランタイム層では、`AgentRuntime` が完全なエージェントループをオーケストレーション：

```text
RunRequest
  -> セッションのロードまたは作成
  -> 入力の准入
  -> コンテキストの構築
  -> モデルプロバイダーの呼び出し
  -> アシスタント出力のストリーム/永続化
  -> ツール呼び出しの実行
  -> ツール結果の追加
  -> 完了、制限、またはエラーまで繰り返し
  -> AgentEvent の発行
```

ランタイムは以下を統合：

- `ProviderRegistry`
- `ContextPipeline`
- `ToolRuntime`
- `RuntimeStore`
- `RuntimePolicy`
- `CompactionService`
- `SessionGate`
- オプションのイベントパブリッシャー
- オプションのスナップショットストア
- オプションのバックグラウンドジョブプール

## 設定

`AgentConfig` は階層設定をサポート：

1. デフォルト値
2. ファイルソース
3. 環境変数
4. 手動ビルダーセッター

```rust
use behest::prelude::*;

let config = AgentConfig::builder()
    .with_file("behest.toml")?
    .with_env("BEHEST")?
    .build()?;

let runtime = config.into_runtime().await?;
```

シークレットは `env:VAR_NAME` 間接参照でロード可能：

```toml
[providers.openai]
api_key = "env:OPENAI_API_KEY"
```

完全な設定構造は [`behest.toml` の例](examples/hello_config.rs) を参照。

## プロバイダーアダプター

具体的なプロバイダーアダプターは feature gate で有効化。

| Feature | アダプター | Chat | Stream | Embeddings | Tools |
|---|---|---:|---:|---:|---:|
| `openai` | `OpenAiChatAdapter`、`OpenAiEmbeddingAdapter` | はい | はい | はい | はい |
| `anthropic` | `AnthropicChatAdapter` | はい | はい | いいえ | はい |

アダプターの有効化：

```toml
[dependencies]
behest = { version = "0.2", features = ["openai", "anthropic"] }
```

## Feature Flags

<details>
<summary>完全な feature リストを展開</summary>

**デフォルト：**

| Feature | 説明 |
|---|---|
| `tls-rustls` | rustls を使用するデフォルト TLS スタック |

**プロバイダーアダプター：**

| Feature | 説明 |
|---|---|
| `openai` | OpenAI 互換チャットおよび埋め込みアダプター |
| `anthropic` | Anthropic 互換チャットアダプター |

**TLS：**

| Feature | 説明 |
|---|---|
| `tls-rustls` | HTTP / 有効バックエンド用の rustls TLS 統合を有効化 |
| `tls-native` | HTTP / 有効バックエンド用のネイティブ TLS 統合を有効化 |

**ストレージ：**

| Feature | 説明 |
|---|---|
| `redis` | Redis ベースストレージサポートと Redis Streams プリミティブ |
| `redis-cluster` | Redis Cluster サポート；`redis` を含む |
| `sqlx-postgres` | SQLx PostgreSQL ストレージサポート |
| `sqlx-mysql` | SQLx MySQL ストレージサポート |
| `sqlx-sqlite` | SQLx SQLite ストレージサポート |
| `mongodb` | MongoDB セッションストレージサポート |
| `surrealdb` | SurrealDB セッションストレージサポート |
| `object_store` | AWS S3 を含むオブジェクトストレージサポート |
| `storage-all` | Redis、PostgreSQL、MySQL、SQLite、MongoDB、SurrealDB ストレージ feature |

**RAG：**

| Feature | 説明 |
|---|---|
| `rag` | コア RAG コンテキストアダプター |
| `qdrant` | Qdrant 埋め込みストレージバックエンド |
| `tantivy` | Tantivy バックエンドサポート |
| `rag-all` | `rag`、`qdrant`、`tantivy` を有効化 |

**キュー：**

| Feature | 説明 |
|---|---|
| `queue` | コアイベントパブリッシャートレイト |
| `nats` | NATS イベントパブリッシャー |
| `queue-all` | `queue`、`nats`、`redis` を有効化 |

**サーバーとオブザーバビリティ：**

| Feature | 説明 |
|---|---|
| `server` | gRPC サーバーバイナリと protobuf サービス層 |
| `otel` | OpenTelemetry tracing 統合 |

**便利プロファイル：**

| Feature | 説明 |
|---|---|
| `full` | 完全なランタイムプロファイル：OpenAI、Anthropic、Redis、Redis Cluster、NATS、PostgreSQL、MongoDB、SurrealDB、OpenTelemetry、すべての RAG バックエンド、すべてのキューバックエンド、オブジェクトストレージ。`server`、`sqlx-mysql`、`sqlx-sqlite` は意図的に有効化しません。 |

</details>

選択した feature の例：

```toml
[dependencies]
behest = {
    version = "0.2",
    default-features = false,
    features = ["tls-rustls", "openai", "anthropic", "redis", "queue", "nats"]
}
```

## エラーモデル

`behest` は文字列化フレームワーク障害ではなく、型付けされたエラーカテゴリを公開：

- `ProviderError`
- `ToolError`
- `StorageError`
- `ContextError`
- `RuntimeError`
- トップレベル `Error`
- クレートレベル `Result<T>`

プロバイダーエラーは未サポート能力、リトライ可能障害、トランスポート障害、無効レスポンス、アダプター固有エラーを区分。

ツールエラーは欠落ツール、無効引数、実行障害、タイムアウト、未実装外部ツールを区分。

## Lint ポリシー

クレートは意図的に厳格：

- `unsafe_code = "forbid"`
- `missing_docs = "deny"`
- `unreachable_pub = "deny"`
- `clippy::all = "deny"`
- `dbg_macro = "deny"`
- `expect_used = "deny"`
- `todo = "deny"`
- `unimplemented = "deny"`
- `unwrap_used = "deny"`

このプロジェクトは公開 API の明確さと失敗パスの衛生をランタイムコントラクトの一部として扱います。

## 開発

```bash
# フォーマット
cargo fmt --all --check

# すべてのターゲットと feature をチェック
cargo check --all-targets --all-features --locked

# Lint
cargo clippy --all-targets --all-features --locked -- -D warnings

# テスト
cargo test --all-features --locked

# ドキュメント構築
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

完全なローカル検証セットを実行：

```bash
cargo fmt --all --check && \
cargo check --all-targets --all-features --locked && \
cargo clippy --all-targets --all-features --locked -- -D warnings && \
cargo test --all-features --locked && \
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

## ライセンス

以下のいずれか：

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

お選びください。
