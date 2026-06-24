# behest

Rust-native contracts for building production AI agent runtimes: provider-neutral chat,
streaming, tool-calling, embeddings, typed errors, and an in-memory provider registry.

## Why `behest`

**behest** /bɪˈhest/ — *n.* a person's orders or command.

> At the **behest** of the user, the agent acts.

Agent 的本质是在人的指令下自主执行。不是自主意识，不是黑箱推理——而是奉人之命，代人之劳。

这个名字冷峻、克制、精确。没有「智能」「认知」「大脑」这类膨胀隐喻，只陈述一个事实：tool-calling、streaming、memory、queue——所有机制的存在原因，都是因为有人下了命令。

> 他敲下 `/deploy`。三秒后，behest 已替他调度好七个 agent。

> Status: early foundation crate. Public APIs are intentionally small and documented.

## Highlights

- Provider traits for chat and embedding adapters
- Neutral request/response models for messages, tools, streams, and embeddings
- Retry-aware provider error taxonomy
- Feature-gated dependency surface for storage, queues, RAG, and telemetry integrations
- Strict Rust and Clippy lint configuration
- CI workflow for formatting, linting, tests, and docs

## Quick start

```toml
[dependencies]
behest = "0.2"
```

```rust
use behest::prelude::*;

let request = ChatRequest::new(ModelName::new("example-model"))
    .with_user_text("Summarize this project in one sentence.");

let registry = ProviderRegistry::new();
let provider_id = ProviderId::new("my-provider");

// Register a ChatProvider implementation, then route requests through the registry.
// let response = registry.complete(&provider_id, request).await?;
```

## Crate layout

```text
src/
├── adapt/              # Provider adapters (OpenAI, Anthropic)
├── agent/              # Agent definitions, permissions, and tool scoping
├── config/             # Configuration system (file, env, builder)
├── context.rs          # Multi-adapter context factory for chat requests
├── error.rs            # Public error taxonomy
├── grpc/               # gRPC server (feature = "server")
├── lib.rs              # Crate entry point, re-exports
├── prelude.rs          # Common public imports
├── provider/           # Provider traits, request/response models, registry
├── queue/              # External event publishing (feature = "queue")
├── rag/                # Retrieval-Augmented Generation (feature = "rag")
├── runtime/            # Core runtime: FSM, session gate, snapshot, compaction
├── store/              # Persistence layer (memory, sqlx, mongodb, surrealdb)
├── token.rs            # Token counting
├── tool.rs             # Tool registry and execution
├── tool_output.rs      # Tool output truncation
└── tool_scope.rs       # Scoped tool registries per agent
```

## Development

```bash
cargo fmt --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-features --locked
cargo doc --all-features --no-deps
```

## Feature flags

- `tls-rustls` (default): Rustls TLS stack
- `tls-native`: Native TLS stack
- `openai`, `anthropic`: Provider adapters
- `redis`, `redis-cluster`: Redis store, queue, and pub/sub
- `nats`: NATS queue integration
- `sqlx-postgres`, `sqlx-mysql`, `sqlx-sqlite`: SQLx store backends
- `mongodb`, `surrealdb`: Document / multi-model stores
- `otel`: OpenTelemetry tracing
- `rag`, `qdrant`, `tantivy`: RAG context adapters
- `queue`, `queue-all`: Event publishing to message brokers
- `object_store`: Object storage (AWS S3)
- `server`: gRPC server binary
- `full`: All features except `server`
- `storage-all`: All storage backends
- `rag-all`: All RAG backends

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
