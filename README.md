# agents

Rust-native contracts for building production AI agent runtimes: provider-neutral chat,
streaming, tool-calling, embeddings, typed errors, and an in-memory provider registry.

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
agents = "0.1"
```

```rust
use agents::prelude::*;

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
├── error.rs              # Public error taxonomy and Result aliases
├── lib.rs                # Crate entry point and documentation policy
├── prelude.rs            # Common public imports
└── provider/
    ├── capabilities.rs   # Provider feature flags
    ├── config.rs         # Shared HTTP provider config
    ├── embedding.rs      # Embedding request/response contracts
    ├── events.rs         # Chat streaming events
    ├── id.rs             # Strongly typed IDs
    ├── message.rs        # Chat messages and responses
    ├── registry.rs       # Provider registry and dispatch
    ├── tool.rs           # Tool specs and calls
    └── traits.rs         # ChatProvider / EmbeddingProvider traits
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
- `redis`, `redis-cluster`: Redis integrations
- `nats`: NATS queue integration
- `sqlx-postgres`, `sqlx-mysql`, `sqlx-sqlite`: SQLx backends
- `mongodb`, `surrealdb`: Document / multi-model stores
- `otel`: OpenTelemetry integration
- `rag`, `qdrant`, `tantivy`: Retrieval-oriented integrations

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
