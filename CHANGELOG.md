# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-06-24

This is the first public release under the `behest` crate name.

### Added

#### Runtime kernel
- Agent runtime kernel with streaming execution, event persistence, and FSM turn machine.
- TurnState FSM with `TurnTransition` resolver.
- Session-level concurrency gate preventing overlapping runs per session.
- Background job pool with priority scheduling and optional integration.
- Event-sourced `RunState` projection with replay-based recovery.
- Snapshot-based crash recovery (`Snapshot`, `SnapshotStore`, `FileSnapshotStore`).
- Doom loop detector with tool-call fingerprinting.
- Compaction engine: overflow, prompt, prune, select, and modifier strategies.
- Compaction circuit breaker with failure threshold.
- Reactive overflow retry with output recovery attempts.
- Tool output truncation for context-bound responses.
- Event-sourced input admission pipeline.

#### Configuration
- `AgentConfig` module with layered configuration loading (TOML, JSON, ENV).
- `AgentConfigBuilder` with `with_env()` prefix-bound env loading.
- Env var placeholder substitution (`${VAR}` / `${VAR:-default}`) in config values.
- `env:` prefix indirection for provider API keys (secret-safe resolution).
- Model catalog, compaction model routing, `CompactionConfig` and `ToolOutputConfig`.

#### Provider adapters
- OpenAI-compatible adapter: chat completion, streaming, embeddings, tool calling.
- Anthropic Claude adapter: chat completion, streaming, tool calling.
- Shared SSE stream parser, rate-limit handling, retry-after parsing.

#### Store / Persistence
- Pluggable store backends: Memory, Redis, SQL (SQLx PostgreSQL/MySQL/SQLite), MongoDB, SurrealDB.
- Session store (`SessionStore` trait) with CRUD and compaction-aware API.
- Execution store for event persistence and replay.
- Run store with filtered listing.
- Embedding store for RAG context.
- Artifact store for managed outputs.

#### Tool System
- `Tool` trait with runtime tool registry and execution engine.
- `FunctionTool` builder: name, description, parameters, async handler.
- Scoped tool registry with run-level isolation.
- Per-tool flags: `is_read_only`, `is_concurrency_safe`.
- Concurrency-safe tool call partitioning.
- `ToolError` integration in agent outcomes.

#### Event System
- `AgentEvent` enum: `RunStarted`, `RunCompleted`, `ToolCalled`, `ToolCompleted`, `MessageReceived`, `AgentError`, `ContextOverflow`, `DoomLoopDetected`, `CompactionCircuitOpened`, `OutputTruncated`.
- Event publisher abstraction with NATS and Redis Streams backends.
- gRPC event streaming with server-sent mapping.

#### gRPC Server (`agent-server` binary)
- gRPC service layer: `Run`, `Session`, `Provider`, `Tool`, `Event`, `State`, `Usage`.
- Full event type mappings in gRPC layer.

#### Queue
- Async event publisher pipeline with `NATS` and `Redis Streams` backends.
- Configurable subject / stream key per backend.

#### RAG (Retrieval-Augmented Generation)
- Context adapter with embedding-based retrieval.
- Backend integration: `Qdrant` (gRPC), `Tantivy` (local full-text).

#### Agent Registry
- `AgentRegistry` with `AgentDefinition`: name, description, system prompt, tools, permissions.
- Agent permission model.

#### Token Estimation
- Token estimation utilities for context window budget management.

#### Documentation & Tooling
- Project engineering specs (`spac/architecture.md`, `spac/conventions.md`, `spac/testing.md`, `spac/extending.md`, `spac/contributing.md`).
- `.env.example` template covering all configurable sections.
- `examples/` directory with runnable samples: `hello_config`, `tool_registry`, `session_lifecycle`, `provider_setup`, `event_subscription`.

### Changed

- **Breaking**: crate renamed from `agents` to `behest`.
- `#[non_exhaustive]` on public config structs (`RuntimePolicyConfig`, `ToolOutputConfig`).
- Tool scope specs sorted alphabetically for deterministic prompt caching.

### Fixed
- Queue `RunStarted` test now includes `provider`/`model` fields.
- Doom loop detector wired at both `execute` and `resume` call sites.
