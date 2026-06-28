# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] — 2026-06-27

### Added

#### Managed runtime
- `ManagedRuntime`: unified container orchestrating `AgentRuntime`,
  `ComponentRegistry`, an optional `TransportHub`, and a root
  `ShutdownToken`. Provides coordinated lifecycle (`init_all → start_all → serve → stop_all`),
  typed component access, aggregated health, and hot-reload entry point.
- `AgentConfigBuilder::build_managed()`: one-call construction of a fully
  configured `ManagedRuntime`.

#### Hot-swap reload protocol
- `ManagedRuntime::reload`: drain-aware component replacement. Calls
  `pre_replace_hook` on the old instance, starts the new instance,
  atomically swaps the registry entry, then calls `post_replace_hook`.
  Returns the old `Arc<T>` for explicit drain/await.
- `ManagedRuntime::reload_raw`: type-erased counterpart accepting
  `Box<dyn AnyComponent>`.
- `Component::pre_replace_hook` / `Component::post_replace_hook`: optional
  lifecycle hooks for graceful replacement.

#### Drain helper
- `DrainGuard<T>`: reference-counted guard for tracking outstanding
  `Arc<T>` references during hot-swap. Reports drain status when all
  external references are dropped.

#### Health aggregation & endpoints
- `HealthStatus::aggregate`: worst-case aggregation over a map of
  component health statuses (healthy / degraded / unhealthy).
- `HealthStatus::healthz_response`: builds a JSON `/healthz` response
  body with overall status and per-component breakdown.
- `ManagedRuntime::overall_health`, `is_ready`, `healthz_json`:
  convenience methods for health probes and readiness gates.

#### gRPC transport component
- `GrpcTransport`: a `Transport` implementation wrapping a tonic gRPC
  server. Accepts a fully configured tonic `Router` and serves it with
  graceful shutdown.
- `TransportHub::serve_all`: blocking counterpart of `start_all` that
  waits for all transports to complete before returning.
- `TransportHub::health`: concurrent health probes across all registered
  transports.

#### Admin gRPC extensions
- `HealthCheck` and `ReadinessCheck` RPCs in `admin.proto` for external
  health and readiness probing.

#### Examples & tests
- `examples/managed_runtime.rs`: demonstrates `ManagedRuntime` lifecycle
  with health inspection and graceful shutdown.
- `examples/hot_swap.rs`: demonstrates hot-swap reload via
  `ManagedRuntime::reload` with drain-aware reference tracking.
- `tests/managed_runtime.rs`: integration tests for `ManagedRuntime`
  lifecycle, health aggregation, reload, and `healthz_json`.

### Removed

- `reasoning` module removed entirely: `ReasoningGraph`, `ReasoningOperator`, 27
  built-in operators, and the `execute_graph` scheduler. The module was an
  unmaintained IR with no integration into the runtime, no caller in examples or
  tests, and its operators were highly duplicated (each ~80 lines of the same
  prompt-and-parse shape). Removed to keep the crate honest about its scope.
  This is a breaking change to the public API (`behest::reasoning`).
- Multi-language README: removed `fr`, `it`, `ja`, `ko`, and `zh-TW`
  translations. Kept English (`README.md`) and Simplified Chinese
  (`README.zh-CN.md`).
- `examples/reasoning_graph.rs` removed alongside the module.

## [0.3.3] — 2026-06-26

### Added

#### Reasoning graph
- `ReasoningGraph`: directed acyclic graph for multi-step reasoning strategies.
- `ReasoningOperator` trait: atomic state transformation with LLM/tool integration.
- `ControlKind`: edge semantics (pipeline, branch, loop, fan-out/in).
- 22 built-in operators: decompose, analyze, synthesize, hypothesize, verify, etc.
- `CustomOperator`: user-defined operator with closure-based implementation.
- `ReasoningScheduler`: graph traversal with topological ordering.
- `ReasoningControl`: cooperative cancellation and progress tracking.

#### Examples
- Added 14 comprehensive examples covering all major features.

## [0.3.2] — 2026-06-25

### Added

#### Runtime backends
- PostgreSQL-backed `RuntimeEventStore` (`sqlx-postgres` feature).
- Redis-backed `SessionDataStore` (`redis` feature).
- NATS JetStream-backed `RuntimeStreamAdapter` (`nats` feature).
- `SessionDataStore` trait for per-session temporary key-value storage.
- `MemorySessionDataStore` and `FileSessionDataStore` implementations.

### Fixed
- Correct rustdoc intra-doc links in new backend modules.

## [0.3.1] — 2026-06-25

### Changed
- Improved rustdoc comments across 90 files with architectural context and proper intra-doc links.

## [0.3.0] — 2026-06-25

### Added

#### Runtime invocation facade
- `RuntimeInvocation`: transport-neutral emit/on semantics over `AgentRuntime`.
- `EmitRequest`: request builder with session, idempotency, and metadata.
- `EventKind`: typed event subscription with 24 agent + chat variants.
- `Control`: cooperative cancellation/timeout/concurrency hints.
- `InvocationHandle`: background listener with auto-abort on drop.
- `SessionContext`: lightweight invocation-time context carrier.

#### Runtime stream infrastructure
- `RuntimeEventStore`: authoritative replay source with at-least-once delivery.
- `RuntimeStreamAdapter`: best-effort live fanout with room-based routing (Socket.IO-inspired).
- `RuntimeSubscriptionHub`: stitches replay + live for reconnecting clients.
- `RuntimeEventBridge`: drains `AgentRuntime` events into store+adapter.
- `RuntimeEventEnvelope`: globally unique event id, seq, room routing.
- `MemoryRuntimeEventStore` / `MemoryRuntimeStreamAdapter`: in-memory implementations.

## [0.2.2] — 2026-06-25

### Fixed
- Fix docs.rs build: exclude `server` feature to avoid `OUT_DIR` not found error in `tonic::include_proto!`.

## [0.2.1] — 2026-06-25

### Added

#### Documentation
- Multilingual README: Simplified Chinese, Traditional Chinese, French, Japanese, Korean, Italian.
- Visual banner asset for README.

#### gRPC Server
- `ChatService` gRPC handler for streaming chat.
- OpenTelemetry tracing integration with W3C trace context propagation.
- `tonic-reflection` service registration.
- Health check, graceful shutdown, tracing env filter.
- Capability services for embeddings, artifacts, agents, context, compaction, snapshots, and tool registration.
- TLS support, auth interceptor, concurrency limit, validation, idempotency.

### Fixed
- Correct rustdoc intra-doc links across 6 source files.
- CI: install `protoc` for `tonic-build` server feature.

### Changed
- Refactored `grpc` module with breaking proto changes for cleaner architecture.
- Split `agent.rs` run loop into `run_loop.rs`.

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
