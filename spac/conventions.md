# Code Conventions

## 类型系统

- **newtype 强制使用**：`ProviderId`, `ModelName`, `RunId`, `ToolCallId` 等均用 newtype 包裹 `String` / `Uuid`，禁止裸类型
- **builder pattern**：参数超过 3 个的 struct 必须提供 builder。builder 方法返回 `Self`，标注 `#[must_use]`
- **`#[non_exhaustive]`**：所有 public enum 必须加，保护 SemVer 兼容性
- **`impl Default`**：所有配置 struct 提供合理的 `Default` 实现

## 错误处理

- **thiserror**：所有 public error 必须 `#[derive(Error)]`
- **禁止 anyhow 污染 public API**：`anyhow` 不出现在 `pub fn` 签名或 public type 中
- **语义化 variant**：
  - `ProviderError` 区分 `Authentication` / `RateLimited` / `Timeout` / `Overloaded` / `Unsupported` / `Transport` / `Decode` / `BadRequest` / `Provider`
  - `StorageError` 区分 `NotFound` / `ConnectionFailed` / `SerializationFailed` / `DataCorruption` / `MigrationFailed`
  - 不得将不同错误压扁成字符串
- **retryable 标记**：`ProviderError::is_retryable()` 和 `is_context_overflow()` 由 provider error 自身提供，调用方不必了解 provider 细节
- **source chain**：lower-level error 通过 `#[source]` 保留，不丢弃

## 异步

- **`async-trait`**：provider trait 使用 `#[async_trait]`
- **`Pin<Box<dyn Stream>>`**：streaming 返回类型统一使用 `ChatStream` type alias
- **Tokio**：默认 multi-thread runtime，使用 `sync`, `time`, `signal` features
- **timeout 强制**：所有 provider 调用通过 `tokio::time::timeout` 包裹，由 `RuntimePolicy::provider_timeout` 控制

## 模块组织

- `context.rs` / `tool.rs` / `tool_output.rs` / `tool_scope.rs` / `token.rs` / `error.rs` — 单文件模块，位于 `src/` 根
- `config/` / `agent/` / `provider/` / `runtime/` / `store/` / `adapt/` / `rag/` / `queue/` / `grpc/` — 目录模块，含 `mod.rs` 子模块 gate
- 运行时内部子模块（`runtime/accumulator.rs` 等）通过 `runtime/mod.rs` 声明并 re-export 公共符号

## 代码规模

- 文件不超过 1000 行
- 函数不超过 200 行
- 禁止 `TODO` / `FIXME` / 临时代码 / 注释掉的 dead code

## 格式化

```toml
# rustfmt.toml
edition = "2024"
max_width = 100
newline_style = "Unix"
use_field_init_shorthand = true
use_try_shorthand = true
```

## 禁止项

- `unsafe` — crate-level `#![forbid(unsafe_code)]`
- `missing_docs` — `#![deny(missing_docs)]`，所有 public API 必须有文档
- `unreachable_pub` — `#![deny(unreachable_pub)]`，禁止未用 `pub` 实际不可达的声明
- `rust_2018_idioms` — `#![warn(rust_2018_idioms)]`
- `unwrap()` / `expect()` — clippy `unwrap_used = "deny"`, `expect_used = "deny"`。测试中通过 `#[allow(clippy::unwrap_used)]` 豁免
- `todo!()` / `unimplemented!()` — clippy `todo = "deny"`, `unimplemented = "deny"`
- `dbg!()` — clippy `dbg_macro = "deny"`
- `println!()` — 使用 `tracing` 宏替代

## 质量门禁

每次改动必须通过，与 CI 序列完全一致：

```bash
cargo fmt --all --check
cargo check --all-targets --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-features --locked
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

### Clippy 配置

```toml
[lints.clippy]
all = "deny"
dbg_macro = "deny"
expect_used = "deny"
todo = "deny"
unimplemented = "deny"
unwrap_used = "deny"
pedantic = { level = "warn", priority = -1 }
doc_markdown = { level = "allow", priority = 1 }
struct_excessive_bools = { level = "allow", priority = 1 }
unnecessary_wraps = { level = "allow", priority = 1 }
```

## Feature Flags

### 命名约定

- 无前缀库名：`redis`, `nats`, `mongodb`, `qdrant`
- TLS 栈：`tls-rustls`（default）, `tls-native`
- 聚合 feature：`full`, `storage-all`, `rag-all`, `queue-all`

### 规则

- additive only — 不删除已有 feature
- 互斥 feature 需在 `Cargo.toml` 注释说明（仅 `tls-rustls` / `tls-native` 互斥）
- 新增 provider adapter 必须 gate 在独立 feature 后
- gateway feature 格式：`rag`（core, no deps）→ `rag-all`（includes `rag` + all backends）

## 依赖准则

### 新增依赖前必须说明

1. 必要性 — 为什么不能用现有依赖替代
2. 维护状态 — crates.io 活跃度、最近发布时间
3. license — 禁止 GPL / AGPL
4. 体积影响 — 编译时间和二进制大小

### 关键依赖注意事项

- `reqwest` — 禁止 `default-features = true`，精确控制 features
- `tokio` — 禁用 default features，按需启用
- `serde` — 用 `derive` feature
- `secrecy` — 敏感配置值包装，不得 log 或序列化
- `serde_json` — 使用 `preserve_order` + `raw_value`
- `schemars` + `jsonschema` — tool parameters schema 生成与运行时校验
- `uuid` — 使用 `v4` + `v7` + `serde` features，统一用 `Uuid::now_v7()` 作为 ID 生成策略
