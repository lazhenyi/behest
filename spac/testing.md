# Testing Strategy

## 必须写测试的场景

- 核心业务逻辑：`AgentRuntime` 状态机、turn transition、compaction
- 错误路径：provider 异常、token 超限、session busy、doom loop 触发
- 序列化 / 反序列化：message、tool spec、event、config
- tool 注册与执行：`FunctionTool` 注册、参数校验、执行结果
- session / run 持久化读写语义
- context pipeline 组合：system prompt + history + compaction filter + token trim

## 测试放置

- **unit test**：模块内 `#[cfg(test)] mod tests { use super::*; }`
- **integration test**：`tests/` 目录
- **doc test**：`lib.rs` 和关键 public type 的文档示例

## 测试工具

| 工具 | 用途 | 位置 |
|------|------|------|
| `tokio::test` | 异步测试 | dev-dependency |
| `tokio-test` | 异步测试工具 | dev-dependency |
| `tempfile` | 临时目录（snapshot store 测试） | dev-dependency |
| `pretty_assertions` | 可读 diff | dev-dependency |
| `insta` | snapshot testing（JSON） | dev-dependency |
| `proptest` | property-based testing | dev-dependency |
| `wiremock` | HTTP mock（provider adapter 测试） | dev-dependency |
| `tracing-test` | tracing 断言 | dev-dependency |

## 测试中使用 unwrap

允许在 `#[cfg(test)]` 模块中使用 `unwrap()` 和 `expect()`：

```rust
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    // ...
}
```

## 测试运行

```bash
# 全量
cargo test --all-features --locked

# 指定模块
cargo test --lib runtime::agent::tests

# 使用 nextest（可选）
cargo nextest run --all-features
```

## 测试设计原则

- 每个 test 独立，不依赖执行顺序
- mocking 优先使用 trait object 而非 macro-based mock
- provider mock 实现 `ChatProvider` trait，不 mock HTTP 层（通用 adapter 测试除外）
- snapshot test 用 `insta` + `cargo insta review`
