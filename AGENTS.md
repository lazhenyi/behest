# AGENTS.md — agents crate

`agents` — Rust-native cloud agent runtime library (edition 2024, MSRV 1.88).  
License: MIT OR Apache-2.0.

## 不可违反约束

1. 禁止 `unsafe`
2. 禁止 `unwrap()` / `expect()`（测试例外）
3. 禁止 `todo!()` / `unimplemented!()` / `dbg!()`
4. 禁止 `anyhow` 出现在 public API
5. 禁止 GPL / AGPL 依赖
6. 禁止自动 `git add` / `git commit`
7. 禁止未确认的 destructive command
8. 禁止覆盖用户未提交改动
9. 禁止 `TODO` / `FIXME` / 临时代码

## 质量门禁

```bash
cargo fmt --all --check
cargo check --all-targets --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-features --locked
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

## 详细规范

按需读取，按主题拆分：

| 规范文件 | 内容 |
|---------|------|
| [spac/architecture.md](spac/architecture.md) | 架构概览、模块边界、runtime 内核设计（FSM / session gate / snapshot / compaction / doom loop） |
| [spac/conventions.md](spac/conventions.md) | 代码约定（类型系统、错误处理、异步、格式、禁止项）、feature flags 策略、依赖准则 |
| [spac/testing.md](spac/testing.md) | 测试策略、测试工具、测试放置、设计原则 |
| [spac/extending.md](spac/extending.md) | provider adapter 开发步骤、runtime 扩展（AgentEvent / TurnState / Store backend / Context adapter） |
| [spac/contributing.md](spac/contributing.md) | agent 视角的 commit / PR / 工作流规范 |

## 工作流

非 trivial 任务动手前先输出计划并等待批准。

trivial = 单文件 + 不改 public API + 不改数据结构 + 不新增依赖。
