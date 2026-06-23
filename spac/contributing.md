# Agent Contributing Guide

> 面向 AI agent 的 commit / PR / 工作流规范。面向人类开发者的指南见仓库根目录 `CONTRIBUTING.md`。

## 工作流

任何非 trivial 任务，动手前必须先输出计划，等待用户批准。

trivial 任务定义：
1. 单文件
2. 不改变 public API
3. 不新增依赖
4. 不改变数据结构
5. 不改变数据库 schema
6. 不新增 `AgentEvent` / `TurnState` / `Tool` trait / `Store` trait variant
7. 不引入新的 trait 实现（`RunStore` / `ArtifactStore` 等）

拿任务后：
1. 先看 `git status --short`
2. 搜索相关符号
3. 阅读相关 SPAC 文档（`spac/architecture.md` 等）
4. 充分理解后再改
5. 通过质量门禁验证：
   ```bash
   cargo fmt --all --check
   cargo clippy --all-targets --all-features --locked -- -D warnings
   cargo test --all-features --locked
   ```
6. 输出变更摘要

## Commit Message

使用带 scope 的 Conventional Commits：

```text
feat(provider): add streaming adapter contract
fix(runtime): handle empty event stream
refactor(store): split persistence adapter
docs: document feature flags
```

## PR 描述模板

```markdown
## Summary

## Changes

## Test
```

## 禁止操作

- 自动 `git add`
- 自动 `git commit`
- 未经确认的 `git reset --hard` / `rm -rf` / `sudo`
- 覆盖用户未提交改动
- 安装系统包
- 修改全局配置（shell / IDE / Git global）
- 下载或执行未经审查的脚本

## 破坏性命令

任何 destructive command 必须先请求确认。允许的命令范围仅限于 cargo 工具链和项目目录内的文件操作。

## 最终回复格式

```markdown
变更摘要：
- ...

验证命令：
```bash
...
```
