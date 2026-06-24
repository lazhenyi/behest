# Proto Versioning Policy

behest gRPC API 使用 Protocol Buffers 定义。proto 文件位于 `src/grpc/proto/agent/v1/`。

## 包名与版本

当前包名: `agent.v1`。

版本通过包名路径表达。breaking change 时升级到 `agent.v2`,旧版本不再维护。

## Breaking Change 策略

behest 是 library crate, proto 是可选 feature (`server`)。breaking change 随 crate major version 发布。

规则:

1. 删除或重命名 field/message/enum/service/rpc → breaking。
2. 修改 field number → breaking。
3. 修改 field type → breaking。
4. 添加 optional field → non-breaking (向前兼容)。
5. 添加新 message/enum/service/rpc → non-breaking。

## 兼容性保证

- minor version 内不引入 breaking proto change。
- breaking proto change 随 major version 发布。
- 不提供双版本并行支持。

## Lint 与 Breaking Detection

项目提供 `buf.yaml` 和 `buf.gen.yaml` 配置。需要安装 [buf](https://buf.build/docs/installation) 后运行:

```bash
buf lint src/grpc/proto
buf breaking src/grpc/proto --against .git#branch=main
```

CI 中暂未集成 buf (需要网络访问 buf registry)。本地开发时建议手动运行。

## 客户端生成

`tonic-build` 配置为 `build_client(false)`,只生成 server 端代码。客户端通过 proto 文件自行生成:

```bash
buf generate src/grpc/proto
# 或
protoc --rust_out=. --tonic_out=. src/grpc/proto/agent/v1/*.proto
```

proto 文件本身是 API 交付物。
