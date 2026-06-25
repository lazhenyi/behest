<div align="center">

# behest

**프로덕션 AI Agent 런타임을 위한 Rust 네이티브 빌딩 블록**

<img src="assets/banner.webp" alt="behest — Rust 네이티브 Agent 런타임" width="100%">

[![CI](https://github.com/lazhenyi/behest/actions/workflows/ci.yml/badge.svg)](https://github.com/lazhenyi/behest/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [Français](README.fr.md) · [日本語](README.ja.md) · **한국어** · [Italiano](README.it.md)

</div>

---

## 소개

`behest`는 채팅, 스트리밍, 도구 호출, 임베딩, 런타임 실행, 스토리지, 큐, RAG, 옵저버빌리티, 그리고 선택적 gRPC 서비스를 위한 프로바이더 중립 계약을 제공합니다.

불투명한 "에이전트 프레임워크" 마법 대신, 모델 프로바이더, 도구 실행, 영속성, 운영 경계를 명시적으로 제어해야 하는 시스템을 위해 설계되었습니다.

> 상태: 초기 기반 크레이트. 공개 API는 의도적으로 컴팩트하고, 강력하게 타입이 지정되어 있으며, 문서화되어 있습니다.

## behest라는 이름

**behest** /bɪˈhest/ — *명사* 한 사람의 명령이나 지시.

> At the **behest** of the user, the agent acts.

에이전트 런타임의 핵심은 "자율 의식"이 아닌 제어된 위임입니다: 사용자가 의도를 발행하면, 시스템이 명시적 경계 내에서 컨텍스트를 구성하고, 모델을 호출하고, 도구를 실행하고, 상태를 영속화하고, 이벤트를 발행합니다 — 감사 가능, 복구 가능, 제약 가능, 교체 가능.

`behest`라는 이름은 "brain / cognition / intelligence"와 같은 부풀린 은유를 의도적으로 피합니다. 엔지니어링 사실만 진술합니다:

> tool-calling, streaming, memory, queue, RAG, snapshot — 모든 메커니즘은 누군가 명령을 내렸기 때문에 존재합니다.

## 설계 목표

- **Rust 네이티브 우선**: 타입이 지정된 API, 명시적 오류, 숨겨진 런타임 가정 없음.
- **프로바이더 중립 코어**: OpenAI, Anthropic, 로컬 모델, 프록시 또는 내부 프로바이더가 동일한 계약을 구현 가능.
- **스트리밍 우선 런타임**: 에이전트 루프는 스트림 모델 이벤트를 중심으로 설계, 비스트리밍은 폴백.
- **타입이 지정된 도구 경계**: 도구는 JSON Schema로 설명되고 명시적 레지스트리를 통해 실행.
- **플러그 가능 영속성**: 기본은 메모리, 외부 스토리지는 feature flag로 활성화.
- **운영 서페이스**: 이벤트 발행, 스냅샷, 세션 게이트, 압축, 재시도 정책, 선택적 gRPC 서버.
- **작은 공개 API**: 프레임워크 확장보다 기반 프리미티브 우선.

## 기능 개요

| 영역 | 기능 |
|---|---|
| 프로바이더 계약 | `ChatProvider`, `EmbeddingProvider`, 요청/응답 모델, 스트림 이벤트, 프로바이더 능력 |
| 프로바이더 레지스트리 | 채팅 및 임베딩 프로바이더를 위한 인메모리 라우팅 |
| 채팅 모델 타입 | 메시지, 콘텐츠 파트, 도구 호출, 응답 형식, 토큰 사용량, 종료 이유 |
| 도구 런타임 | `Tool`, `FunctionTool`, `ExternalTool`, `ToolRegistry`, 스키마 생성, 실행 디스패치 |
| 에이전트 런타임 | 컨텍스트 구성, 모델 호출, 도구 루프, 세션 영속화, 이벤트 발사 |
| 런타임 호출 | `RuntimeInvocation`, `EmitRequest`, `EventKind`, `Control`, 전송 중립 emit/on 파사드 |
| 런타임 스트림 | `RuntimeEventStore`, `RuntimeStreamAdapter`, `RuntimeSubscriptionHub`, 리플레이 + 라이브 팬아웃 |
| 런타임 보안 | 세션 게이트, 런타임 정책, 입력 허가, 둠 루프 감지, 도구 출력 절단 |
| 스토리지 | 메모리 스토리지, Redis, SQLx, MongoDB, SurrealDB, 객체 스토리지, Qdrant 임베딩 |
| 컨텍스트와 RAG | 컨텍스트 어댑터, 정적/함수 어댑터, 선택적 RAG 어댑터 |
| 큐 | NATS 또는 Redis Streams를 통한 선택적 이벤트 발행 |
| 구성 | 빌더, 파일 기반 구성, 환경 변수 로딩, 시크릿 간접 참조 |
| 서버 | `server` feature의 선택적 gRPC 서버 바이너리 |
| 옵저버빌리티 | tracing과 선택적 OpenTelemetry 통합 |

## 빠른 시작

```toml
[dependencies]
behest = "0.2"
```

프로바이더 중립 채팅 요청 생성:

```rust
use behest::prelude::*;

let request = ChatRequest::new(ModelName::new("example-model"))
    .with_message(Message::system_text("You are concise."))
    .with_user_text("Summarize this project in one sentence.");
```

레지스트리에 프로바이더를 등록하고 요청 라우팅:

```rust
use behest::prelude::*;

let registry = ProviderRegistry::new();
let provider_id = ProviderId::new("my-provider");

// 먼저 ChatProvider 구현을 등록.
// registry.register_chat(my_provider);

// 그런 다음 중립 레지스트리를 통해 라우팅.
// let response = registry.complete(&provider_id, request).await?;
```

더 많은 예제는 [`examples/`](examples/)에서 확인.

## 커스텀 프로바이더 구현

`behest`는 특정 벤더 SDK를 코어에 강제하지 않습니다. 모든 모델 백엔드, 게이트웨이, 로컬 추론 서비스 또는 내부 프로바이더에 `ChatProvider`를 구현하세요.

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

스트리밍 프로바이더는 `stream`을 오버라이드할 수 있습니다.

## 도구 정의 및 실행

도구는 명시적인 런타임 객체입니다. 각 도구는 안정적인 이름, 사람이 읽을 수 있는 설명, JSON Schema 인수 계약을 노출합니다.

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

프로바이더가 반환한 도구 호출은 레지스트리를 통해 실행 가능:

```rust
use behest::prelude::*;
use serde_json::json;

let call = ToolCall::new("call_1", "echo", json!({ "message": "hello" }));
let output = registry.execute(&call).await?;
```

## 런타임 모델

런타임 계층에서 `AgentRuntime`는 전체 에이전트 루프를 오케스트레이션:

```text
RunRequest
  -> 세션 로드 또는 생성
  -> 입력 허가
  -> 컨텍스트 구성
  -> 모델 프로바이더 호출
  -> 어시스턴트 출력 스트림/영속화
  -> 도구 호출 실행
  -> 도구 결과 추가
  -> 완료, 제한 또는 오류까지 반복
  -> AgentEvent 발행
```

런타임은 다음을 통합:

- `ProviderRegistry`
- `ContextPipeline`
- `ToolRuntime`
- `RuntimeStore`
- `RuntimePolicy`
- `CompactionService`
- `SessionGate`
- 선택적 이벤트 발행자
- 선택적 스냅샷 스토어
- 선택적 백그라운드 작업 풀

## 구성

`AgentConfig`는 계층 구성을 지원:

1. 기본값
2. 파일 소스
3. 환경 변수
4. 수동 빌더 세터

```rust
use behest::prelude::*;

let config = AgentConfig::builder()
    .with_file("behest.toml")?
    .with_env("BEHEST")?
    .build()?;

let runtime = config.into_runtime().await?;
```

시크릿은 `env:VAR_NAME` 간접 참조를 통해 로드 가능:

```toml
[providers.openai]
api_key = "env:OPENAI_API_KEY"
```

전체 구조는 [`behest.toml` 예제](examples/hello_config.rs)를 참조.

## 프로바이더 어댑터

구체적인 프로바이더 어댑터는 feature gate로 활성화.

| Feature | 어댑터 | Chat | Stream | Embeddings | Tools |
|---|---|---:|---:|---:|---:|
| `openai` | `OpenAiChatAdapter`, `OpenAiEmbeddingAdapter` | 예 | 예 | 예 | 예 |
| `anthropic` | `AnthropicChatAdapter` | 예 | 예 | 아니오 | 예 |

어댑터 활성화:

```toml
[dependencies]
behest = { version = "0.2", features = ["openai", "anthropic"] }
```

## Feature Flags

<details>
<summary>전체 feature 목록 펼치기</summary>

**기본값:**

| Feature | 설명 |
|---|---|
| `tls-rustls` | rustls를 사용하는 기본 TLS 스택 |

**프로바이더 어댑터:**

| Feature | 설명 |
|---|---|
| `openai` | OpenAI 호환 채팅 및 임베딩 어댑터 |
| `anthropic` | Anthropic 호환 채팅 어댑터 |

**TLS:**

| Feature | 설명 |
|---|---|
| `tls-rustls` | HTTP / 활성화된 백엔드에 대한 rustls TLS 통합 활성화 |
| `tls-native` | HTTP / 활성화된 백엔드에 대한 네이티브 TLS 통합 활성화 |

**스토리지:**

| Feature | 설명 |
|---|---|
| `redis` | Redis 기반 스토리지 지원 및 Redis Streams 프리미티브 |
| `redis-cluster` | Redis Cluster 지원; `redis` 포함 |
| `sqlx-postgres` | SQLx PostgreSQL 스토리지 지원 |
| `sqlx-mysql` | SQLx MySQL 스토리지 지원 |
| `sqlx-sqlite` | SQLx SQLite 스토리지 지원 |
| `mongodb` | MongoDB 세션 스토리지 지원 |
| `surrealdb` | SurrealDB 세션 스토리지 지원 |
| `object_store` | AWS S3 포함 객체 스토리지 지원 |
| `storage-all` | Redis, PostgreSQL, MySQL, SQLite, MongoDB, SurrealDB 스토리지 feature |

**RAG:**

| Feature | 설명 |
|---|---|
| `rag` | 코어 RAG 컨텍스트 어댑터 |
| `qdrant` | Qdrant 임베딩 스토리지 백엔드 |
| `tantivy` | Tantivy 백엔드 지원 |
| `rag-all` | `rag`, `qdrant`, `tantivy` 활성화 |

**큐:**

| Feature | 설명 |
|---|---|
| `queue` | 코어 이벤트 발행자 트레이트 |
| `nats` | NATS 이벤트 발행자 |
| `queue-all` | `queue`, `nats`, `redis` 활성화 |

**서버 및 옵저버빌리티:**

| Feature | 설명 |
|---|---|
| `server` | gRPC 서버 바이너리 및 protobuf 서비스 계층 |
| `otel` | OpenTelemetry tracing 통합 |

**편의 프로필:**

| Feature | 설명 |
|---|---|
| `full` | 완전한 런타임 프로필: OpenAI, Anthropic, Redis, Redis Cluster, NATS, PostgreSQL, MongoDB, SurrealDB, OpenTelemetry, 모든 RAG 백엔드, 모든 큐 백엔드, 객체 스토리지. `server`, `sqlx-mysql`, `sqlx-sqlite`는 의도적으로 활성화하지 않음. |

</details>

선택된 feature 예제:

```toml
[dependencies]
behest = {
    version = "0.2",
    default-features = false,
    features = ["tls-rustls", "openai", "anthropic", "redis", "queue", "nats"]
}
```

## 오류 모델

`behest`는 문자열화된 프레임워크 실패 대신 타입이 지정된 오류 카테고리를 노출:

- `ProviderError`
- `ToolError`
- `StorageError`
- `ContextError`
- `RuntimeError`
- 최상위 `Error`
- 크레이트 수준 `Result<T>`

프로바이더 오류는 지원되지 않는 능력, 재시도 가능한 실패, 전송 실패, 잘못된 응답, 어댑터 특정 오류를 구분.

도구 오류는 누락된 도구, 잘못된 인수, 실행 실패, 타임아웃, 구현되지 않은 외부 도구를 구분.

## Lint 정책

크레이트는 의도적으로 엄격:

- `unsafe_code = "forbid"`
- `missing_docs = "deny"`
- `unreachable_pub = "deny"`
- `clippy::all = "deny"`
- `dbg_macro = "deny"`
- `expect_used = "deny"`
- `todo = "deny"`
- `unimplemented = "deny"`
- `unwrap_used = "deny"`

이 프로젝트는 공개 API의 명확성과 실패 경로의 위생을 런타임 계약의 일부로 취급합니다.

## 개발

```bash
# 포맷
cargo fmt --all --check

# 모든 타겟과 feature 확인
cargo check --all-targets --all-features --locked

# Lint
cargo clippy --all-targets --all-features --locked -- -D warnings

# 테스트
cargo test --all-features --locked

# 문서 빌드
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

전체 로컬 검증 세트 실행:

```bash
cargo fmt --all --check && \
cargo check --all-targets --all-features --locked && \
cargo clippy --all-targets --all-features --locked -- -D warnings && \
cargo test --all-features --locked && \
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

## 라이선스

다음 중 하나:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

선택하세요.
