<div align="center">

# behest

**Building blocks Rust nativi per runtime di agenti AI in produzione**

<img src="assets/banner.webp" alt="behest — Runtime agenti Rust nativo" width="100%">

[![CI](https://github.com/lazhenyi/behest/actions/workflows/ci.yml/badge.svg)](https://github.com/lazhenyi/behest/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · **Italiano**

</div>

---

## Panoramica

`behest` fornisce contratti neutri rispetto al provider per chat, streaming, chiamate di strumenti, embedding, esecuzione runtime, code, RAG, osservabilità e servizio gRPC opzionale.

È progettato per sistemi che necessitano controllo esplicito su provider di modelli, esecuzione di strumenti, persistenza e confini operativi — invece della magia opaca dei "framework per agenti".

> Stato: crate fondazione early-stage. Le API pubbliche sono intenzionalmente compatte, fortemente tipizzate e documentate.

## Perché behest

**behest** /bɪˈhest/ — *s.* gli ordini o i comandi di una persona.

> At the **behest** of the user, the agent acts.

Il cuore di un runtime per agenti non è la "coscienza autonoma" ma la delega controllata: l'utente emette un'intenzione e il sistema compone il contesto, invoca modelli, esegue strumenti, persiste lo stato, pubblica eventi entro confini espliciti — audibile, recuperabile, vincolabile e sostituibile.

Il nome `behest` evita deliberatamente metafore gonfiate come "brain / cognition / intelligence". Enuncia solo un fatto ingegneristico:

> tool-calling, streaming, memory, queue, RAG, snapshot — tutti i meccanismi esistono perché qualcuno ha dato un ordine.

## Obiettivi di design

- **Rust nativo prima di tutto**: API tipizzate, errori espliciti, nessuna ipotesi runtime nascosta.
- **Core neutro rispetto al provider**: OpenAI, Anthropic, modelli locali, proxy o provider interni possono implementare gli stessi contratti.
- **Runtime orientato al streaming**: il loop dell'agente è progettato attorno a eventi modello streammati, con fallback non-streaming.
- **Confine strumenti tipizzato**: gli strumenti sono descritti da JSON Schema ed eseguiti attraverso registri espliciti.
- **Persistenza pluggabile**: memoria per default, storage esterni tramite feature flag.
- **Superficie operativa**: pubblicazione eventi, snapshot, gate di sessione, compattazione, politica di retry, server gRPC opzionale.
- **Piccola API pubblica**: primitivi fondamentali rispetto alla proliferazione di framework.

## Funzionalità

| Area | Capacità |
|---|---|
| Contratti provider | `ChatProvider`, `EmbeddingProvider`, modelli richiesta/risposta, eventi stream, capacità provider |
| Registro provider | Routing in memoria per provider chat ed embedding |
| Tipi modello chat | messaggi, parti di contenuto, chiamate strumenti, formati risposta, utilizzo token, ragioni di fine |
| Runtime strumenti | `Tool`, `FunctionTool`, `ExternalTool`, `ToolRegistry`, generazione schema, dispatch esecuzione |
| Runtime agente | costruzione contesto, chiamate modello, loop strumenti, persistenza sessione, emissione eventi |
| Invocazione runtime | `RuntimeInvocation`, `EmitRequest`, `EventKind`, `Control`, facade emit/on neutra al trasporto |
| Stream runtime | `RuntimeEventStore`, `RuntimeStreamAdapter`, `RuntimeSubscriptionHub`, replay + fanout dal vivo |
| Sicurezza runtime | gate sessione, politica runtime, ammissione input, rilevamento loop infinito, troncamento output strumenti |
| Storage | storage memoria, Redis, SQLx, MongoDB, SurrealDB, storage oggetti, embedding Qdrant |
| Contesto e RAG | adattatori contesto, adattatori statici/funzionale, adattatore RAG opzionale |
| Code | pubblicazione eventi opzionale tramite NATS o Redis Streams |
| Configurazione | builder, configurazione basata su file, caricamento variabili ambiente, indirezione secret |
| Server | binario server gRPC opzionale tramite feature `server` |
| Osservabilità | tracing e integrazione OpenTelemetry opzionale |

## Inizio rapido

```toml
[dependencies]
behest = "0.2"
```

Creare una richiesta chat neutra rispetto al provider:

```rust
use behest::prelude::*;

let request = ChatRequest::new(ModelName::new("example-model"))
    .with_message(Message::system_text("You are concise."))
    .with_user_text("Summarize this project in one sentence.");
```

Registrare provider in un registro e instradare le richieste:

```rust
use behest::prelude::*;

let registry = ProviderRegistry::new();
let provider_id = ProviderId::new("my-provider");

// Prima registrare un'implementazione ChatProvider.
// registry.register_chat(my_provider);

// Poi instradare attraverso il registro neutro.
// let response = registry.complete(&provider_id, request).await?;
```

Più esempi in [`examples/`](examples/).

## Implementare un provider personalizzato

`behest` non forza un SDK vendor nel core. Implementare `ChatProvider` per qualsiasi backend modello, gateway, servizio di inferenza locale o provider interno.

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

I provider streaming possono sovrascrivere `stream`.

## Definire ed eseguire strumenti

Gli strumenti sono oggetti runtime espliciti. Ogni strumento espone un nome stabile, una descrizione leggibile e un contratto argomenti JSON Schema.

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

Le chiamate strumento restituite da un provider possono essere eseguite attraverso il registro:

```rust
use behest::prelude::*;
use serde_json::json;

let call = ToolCall::new("call_1", "echo", json!({ "message": "hello" }));
let output = registry.execute(&call).await?;
```

## Modello runtime

Al livello runtime, `AgentRuntime` orchestra il loop completo dell'agente:

```text
RunRequest
  -> caricare o creare sessione
  -> ammettere input
  -> costruire contesto
  -> chiamare provider modello
  -> streamare / persistere output assistente
  -> eseguire chiamate strumento
  -> aggiungere risultati strumento
  -> ripetere fino a completamento, limite o errore
  -> emettere AgentEvent
```

Il runtime unisce:

- `ProviderRegistry`
- `ContextPipeline`
- `ToolRuntime`
- `RuntimeStore`
- `RuntimePolicy`
- `CompactionService`
- `SessionGate`
- editore eventi opzionale
- storage snapshot opzionale
- pool job background opzionale

## Configurazione

`AgentConfig` supporta configurazione a livelli:

1. valori default
2. sorgenti file
3. variabili ambiente
4. setter manuali del builder

```rust
use behest::prelude::*;

let config = AgentConfig::builder()
    .with_file("behest.toml")?
    .with_env("BEHEST")?
    .build()?;

let runtime = config.into_runtime().await?;
```

I secret possono essere caricati tramite indirezione `env:VAR_NAME`:

```toml
[providers.openai]
api_key = "env:OPENAI_API_KEY"
```

Struttura completa nell'[esempio `behest.toml`](examples/hello_config.rs).

## Adattatori provider

Gli adattatori provider concreti sono attivati tramite feature gate.

| Feature | Adattatore | Chat | Stream | Embeddings | Tools |
|---|---|---:|---:|---:|---:|
| `openai` | `OpenAiChatAdapter`, `OpenAiEmbeddingAdapter` | sì | sì | sì | sì |
| `anthropic` | `AnthropicChatAdapter` | sì | sì | no | sì |

Attivare adattatori:

```toml
[dependencies]
behest = { version = "0.2", features = ["openai", "anthropic"] }
```

## Feature Flags

<details>
<summary>Clicca per espandere la lista completa delle feature</summary>

**Default:**

| Feature | Descrizione |
|---|---|
| `tls-rustls` | Stack TLS default che usa rustls |

**Adattatori provider:**

| Feature | Descrizione |
|---|---|
| `openai` | Adattatori chat ed embedding compatibili OpenAI |
| `anthropic` | Adattatore chat compatibile Anthropic |

**TLS:**

| Feature | Descrizione |
|---|---|
| `tls-rustls` | Abilita integrazione TLS rustls per HTTP / backend abilitati |
| `tls-native` | Abilita integrazione TLS nativa per HTTP / backend abilitati |

**Storage:**

| Feature | Descrizione |
|---|---|
| `redis` | Supporto storage Redis e primitivi Redis Streams |
| `redis-cluster` | Supporto Redis Cluster; implica `redis` |
| `sqlx-postgres` | Supporto storage SQLx PostgreSQL |
| `sqlx-mysql` | Supporto storage SQLx MySQL |
| `sqlx-sqlite` | Supporto storage SQLx SQLite |
| `mongodb` | Supporto storage sessione MongoDB |
| `surrealdb` | Supporto storage sessione SurrealDB |
| `object_store` | Supporto storage oggetti, incluso AWS S3 |
| `storage-all` | Feature storage Redis, PostgreSQL, MySQL, SQLite, MongoDB e SurrealDB |

**RAG:**

| Feature | Descrizione |
|---|---|
| `rag` | Adattatore contesto RAG core |
| `qdrant` | Backend storage embedding Qdrant |
| `tantivy` | Supporto backend Tantivy |
| `rag-all` | Abilita `rag`, `qdrant` e `tantivy` |

**Code:**

| Feature | Descrizione |
|---|---|
| `queue` | Trait editore eventi core |
| `nats` | Editore eventi NATS |
| `queue-all` | Abilita `queue`, `nats` e `redis` |

**Server e osservabilità:**

| Feature | Descrizione |
|---|---|
| `server` | Binario server gRPC e livello servizio protobuf |
| `otel` | Integrazione tracing OpenTelemetry |

**Profilo di comodità:**

| Feature | Descrizione |
|---|---|
| `full` | Profilo runtime completo pronto all'uso: OpenAI, Anthropic, Redis, Redis Cluster, NATS, PostgreSQL, MongoDB, SurrealDB, OpenTelemetry, tutti i backend RAG, tutte le code e storage oggetti. Non abilita `server`, `sqlx-mysql` o `sqlx-sqlite` intenzionalmente. |

</details>

Esempio con feature selezionate:

```toml
[dependencies]
behest = {
    version = "0.2",
    default-features = false,
    features = ["tls-rustls", "openai", "anthropic", "redis", "queue", "nats"]
}
```

## Modello errori

`behest` espone categorie di errori tipizzate invece di fallimenti framework in stringhe:

- `ProviderError`
- `ToolError`
- `StorageError`
- `ContextError`
- `RuntimeError`
- `Error` di alto livello
- `Result<T>` a livello crate

Gli errori provider distinguono capacità non supportate, fallimenti ritentabili, fallimenti trasporto, risposte invalide e errori specifici adattatore.

Gli errori strumento distinguono strumenti mancanti, argomenti invalidi, fallimenti esecuzione, timeout e strumenti esterni non implementati.

## Politica lint

Il crate è intenzionalmente rigoroso:

- `unsafe_code = "forbid"`
- `missing_docs = "deny"`
- `unreachable_pub = "deny"`
- `clippy::all = "deny"`
- `dbg_macro = "deny"`
- `expect_used = "deny"`
- `todo = "deny"`
- `unimplemented = "deny"`
- `unwrap_used = "deny"`

Questo progetto tratta la chiarezza dell'API pubblica e l'igiene dei percorsi di errore come parte del contratto runtime.

## Sviluppo

```bash
# Formattazione
cargo fmt --all --check

# Controllare tutti i target e feature
cargo check --all-targets --all-features --locked

# Lint
cargo clippy --all-targets --all-features --locked -- -D warnings

# Test
cargo test --all-features --locked

# Costruire documentazione
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

Eseguire l'insieme completo di verifica locale:

```bash
cargo fmt --all --check && \
cargo check --all-targets --all-features --locked && \
cargo clippy --all-targets --all-features --locked -- -D warnings && \
cargo test --all-features --locked && \
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

## Licenza

Sotto una delle seguenti:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

a scelta.
