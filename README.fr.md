<div align="center">

# behest

**Briques Rust natives pour les runtimes d'agents AI en production**

<img src="assets/banner.webp" alt="behest — Runtime d'agents Rust natif" width="100%">

[![CI](https://github.com/lazhenyi/behest/actions/workflows/ci.yml/badge.svg)](https://github.com/lazhenyi/behest/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · **Français** · [日本語](README.ja.md) · [한국어](README.ko.md) · [Italiano](README.it.md)

</div>

---

## Présentation

`behest` fournit des contrats neutres pour le chat, le streaming, l'appel d'outils, les embeddings, l'exécution runtime, le stockage, les files d'attente, RAG, l'observabilité, et optionnellement le service gRPC.

Il est conçu pour les systèmes nécessitant un contrôle explicite sur les fournisseurs de modèles, l'exécution d'outils, la persistance et les limites opérationnelles — au lieu de la magie opaque des « cadres d'agents ».

> Statut : crate fondation early-stage. Les API publiques sont intentionnellement compactes, fortement typées et documentées.

## Pourquoi behest

**behest** /bɪˈhest/ — *n.* les ordres ou commandements d'une personne.

> At the **behest** of the user, the agent acts.

Le cœur d'un runtime d'agent n'est pas la « conscience autonome » mais la délégation contrôlée : l'utilisateur émet une intention, et le système compose le contexte, invoque les modèles, exécute les outils, persiste l'état, publie des événements dans des limites explicites — auditable, récupérable, contraint et remplaçable.

Le nom `behest` évite délibérément les métaphores gonflées comme « brain / cognition / intelligence ». Il ne fait qu'énoncer un fait d'ingénierie :

> tool-calling, streaming, memory, queue, RAG, snapshot — tous les mécanismes existent parce que quelqu'un a passé un ordre.

## Objectifs de conception

- **Rust natif d'abord** : API typées, erreurs explicites, pas d'hypothèses runtime cachées.
- **Cœur neutre en fournisseurs** : OpenAI, Anthropic, modèles locaux, proxys ou fournisseurs internes peuvent implémenter les mêmes contrats.
- **Runtime orienté streaming** : la boucle d'agent est conçue autour des événements modèle en streaming, avec repli non-streaming.
- **Limites d'outils typées** : les outils sont décrits par JSON Schema et exécutés via des registres explicites.
- **Persistance modulable** : mémoire par défaut, stockages externes via feature flags.
- **Surface opérationnelle** : publication d'événements, snapshots, portes de session, compression, politique de retry, serveur gRPC optionnel.
- **Petite API publique** : primitives fondamentales plutôt que prolifération de frameworks.

## Fonctionnalités

| Domaine | Capacité |
|---|---|
| Contrats fournisseur | `ChatProvider`, `EmbeddingProvider`, modèles requête/réponse, événements stream, capacités fournisseur |
| Registre fournisseur | Routage mémoire pour les fournisseurs chat et embedding |
| Types modèle chat | messages, parties de contenu, appels d'outils, formats de réponse, utilisation de tokens, raisons de fin |
| Runtime d'outils | `Tool`, `FunctionTool`, `ExternalTool`, `ToolRegistry`, génération de schema, dispatch d'exécution |
| Runtime d'agent | construction de contexte, appels modèle, boucle d'outils, persistance de session, émission d'événements |
| Invocation runtime | `RuntimeInvocation`, `EmitRequest`, `EventKind`, `Control`, facade emit/on neutre au transport |
| Stream runtime | `RuntimeEventStore`, `RuntimeStreamAdapter`, `RuntimeSubscriptionHub`, replay + diffusion en direct |
| Graphe de raisonnement | `ReasoningGraph`, `ReasoningOperator`, `ControlKind`, stratégies de raisonnement basées sur DAG |
| Sécurité runtime | porte de session, politique runtime, admission d'entrée, détection de boucle morte, troncature de sortie d'outils |
| Stockage | stockage mémoire, Redis, SQLx, MongoDB, SurrealDB, stockage objet, embeddings Qdrant |
| Contexte et RAG | adaptateurs de contexte, adaptateurs statiques/fonction, adaptateur RAG optionnel |
| Files d'attente | publication d'événements optionnelle via NATS ou Redis Streams |
| Configuration | constructeur, configuration basée sur fichiers, chargement de variables d'environnement, indirection de secrets |
| Serveur | binaire serveur gRPC optionnel via la feature `server` |
| Observabilité | tracing et intégration OpenTelemetry optionnelle |

## Démarrage rapide

```toml
[dependencies]
behest = "0.2"
```

Créer une requête chat neutre en fournisseur :

```rust
use behest::prelude::*;

let request = ChatRequest::new(ModelName::new("example-model"))
    .with_message(Message::system_text("You are concise."))
    .with_user_text("Summarize this project in one sentence.");
```

Enregistrer des fournisseurs dans un registre et router les requêtes :

```rust
use behest::prelude::*;

let registry = ProviderRegistry::new();
let provider_id = ProviderId::new("my-provider");

// Enregistrer d'abord une implémentation ChatProvider.
// registry.register_chat(my_provider);

// Puis router via le registre neutre.
// let response = registry.complete(&provider_id, request).await?;
```

Plus d'exemples dans [`examples/`](examples/).

## Implémenter un fournisseur personnalisé

`behest` ne force pas un SDK vendeur dans le cœur. Implémentez `ChatProvider` pour n'importe quel backend modèle, passerelle, service d'inférence local ou fournisseur interne.

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

Les fournisseurs streaming peuvent surcharger `stream`.

## Définir et exécuter des outils

Les outils sont des objets runtime explicites. Chaque outil expose un nom stable, une description lisible et un contrat d'arguments JSON Schema.

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

Les appels d'outils retournés par un fournisseur peuvent être exécutés via le registre :

```rust
use behest::prelude::*;
use serde_json::json;

let call = ToolCall::new("call_1", "echo", json!({ "message": "hello" }));
let output = registry.execute(&call).await?;
```

## Modèle runtime

Au niveau runtime, `AgentRuntime` orchestre la boucle d'agent complète :

```text
RunRequest
  -> charger ou créer une session
  -> admettre l'entrée
  -> construire le contexte
  -> appeler le fournisseur modèle
  -> streamer / persister la sortie assistant
  -> exécuter les appels d'outils
  -> ajouter les résultats d'outils
  -> répéter jusqu'à complétion, limite ou erreur
  -> émettre des AgentEvent
```

Le runtime rassemble :

- `ProviderRegistry`
- `ContextPipeline`
- `ToolRuntime`
- `RuntimeStore`
- `RuntimePolicy`
- `CompactionService`
- `SessionGate`
- éditeur d'événements optionnel
- stockage snapshot optionnel
- pool de tâches arrière-plan optionnel

## Configuration

`AgentConfig` supporte la configuration en couches :

1. valeurs par défaut
2. sources fichier
3. variables d'environnement
4. setters manuels du constructeur

```rust
use behest::prelude::*;

let config = AgentConfig::builder()
    .with_file("behest.toml")?
    .with_env("BEHEST")?
    .build()?;

let runtime = config.into_runtime().await?;
```

Les secrets peuvent être chargés via l'indirection `env:VAR_NAME` :

```toml
[providers.openai]
api_key = "env:OPENAI_API_KEY"
```

Structure complète de configuration dans l'[exemple `behest.toml`](examples/hello_config.rs).

## Adaptateurs fournisseur

Les adaptateurs fournisseur concrets sont activés par feature gate.

| Feature | Adaptateur | Chat | Stream | Embeddings | Tools |
|---|---|---:|---:|---:|---:|
| `openai` | `OpenAiChatAdapter`, `OpenAiEmbeddingAdapter` | oui | oui | oui | oui |
| `anthropic` | `AnthropicChatAdapter` | oui | oui | non | oui |

Activer les adaptateurs :

```toml
[dependencies]
behest = { version = "0.2", features = ["openai", "anthropic"] }
```

## Feature Flags

<details>
<summary>Cliquez pour développer la liste complète des features</summary>

**Par défaut :**

| Feature | Description |
|---|---|
| `tls-rustls` | Pile TLS par défaut utilisant rustls |

**Adaptateurs fournisseur :**

| Feature | Description |
|---|---|
| `openai` | Adaptateurs chat et embedding compatibles OpenAI |
| `anthropic` | Adaptateur chat compatible Anthropic |

**TLS :**

| Feature | Description |
|---|---|
| `tls-rustls` | Active l'intégration TLS rustls pour HTTP / backends activés |
| `tls-native` | Active l'intégration TLS native pour HTTP / backends activés |

**Stockage :**

| Feature | Description |
|---|---|
| `redis` | Support stockage Redis et primitives Redis Streams |
| `redis-cluster` | Support Redis Cluster ; implique `redis` |
| `sqlx-postgres` | Support stockage SQLx PostgreSQL |
| `sqlx-mysql` | Support stockage SQLx MySQL |
| `sqlx-sqlite` | Support stockage SQLx SQLite |
| `mongodb` | Support stockage session MongoDB |
| `surrealdb` | Support stockage session SurrealDB |
| `object_store` | Support stockage objet, incluant AWS S3 |
| `storage-all` | Features stockage Redis, PostgreSQL, MySQL, SQLite, MongoDB et SurrealDB |

**RAG :**

| Feature | Description |
|---|---|
| `rag` | Adaptateur contexte RAG core |
| `qdrant` | Backend stockage embedding Qdrant |
| `tantivy` | Support backend Tantivy |
| `rag-all` | Active `rag`, `qdrant` et `tantivy` |

**Files d'attente :**

| Feature | Description |
|---|---|
| `queue` | Traits éditeur d'événements core |
| `nats` | Éditeur d'événements NATS |
| `queue-all` | Active `queue`, `nats` et `redis` |

**Serveur et observabilité :**

| Feature | Description |
|---|---|
| `server` | Binaire serveur gRPC et couche service protobuf |
| `otel` | Intégration tracing OpenTelemetry |

**Profil de commodité :**

| Feature | Description |
|---|---|
| `full` | Profil runtime complet prêt à l'emploi : OpenAI, Anthropic, Redis, Redis Cluster, NATS, PostgreSQL, MongoDB, SurrealDB, OpenTelemetry, tous les backends RAG, toutes les files d'attente et stockage objet. N'active pas `server`, `sqlx-mysql` ou `sqlx-sqlite` intentionnellement. |

</details>

Exemple avec features sélectionnées :

```toml
[dependencies]
behest = {
    version = "0.2",
    default-features = false,
    features = ["tls-rustls", "openai", "anthropic", "redis", "queue", "nats"]
}
```

## Modèle d'erreur

`behest` expose des catégories d'erreurs typées au lieu d'échecs framework en chaînes de caractères :

- `ProviderError`
- `ToolError`
- `StorageError`
- `ContextError`
- `RuntimeError`
- `Error` de haut niveau
- `Result<T>` au niveau crate

Les erreurs provider distinguent capacités non supportées, échecs réessayables, échecs transport, réponses invalides et erreurs spécifiques d'adaptateur.

Les erreurs d'outils distinguent outils manquants, arguments invalides, échecs d'exécution, timeouts et outils externes non implémentés.

## Politique de lint

Le crate est intentionnellement strict :

- `unsafe_code = "forbid"`
- `missing_docs = "deny"`
- `unreachable_pub = "deny"`
- `clippy::all = "deny"`
- `dbg_macro = "deny"`
- `expect_used = "deny"`
- `todo = "deny"`
- `unimplemented = "deny"`
- `unwrap_used = "deny"`

Ce projet traite la clarté de l'API publique et l'hygiène des chemins d'erreur comme faisant partie du contrat runtime.

## Développement

```bash
# Format
cargo fmt --all --check

# Vérifier toutes les cibles et features
cargo check --all-targets --all-features --locked

# Lint
cargo clippy --all-targets --all-features --locked -- -D warnings

# Tests
cargo test --all-features --locked

# Construire la documentation
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

Exécuter l'ensemble complet de vérification locale :

```bash
cargo fmt --all --check && \
cargo check --all-targets --all-features --locked && \
cargo clippy --all-targets --all-features --locked -- -D warnings && \
cargo test --all-features --locked && \
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --locked
```

## Licence

Sous l'une des licences suivantes :

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

à votre choix.
