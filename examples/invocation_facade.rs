//! Demonstrates a high-level facade over [`RuntimeInvocation`]'s `emit` / `on` API.
//!
//! The [`AgentFacade`] wraps the raw primitives into convenient methods:
//!
//! | Method       | Wraps                 | Use case                          |
//! |--------------|-----------------------|-----------------------------------|
//! | [`ask`]      | [`emit`]              | One-shot Q&A, returns [`RunOutput`] |
//! | [`stream`]   | [`on`] + [`emit`]     | Streaming with per-delta callback  |
//! | [`on_event`] | [`on`]                | Lifecycle event subscription       |
//!
//! Run with:
//! ```bash
//! cargo run --example invocation_facade
//! ```
//!
//! Requires provider env vars (same as [`run_streaming`](run_streaming)).
//!
//! [`emit`]: RuntimeInvocation::emit
//! [`on`]: RuntimeInvocation::on

use std::sync::Arc;

use behest::prelude::*;
use behest::runtime::AgentEvent;

// ---------------------------------------------------------------------------
// Facade definition
// ---------------------------------------------------------------------------

/// High-level facade over [`RuntimeInvocation`]'s `emit` / `on`.
///
/// Carries provider and model identity so callers don't repeat them on every
/// call. Wraps the raw invocation primitives into task-oriented methods:
///
/// ```ignore
/// let facade = AgentFacade::new(invocation, provider, model);
/// let output = facade.ask("What is Rust?").await?;
/// facade.stream("Write a poem", |token| print!("{token}")).await?;
/// ```
#[derive(Clone)]
struct AgentFacade {
    inv: RuntimeInvocation,
    provider: ProviderId,
    model: ModelName,
}

impl AgentFacade {
    /// Wraps a [`RuntimeInvocation`] with a fixed provider and model.
    fn new(inv: RuntimeInvocation, provider: ProviderId, model: ModelName) -> Self {
        Self {
            inv,
            provider,
            model,
        }
    }

    /// One-shot Q&A.
    ///
    /// Wraps [`RuntimeInvocation::emit`] with provider/model already set.
    /// Returns the final [`RunOutput`] on success.
    async fn ask(&self, input: &str) -> Result<RunOutput, InvocationError> {
        let provider = self.provider.clone();
        let model = self.model.clone();
        let input = input.to_owned();
        self.inv
            .emit(move |_session, _control| async move { EmitRequest::new(provider, model, input) })
            .await
    }

    /// Streaming Q&A.
    ///
    /// Registers a [`TextDelta`] listener via [`RuntimeInvocation::on`], then
    /// calls [`ask`](Self::ask). Each text delta is forwarded to `on_delta`.
    /// The listener is aborted when this method returns.
    async fn stream<F>(&self, input: &str, on_delta: F) -> Result<RunOutput, InvocationError>
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        let handle = self
            .inv
            .on(EventKind::TextDelta, {
                let on_delta = Arc::new(on_delta);
                move |envelope, _session, _control| {
                    let on_delta = Arc::clone(&on_delta);
                    if let AgentEvent::TextDelta(td) = &envelope.event {
                        on_delta(td.delta.clone());
                    }
                    async {}
                }
            })
            .await?;

        let provider = self.provider.clone();
        let model = self.model.clone();
        let input = input.to_owned();
        let result = self
            .inv
            .emit(move |_session, _control| async move { EmitRequest::new(provider, model, input) })
            .await;

        drop(handle);
        result
    }

    /// Subscribes to runtime lifecycle events.
    ///
    /// Returns an [`InvocationHandle`] whose [`Drop`] implementation aborts the
    /// listener task. Callers should keep the handle alive for the desired
    /// subscription duration.
    async fn on_event<F, Fut>(
        &self,
        kind: EventKind,
        handler: F,
    ) -> Result<InvocationHandle, InvocationError>
    where
        F: Fn(RuntimeEventEnvelope, InvocationSession, Control) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        self.inv.on(kind, handler).await
    }

    /// Injects typed data into all handlers via [`RuntimeInvocation::set_data`].
    ///
    /// Values are keyed by [`TypeId`]; storing a second value of the same type
    /// overwrites the first. Retrievable inside handlers via [`Control::data`].
    fn set_data<T: Send + Sync + 'static>(&self, val: T) {
        self.inv.set_data(val);
    }
}

// ---------------------------------------------------------------------------
// Provider loading helpers (self-contained, same pattern as run_streaming)
// ---------------------------------------------------------------------------

#[cfg(feature = "openai")]
fn load_openapi_config() -> Option<ProviderConfig> {
    let base_url = std::env::var("BEHEST_OPENAPI_BASE_URL").ok()?;
    let api_key = std::env::var("BEHEST_OPENAPI_KEY").ok()?;
    let mut cfg = ProviderConfig::new(base_url)
        .with_provider_type(ProviderType::OpenAi)
        .with_api_key(api_key);
    if let Ok(model) = std::env::var("BEHEST_OPENAPI_MODEL") {
        cfg = cfg.with_model(model);
    }
    Some(cfg)
}

#[cfg(not(feature = "openai"))]
fn load_openapi_config() -> Option<ProviderConfig> {
    let _ = std::env::var("BEHEST_OPENAPI_BASE_URL").ok()?;
    None
}

#[cfg(feature = "anthropic")]
fn load_anthropic_config() -> Option<ProviderConfig> {
    let base_url = std::env::var("BEHEST_ANTHROPIC_BASE_URL").ok()?;
    let api_key = std::env::var("BEHEST_ANTHROPIC_KEY").ok()?;
    let mut cfg = ProviderConfig::new(base_url)
        .with_provider_type(ProviderType::Anthropic)
        .with_api_key(api_key);
    if let Ok(model) = std::env::var("BEHEST_ANTHROPIC_MODEL") {
        cfg = cfg.with_model(model);
    }
    Some(cfg)
}

#[cfg(not(feature = "anthropic"))]
fn load_anthropic_config() -> Option<ProviderConfig> {
    let _ = std::env::var("BEHEST_ANTHROPIC_BASE_URL").ok()?;
    None
}

#[derive(Clone, Debug)]
struct AppConfig {
    max_tokens: u32,
    temperature: f32,
}

// ---------------------------------------------------------------------------
// Demo helpers — each covers one facade usage pattern
// ---------------------------------------------------------------------------

async fn demo_stream(facade: &AgentFacade) -> Result<(), behest::Error> {
    let run_completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = Arc::clone(&run_completed);

    let _lifecycle = facade
        .on_event(EventKind::RunCompleted, move |_, _, _| {
            flag.store(true, std::sync::atomic::Ordering::Release);
            async {}
        })
        .await
        .map_err(|e| behest::Error::Config(e.to_string()))?;

    println!("\n--- on_event: registered RunCompleted handler ---");

    println!("\n--- stream: sending greeting ---");
    let output = facade
        .stream("Hello! Give me a one-sentence greeting.", |token| {
            print!("{token}");
        })
        .await
        .map_err(|e| behest::Error::Config(e.to_string()))?;

    println!();
    println!(
        "run completed: iterations={}, finish={:?}, tokens={}+{}",
        output.iterations,
        output.finish_reason,
        output.total_usage.input_tokens,
        output.total_usage.output_tokens,
    );
    println!(
        "RunCompleted event received: {}",
        run_completed.load(std::sync::atomic::Ordering::Acquire)
    );

    Ok(())
}

async fn demo_ask(facade: &AgentFacade) -> Result<(), behest::Error> {
    println!("\n--- ask: one-shot Q&A ---");
    let output = facade
        .ask("What is 42 in decimal?")
        .await
        .map_err(|e| behest::Error::Config(e.to_string()))?;
    println!(
        "finish={:?}, tokens={}+{}",
        output.finish_reason, output.total_usage.input_tokens, output.total_usage.output_tokens,
    );
    Ok(())
}

async fn demo_context_injection(facade: &AgentFacade) -> Result<(), behest::Error> {
    println!("\n--- set_data: injecting typed context into handlers ---");

    facade.set_data(AppConfig {
        max_tokens: 2048,
        temperature: 0.7,
    });

    let _cfg_handler = facade
        .on_event(
            EventKind::RunStarted,
            move |_, _session, control| async move {
                if let Some(cfg) = control.data::<AppConfig>() {
                    println!(
                        "  [handler] AppConfig: max_tokens={}, temperature={}",
                        cfg.max_tokens, cfg.temperature
                    );
                }
            },
        )
        .await
        .map_err(|e| behest::Error::Config(e.to_string()))?;

    let _ = facade
        .ask("Hello! Say 'config loaded'.")
        .await
        .map_err(|e| behest::Error::Config(e.to_string()))?;

    println!("(AppConfig flows via set_data -> handler through Control.data)");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), behest::Error> {
    let openapi = load_openapi_config();
    let anthropic = load_anthropic_config();

    let mut builder = AgentConfigBuilder::default().with_env("BEHEST")?;
    let mut provider_id = None;
    let mut model = None;

    if let Some(cfg) = openapi {
        let id = ProviderId::new("openai");
        model.clone_from(&cfg.model);
        builder = builder.with_provider(id.clone(), cfg);
        provider_id = Some(id);
    }
    if let Some(cfg) = anthropic {
        let id = ProviderId::new("anthropic");
        if model.is_none() {
            model.clone_from(&cfg.model);
        }
        if provider_id.is_none() {
            builder = builder.with_provider(id.clone(), cfg);
            provider_id = Some(id);
        }
    }

    let Some(provider_id) = provider_id else {
        println!("No provider configured; set BEHEST_OPENAPI_* or BEHEST_ANTHROPIC_* env vars");
        return Ok(());
    };
    let model = model.unwrap_or_else(|| ModelName::new("gpt-4o"));

    let config = builder.build()?;
    let runtime = Arc::new(config.into_runtime().await?);

    let invocation = RuntimeInvocation::new(runtime);
    let facade = AgentFacade::new(invocation, provider_id, model);

    println!("=== AgentFacade constructed ===");
    println!("provider: {}", facade.provider);
    println!("model:    {}", facade.model);

    demo_stream(&facade).await?;
    demo_ask(&facade).await?;
    demo_context_injection(&facade).await?;

    Ok(())
}
