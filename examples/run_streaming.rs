//! Demonstrates running an agent and subscribing to streaming events.
//!
//! Requires at least one of `openai` or `anthropic` features and the corresponding
//! environment variables:
//! ```text
//! BEHEST_OPENAPI_BASE_URL=https://api.openai.com/v1
//! BEHEST_OPENAPI_KEY=sk-...
//! BEHEST_OPENAPI_MODEL=gpt-4o
//!
//! BEHEST_ANTHROPIC_BASE_URL=https://api.anthropic.com
//! BEHEST_ANTHROPIC_KEY=sk-ant-...
//! BEHEST_ANTHROPIC_MODEL=claude-sonnet-4-20250514
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example run_streaming
//! ```

use std::env;
use std::sync::Arc;

use behest::prelude::*;
use behest::runtime::AgentEvent;

#[cfg(feature = "openai")]
fn load_openapi_config() -> Option<ProviderConfig> {
    let base_url = env::var("BEHEST_OPENAPI_BASE_URL").ok()?;
    let api_key = env::var("BEHEST_OPENAPI_KEY").ok()?;
    let mut cfg = ProviderConfig::new(base_url)
        .with_provider_type(ProviderType::OpenAi)
        .with_api_key(api_key);
    if let Ok(model) = env::var("BEHEST_OPENAPI_MODEL") {
        cfg = cfg.with_model(model);
    }
    Some(cfg)
}

#[cfg(not(feature = "openai"))]
fn load_openapi_config() -> Option<ProviderConfig> {
    let _ = env::var("BEHEST_OPENAPI_BASE_URL").ok()?;
    eprintln!("note: openai feature is disabled, skipping openai provider");
    None
}

#[cfg(feature = "anthropic")]
fn load_anthropic_config() -> Option<ProviderConfig> {
    let base_url = env::var("BEHEST_ANTHROPIC_BASE_URL").ok()?;
    let api_key = env::var("BEHEST_ANTHROPIC_KEY").ok()?;
    let mut cfg = ProviderConfig::new(base_url)
        .with_provider_type(ProviderType::Anthropic)
        .with_api_key(api_key);
    if let Ok(model) = env::var("BEHEST_ANTHROPIC_MODEL") {
        cfg = cfg.with_model(model);
    }
    Some(cfg)
}

#[cfg(not(feature = "anthropic"))]
fn load_anthropic_config() -> Option<ProviderConfig> {
    let _ = env::var("BEHEST_ANTHROPIC_BASE_URL").ok()?;
    eprintln!("note: anthropic feature is disabled, skipping anthropic provider");
    None
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

    let mut events = runtime.subscribe();
    tokio::spawn(async move {
        while let Ok(event) = events.recv().await {
            match &event {
                AgentEvent::RunStarted(e) => {
                    println!("[event] run started: {:?}", e.run_id);
                }
                AgentEvent::TextDelta(e) => {
                    print!("{}", e.delta);
                }
                AgentEvent::ToolCallStarted(e) => {
                    println!("\n[event] tool call: {} ({})", e.tool_name, e.call_id);
                }
                AgentEvent::ToolExecutionFinished(e) => {
                    println!(
                        "[event] tool finished: {} in {}ms",
                        e.tool_name, e.duration_ms
                    );
                }
                AgentEvent::RunCompleted(_) => {
                    println!("\n[event] run completed");
                }
                AgentEvent::RunFailed(e) => {
                    println!("\n[event] run failed: {}", e.error);
                }
                AgentEvent::UsageRecorded(e) => {
                    println!(
                        "[event] usage: {} in + {} out",
                        e.usage.input_tokens, e.usage.output_tokens
                    );
                }
                _ => {}
            }
        }
    });

    let request = RunRequest::new(
        provider_id,
        model,
        "Hello! Give me a one-sentence greeting.",
    );
    let output = runtime
        .run(request)
        .await
        .map_err(|e| behest::Error::Config(e.to_string()))?;
    println!(
        "\nRun completed: {} iterations, finish={:?}, usage={} in + {} out",
        output.iterations,
        output.finish_reason,
        output.total_usage.input_tokens,
        output.total_usage.output_tokens,
    );

    Ok(())
}
