//! Makes a real API request to a configured provider and streams the
//! response tokens to stdout as they arrive, showing the raw streaming
//! cycle without session management or tool execution.
//!
//! Requires at least one of `openai` or `anthropic` features and the
//! corresponding environment variables:
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
//! cargo run --example real_request
//! ```

use std::env;
use std::io::{self, Write};

use behest::prelude::*;
use futures_util::StreamExt;

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
    let mut provider_id: Option<ProviderId> = None;
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

    let Some(pid) = provider_id else {
        println!("No provider configured; set BEHEST_OPENAPI_* or BEHEST_ANTHROPIC_* env vars");
        return Ok(());
    };
    let model = model.unwrap_or_else(|| ModelName::new("gpt-4o"));

    let config = builder.build()?;
    let runtime = config.into_runtime().await?;

    let request = ChatRequest::new(model).with_user_text("Hello! Give me a one-sentence greeting.");
    let mut stream = runtime.providers().stream(&pid, request).await?;

    let mut stdout = io::stdout().lock();
    let mut finish_reason = None;
    let mut usage = None;

    while let Some(result) = stream.next().await {
        match result? {
            ChatStreamEvent::Started {
                provider: ref pid,
                model: ref model_name,
            } => {
                let _ = writeln!(stdout, "[streaming from {pid} / {model_name}]");
            }
            ChatStreamEvent::TextDelta { delta }
            | ChatStreamEvent::ToolCallArgumentsDelta { delta, .. } => {
                let _ = write!(stdout, "{delta}");
                let _ = stdout.flush();
            }
            ChatStreamEvent::ToolCallStarted { id, name } => {
                let _ = writeln!(stdout, "\n[tool call {id}: {name}]");
            }
            ChatStreamEvent::ToolCallCompleted { call } => {
                let _ = writeln!(stdout, "\n[tool completed: {}]", call.name);
            }
            ChatStreamEvent::Finished {
                finish_reason: reason,
                usage: tok,
            } => {
                finish_reason = Some(reason);
                usage = tok;
            }
            _ => {}
        }
    }

    let _ = writeln!(stdout);
    let reason = finish_reason.unwrap_or(FinishReason::Error);
    let _ = writeln!(
        stdout,
        "Finish: {reason:?}  |  Tokens: {in_tok} in + {out_tok} out",
        in_tok = usage.map_or(0, |u| u.input_tokens),
        out_tok = usage.map_or(0, |u| u.output_tokens),
    );

    Ok(())
}
