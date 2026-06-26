//! Demonstrates programmatic provider configuration from environment variables.
//!
//! Supported env vars:
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
//! If none of the env vars are set, the example exits silently.
//!
//! Run with:
//! ```bash
//! BEHEST_OPENAPI_BASE_URL=https://api.openai.com/v1 \
//! BEHEST_OPENAPI_KEY=sk-test \
//! BEHEST_OPENAPI_MODEL=gpt-4o \
//! cargo run --example provider_setup --features openai
//! ```

use behest::prelude::*;
use std::env;

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
    let _base_url = env::var("BEHEST_OPENAPI_BASE_URL").ok()?;
    let _api_key = env::var("BEHEST_OPENAPI_KEY").ok()?;
    eprintln!("note: openai feature is disabled, skipping openapi provider");
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
    let _base_url = env::var("BEHEST_ANTHROPIC_BASE_URL").ok()?;
    let _api_key = env::var("BEHEST_ANTHROPIC_KEY").ok()?;
    eprintln!("note: anthropic feature is disabled, skipping anthropic provider");
    None
}

fn main() -> Result<(), behest::Error> {
    let openapi = load_openapi_config();
    let anthropic = load_anthropic_config();

    if openapi.is_none() && anthropic.is_none() {
        return Ok(());
    }

    let mut builder = AgentConfigBuilder::default();
    if let Some(cfg) = openapi {
        builder = builder.with_provider(ProviderId::new("openai"), cfg);
    }
    if let Some(cfg) = anthropic {
        builder = builder.with_provider(ProviderId::new("anthropic"), cfg);
    }
    let config = builder.build()?;

    config.validate()?;

    println!("Providers:");
    for (id, cfg) in &config.providers {
        println!("  {id}: {:?}", cfg.provider_type);
        if let Some(ref model) = cfg.model {
            println!("    default model: {model}");
        }
    }

    Ok(())
}
