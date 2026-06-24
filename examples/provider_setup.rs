//! Demonstrates programmatic provider configuration and registry.
//!
//! To use with a real provider, set your API key:
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! cargo run --example provider_setup --features openai
//! ```

use behest::prelude::*;

fn main() -> Result<(), behest::Error> {
    let config = AgentConfigBuilder::default()
        .with_provider(
            ProviderId::new("openai"),
            ProviderConfig {
                provider_type: None,
                // To use OpenAI, enable feature "openai" and set:
                // provider_type: Some(ProviderType::OpenAi),
                base_url: "https://api.openai.com/v1".to_string(),
                model: Some(ModelName::new("gpt-4o")),
                models: Vec::new(),
                compaction_model: None,
                api_key: None,
                organization: None,
                timeout_secs: 60,
                connect_timeout_secs: 10,
                max_retries: 2,
            },
        )
        .build()?;

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
