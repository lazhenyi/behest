//! Demonstrates loading `AgentConfig` from a TOML file via `ConfigLoader`.
//!
//! Run with:
//! ```bash
//! cargo run --example config_from_toml
//! ```

use behest::config::{AgentConfig, ConfigLoader};
use std::fs;
use std::path::Path;

const TOML_CONTENT: &str = r#"
[runtime.policy]
max_iterations = 15
provider_timeout_secs = 120

[runtime]
max_history_messages = 100
snapshot_dir = "/tmp/behest_snapshots"

[stores]
session_backend = "memory"

[providers.openai]
base_url = "https://api.openai.com/v1"
api_key = "env:OPENAI_API_KEY"
"#;

fn main() -> Result<(), behest::Error> {
    let path = Path::new("/tmp/behest_example_config.toml");
    fs::write(path, TOML_CONTENT)
        .map_err(|e| behest::Error::Config(format!("failed to write temp config: {e}")))?;

    let config: AgentConfig = ConfigLoader::new()
        .with_file(path.display().to_string())
        .load()
        .map_err(|e| behest::Error::Config(format!("load failed: {e}")))?;

    println!("Loaded config from TOML file at {}", path.display());
    println!(
        "  runtime.policy.max_iterations = {}",
        config.runtime.policy.max_iterations
    );
    println!(
        "  runtime.max_history_messages = {}",
        config.runtime.max_history_messages
    );
    println!("  runtime.snapshot_dir = {:?}", config.runtime.snapshot_dir);
    println!(
        "  stores.session_backend = {:?}",
        config.stores.session_backend
    );
    let provider_count = config.providers.len();
    println!("  providers.count = {provider_count}");

    let _ = fs::remove_file(path);

    Ok(())
}
