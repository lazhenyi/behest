//! Demonstrates building and validating an `AgentConfig` with env vars.
//!
//! Run with:
//! ```bash
//! cargo run --example hello_config
//! ```

use behest::prelude::*;

fn main() -> Result<(), behest::Error> {
    let config = AgentConfigBuilder::default().with_env("BEHEST")?.build()?;

    config.validate()?;

    println!(
        "Runtime policy: max_retries={}",
        config.runtime.policy.max_retries
    );
    println!("Session backend: {:?}", config.stores.session_backend);
    println!("Execution backend: {:?}", config.stores.execution_backend);

    Ok(())
}
