//! Configuration loading from files and environment variables.

use std::path::Path;

use config::Config;
use serde::de::DeserializeOwned;

use crate::error::Result as CrateResult;

use super::AgentConfig;

/// Loads configuration from the given file path.
///
/// The file format is detected from the extension (`.toml`, `.json`, `.yaml`, `.yml`).
///
/// # Errors
///
/// Returns an error when the file cannot be read or parsed.
pub fn load_file<T: DeserializeOwned>(path: impl AsRef<Path>) -> CrateResult<T> {
    let path = path.as_ref();
    let builder = Config::builder().add_source(config::File::from(path));

    let settings = builder.build().map_err(|e| {
        crate::error::Error::Config(format!(
            "failed to load config file {}: {e}",
            path.display()
        ))
    })?;

    settings.try_deserialize().map_err(|e| {
        crate::error::Error::Config(format!(
            "failed to deserialize config from {}: {e}",
            path.display()
        ))
    })
}

/// Loads configuration from environment variables with the given prefix.
///
/// Environment variables are matched case-insensitively. Nested keys
/// are separated by `__` (double underscore).
///
/// # Example
///
/// ```text
/// AGENTS__RUNTIME__MAX_HISTORY_MESSAGES=100
/// AGENTS__PROVIDERS__OPENAI__BASE_URL="https://api.openai.com/v1"
/// AGENTS__PROVIDERS__OPENAI__API_KEY="env:OPENAI_API_KEY"
/// ```
///
/// # Errors
///
/// Returns an error when the environment-based configuration cannot be built.
pub fn load_env<T: DeserializeOwned>(prefix: &str) -> CrateResult<T> {
    let builder =
        Config::builder().add_source(config::Environment::with_prefix(prefix).separator("__"));

    let settings = builder.build().map_err(|e| {
        crate::error::Error::Config(format!(
            "failed to load config from environment (prefix={prefix}): {e}"
        ))
    })?;

    settings.try_deserialize().map_err(|e| {
        crate::error::Error::Config(format!(
            "failed to deserialize config from environment (prefix={prefix}): {e}"
        ))
    })
}

/// Merges multiple configuration layers: manual builder (highest priority),
/// then file, then environment, then defaults.
#[derive(Default)]
pub struct ConfigLoader {
    file_sources: Vec<String>,
    env_prefixes: Vec<String>,
}

impl ConfigLoader {
    /// Creates a new config loader.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a config file to the loader.
    #[must_use]
    pub fn with_file(mut self, path: impl Into<String>) -> Self {
        self.file_sources.push(path.into());
        self
    }

    /// Adds an environment variable prefix to the loader.
    #[must_use]
    pub fn with_env(mut self, prefix: impl Into<String>) -> Self {
        self.env_prefixes.push(prefix.into());
        self
    }

    /// Loads and merges all sources into a configuration struct.
    ///
    /// # Errors
    ///
    /// Returns an error when any source cannot be read or parsed.
    pub fn load<T: DeserializeOwned>(&self) -> CrateResult<T> {
        let mut builder = Config::builder();

        for file in &self.file_sources {
            builder = builder.add_source(config::File::from(Path::new(file)).required(false));
        }

        for prefix in &self.env_prefixes {
            builder = builder.add_source(config::Environment::with_prefix(prefix).separator("__"));
        }

        let settings = builder
            .build()
            .map_err(|e| crate::error::Error::Config(format!("failed to build config: {e}")))?;

        settings
            .try_deserialize()
            .map_err(|e| crate::error::Error::Config(format!("failed to deserialize config: {e}")))
    }
}

/// Loads configuration using a layered strategy.
///
/// Order (lowest to highest priority): defaults → file → env.
/// The caller can then apply manual overrides on top via builder setters.
///
/// # Errors
///
/// Returns an error when loading or deserialization fails.
pub fn load_layered<T: DeserializeOwned>(
    file: Option<&Path>,
    env_prefix: Option<&str>,
) -> CrateResult<T> {
    let mut loader = ConfigLoader::new();
    if let Some(path) = file {
        loader = loader.with_file(path.display().to_string());
    }
    if let Some(prefix) = env_prefix {
        loader = loader.with_env(prefix);
    }
    loader.load()
}

impl AgentConfig {
    /// Loads config from a file, environment, and returns a builder
    /// pre-populated with their merged values. The caller can then
    /// add manual overrides on top.
    ///
    /// # Errors
    ///
    /// Returns an error when loading or deserialization fails.
    pub fn load(
        file: Option<&Path>,
        env_prefix: Option<&str>,
    ) -> CrateResult<super::AgentConfigBuilder> {
        let base: Self = load_layered(file, env_prefix)?;
        Ok(super::AgentConfigBuilder::from_config(base))
    }
}
