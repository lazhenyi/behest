//! Configuration loading from files and environment variables.

use std::path::Path;

use config::Config;
use serde::de::DeserializeOwned;

use crate::error::Result as CrateResult;

use super::AgentConfig;

/// Loads configuration from the given file path into the requested type.
///
/// The file format is detected from the extension (`.toml`, `.json`, `.yaml`, `.yml`).
///
/// # Errors
///
/// Returns an error when the file cannot be read, parsed, or deserialized into `T`.
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

/// Loads configuration from file and environment sources with layered merging.
///
/// Layers (lowest to highest priority):
/// 1. File sources (in insertion order, optional — missing files are skipped)
/// 2. Environment variable sources (in insertion order)
///
/// Manual overrides (e.g. via `AgentConfigBuilder`) can be applied on top
/// after loading.
#[derive(Default)]
pub struct ConfigLoader {
    file_sources: Vec<String>,
    env_prefixes: Vec<String>,
}

impl ConfigLoader {
    /// Creates a new [`ConfigLoader`] with no sources registered.
    ///
    /// Call [`with_file`](Self::with_file) and [`with_env`](Self::with_env)
    /// to add sources before calling [`load`](Self::load).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a config file to the loading chain (optional — missing files are skipped).
    #[must_use]
    pub fn with_file(mut self, path: impl Into<String>) -> Self {
        self.file_sources.push(path.into());
        self
    }

    /// Adds an environment variable prefix to the loading chain.
    ///
    /// Variables are matched case-insensitively; nested keys use `__` as separator.
    #[must_use]
    pub fn with_env(mut self, prefix: impl Into<String>) -> Self {
        self.env_prefixes.push(prefix.into());
        self
    }

    /// Loads and merges all registered sources into the requested configuration type.
    ///
    /// Placeholders (`${VAR}` or `${VAR:-default}`) are substituted after merging.
    ///
    /// # Errors
    ///
    /// Returns an error when any source cannot be read, parsed, or deserialized into `T`.
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

        let mut value: serde_json::Value = settings.try_deserialize().map_err(|e| {
            crate::error::Error::Config(format!("failed to deserialize config: {e}"))
        })?;

        substitute_json(&mut value);

        serde_json::from_value(value)
            .map_err(|e| crate::error::Error::Config(format!("failed to parse final config: {e}")))
    }
}

/// Recursively traverses a JSON value and substitutes `${VAR}` / `${VAR:-default}` placeholders in all strings.
///
/// The operation modifies `value` in place.
pub fn substitute_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::String(s) => {
            *s = substitute_string(s);
        }
        serde_json::Value::Object(map) => {
            for val in map.values_mut() {
                substitute_json(val);
            }
        }
        serde_json::Value::Array(arr) => {
            for val in arr {
                substitute_json(val);
            }
        }
        _ => {}
    }
}

/// Replaces `${VAR_NAME}` or `${VAR_NAME:-default}` patterns with environment variable values.
///
/// When a variable is unset and no default is provided, the placeholder is replaced
/// with an empty string.
#[must_use]
pub fn substitute_string(input: &str) -> String {
    let mut result = String::new();
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '$' && chars.peek() == Some(&'{') {
            chars.next(); // consume '{'
            let mut placeholder = String::new();
            let mut closed = false;
            for pc in chars.by_ref() {
                if pc == '}' {
                    closed = true;
                    break;
                }
                placeholder.push(pc);
            }

            if closed {
                if let Some(pos) = placeholder.find(":-") {
                    let var_name = &placeholder[..pos];
                    let default_val = &placeholder[pos + 2..];
                    match std::env::var(var_name) {
                        Ok(val) => result.push_str(&val),
                        Err(_) => result.push_str(default_val),
                    }
                } else {
                    let var_name = &placeholder;
                    if let Ok(val) = std::env::var(var_name) {
                        result.push_str(&val);
                    }
                    // If not set and no default, replace with empty string
                }
            } else {
                result.push_str("${");
                result.push_str(&placeholder);
            }
        } else {
            result.push(c);
        }
    }
    result
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

/// Recursively merges `overlay` JSON into `base` JSON, modifying `base` in place.
///
/// When both values are JSON objects, keys from `overlay` are merged recursively.
/// When either value is not an object, `overlay` replaces `base` entirely.
pub fn merge_json(base: &mut serde_json::Value, overlay: serde_json::Value) {
    match (base, overlay) {
        (serde_json::Value::Object(base_map), serde_json::Value::Object(overlay_map)) => {
            for (key, val) in overlay_map {
                match base_map.get_mut(&key) {
                    Some(base_val) => merge_json(base_val, val),
                    None => {
                        base_map.insert(key, val);
                    }
                }
            }
        }
        (base, overlay) => {
            *base = overlay;
        }
    }
}
