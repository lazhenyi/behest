//! Declarative component and transport configuration for `behest`.
//!
//! In the composable runtime model, every pluggable element is described
//! uniformly as a [`ComponentConfig`]: a stable `kind` identifier (e.g.
//! `"provider.openai"`), a user-assigned `name`, a list of dependency
//! names, and a free-form `config` blob that the corresponding factory
//! deserializes. Transports are described by the same shape via
//! [`TransportConfig`].
//!
//! # Loading
//!
//! [`ComponentConfig`] and [`TransportConfig`] are both `Deserialize`.
//! They can be loaded from any TOML/JSON/YAML file using the same
//! `with_file` plumbing as the rest of [`crate::config`]. Operators
//! author config files like:
//!
//! ```toml
//! [[component]]
//! name = "openai-primary"
//! kind = "provider.openai"
//! config = { api_key = "${OPENAI_API_KEY}", default_model = "gpt-4o" }
//!
//! [[component]]
//! name = "redis-sessions"
//! kind = "store.session"
//! config = { url = "redis://localhost:6379", ttl_seconds = 86400 }
//!
//! [[transport]]
//! name = "grpc-public"
//! kind = "transport.grpc"
//! config = { addr = "0.0.0.0:50051" }
//! ```
//!
//! # Resolution
//!
//! The [`ComponentRegistry`](crate::runtime::registry::ComponentRegistry)
//! consumes `ComponentConfig` values directly. There is no central
//! factory registry in this crate: each component kind is registered
//! into the registry by the user (or by a future `component_kind!`
//! macro) via
//! [`ComponentFactory`](crate::runtime::registry::ComponentFactory).

#![allow(clippy::pedantic)]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Declarative description of a runtime component.
///
/// `kind` selects the factory; `name` is the user-assigned instance
/// identifier; `depends_on` lists the names of components that must
/// be initialized first; `config` is forwarded to the factory as
/// `serde_json::Value`.
///
/// # Examples
///
/// ```json
/// {
///   "name": "openai-primary",
///   "kind": "provider.openai",
///   "depends_on": [],
///   "config": { "api_key": "sk-...", "default_model": "gpt-4o" }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ComponentConfig {
    /// User-assigned instance name. Must be unique within a registry.
    pub name: String,
    /// Component kind identifier. Used to dispatch to the appropriate
    /// factory. Examples: `"provider.openai"`, `"store.session"`,
    /// `"tool.http"`.
    pub kind: String,
    /// Names of other components this component depends on. Used by
    /// the registry to compute a topological init order.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Whether the registry should treat an init failure of this
    /// component as fatal. Default `false` (i.e. the registry
    /// continues with the rest of the system and records the
    /// component as [`ComponentState::Failed`](crate::runtime::registry::ComponentState::Failed)).
    #[serde(default)]
    pub required: bool,
    /// Free-form configuration payload. The factory selected by
    /// `kind` deserializes this into its concrete `Config` type.
    #[serde(default = "serde_json::Value::default")]
    pub config: serde_json::Value,
    /// Tags for grouping and querying components. Not used by the
    /// registry itself; intended for observability and tooling.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl ComponentConfig {
    /// Construct a new `ComponentConfig` with the given name and kind.
    #[must_use]
    pub fn new(name: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: kind.into(),
            depends_on: Vec::new(),
            required: false,
            config: serde_json::Value::Null,
            tags: Vec::new(),
        }
    }

    /// Attach a JSON configuration payload.
    #[must_use]
    pub fn with_config(mut self, config: serde_json::Value) -> Self {
        self.config = config;
        self
    }

    /// Add a dependency.
    #[must_use]
    pub fn with_dependency(mut self, dep: impl Into<String>) -> Self {
        self.depends_on.push(dep.into());
        self
    }

    /// Mark as required (init failure is fatal).
    #[must_use]
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }
}

/// Declarative description of a transport.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TransportConfig {
    /// User-assigned instance name. Must be unique within a hub.
    pub name: String,
    /// Transport kind identifier. Examples: `"transport.grpc"`,
    /// `"transport.http"`.
    pub kind: String,
    /// Free-form configuration payload. The transport factory
    /// deserializes this into its concrete config type.
    #[serde(default = "serde_json::Value::default")]
    pub config: serde_json::Value,
    /// Whether the transport is enabled. Default `true`. Disabled
    /// transports are still loaded but are not started.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_enabled() -> bool {
    true
}

impl TransportConfig {
    /// Construct a new `TransportConfig` with the given name and kind.
    #[must_use]
    pub fn new(name: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: kind.into(),
            config: serde_json::Value::Null,
            enabled: true,
        }
    }

    /// Attach a JSON configuration payload.
    #[must_use]
    pub fn with_config(mut self, config: serde_json::Value) -> Self {
        self.config = config;
        self
    }

    /// Disable the transport (it will be loaded but not started).
    #[must_use]
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

/// A collection of component and transport configs loaded from a
/// single file. Useful for layered config files where operators
/// declare components and transports in `[[component]]` /
/// `[[transport]]` TOML sections without touching the rest of the
/// runtime config.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ComponentFile {
    /// Component declarations.
    #[serde(default)]
    pub component: Vec<ComponentConfig>,
    /// Transport declarations.
    #[serde(default)]
    pub transport: Vec<TransportConfig>,
}

impl ComponentFile {
    /// Parse a `ComponentFile` from a TOML string.
    ///
    /// # Errors
    /// Returns a TOML parse error if the input is not valid TOML.
    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// Parse a `ComponentFile` from a JSON string.
    ///
    /// # Errors
    /// Returns a JSON parse error if the input is not valid JSON.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Parse a `ComponentFile` from a YAML string.
    ///
    /// # Errors
    /// Returns a YAML parse error if the input is not valid YAML.
    #[allow(dead_code, unused_variables)]
    pub fn from_yaml(text: &str) -> Result<Self, String> {
        // YAML support requires the `serde_yaml` crate; the project
        // does not currently depend on it. Use `from_toml` or
        // `from_json` instead, or call `serde_yaml::from_str`
        // directly after adding the dependency.
        Err("YAML support is not compiled in; use TOML or JSON".to_string())
    }

    /// Total number of declarations (components + transports).
    #[must_use]
    pub fn len(&self) -> usize {
        self.component.len() + self.transport.len()
    }

    /// Returns `true` if there are no declarations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_config_builder_chains() {
        let cfg = ComponentConfig::new("openai-primary", "provider.openai")
            .with_config(serde_json::json!({ "api_key": "sk-test" }))
            .with_dependency("redis-sessions")
            .required();
        assert_eq!(cfg.name, "openai-primary");
        assert_eq!(cfg.kind, "provider.openai");
        assert_eq!(cfg.depends_on, vec!["redis-sessions".to_string()]);
        assert!(cfg.required);
        assert_eq!(cfg.config["api_key"], "sk-test");
    }

    #[test]
    fn transport_config_builder_chains() {
        let cfg = TransportConfig::new("grpc-public", "transport.grpc")
            .with_config(serde_json::json!({ "addr": "0.0.0.0:50051" }))
            .disabled();
        assert_eq!(cfg.name, "grpc-public");
        assert_eq!(cfg.kind, "transport.grpc");
        assert!(!cfg.enabled);
    }

    #[test]
    fn component_file_parses_toml() {
        let toml_str = r#"
[[component]]
name = "openai-primary"
kind = "provider.openai"
config = { api_key = "sk-test" }

[[component]]
name = "redis-sessions"
kind = "store.session"
config = { url = "redis://localhost" }
depends_on = []

[[transport]]
name = "grpc-public"
kind = "transport.grpc"
config = { addr = "0.0.0.0:50051" }
"#;
        let file = match ComponentFile::from_toml(toml_str) {
            Ok(f) => f,
            Err(e) => panic!("parse toml: {e}"),
        };
        assert_eq!(file.component.len(), 2);
        assert_eq!(file.transport.len(), 1);
        assert_eq!(file.component[0].name, "openai-primary");
        assert_eq!(file.component[1].depends_on.len(), 0);
        assert!(file.transport[0].enabled);
    }

    #[test]
    fn component_file_is_empty_by_default() {
        let file = ComponentFile::default();
        assert!(file.is_empty());
        assert_eq!(file.len(), 0);
    }

    #[test]
    fn component_file_parses_json() {
        let json_str = r#"{
            "component": [
                {"name": "x", "kind": "k", "config": {"a": 1}}
            ]
        }"#;
        let file = match ComponentFile::from_json(json_str) {
            Ok(f) => f,
            Err(e) => panic!("parse json: {e}"),
        };
        assert_eq!(file.component.len(), 1);
        assert_eq!(file.component[0].config["a"], 1);
    }
}
