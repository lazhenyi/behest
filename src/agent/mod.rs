//! Agent definition and registry.
//!
//! An [`AgentDefinition`] encapsulates the identity, permissions,
//! and behaviour of a named agent. The [`AgentRegistry`] manages
//! registration, lookup, and default selection.
//!
//! # Agent modes
//!
//! - **Primary**: user-facing agents that can be directly invoked.
//! - **Subagent**: background agents spawned by the task tool.
//!
//! Ported from OpenCode's `Agent.Service` and `Agent.Info`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::provider::ModelName;

/// The operational mode of an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    /// User-facing agent, directly invocable.
    Primary,
    /// Background agent spawned by the task tool.
    Subagent,
}

/// A single permission rule for an agent.
///
/// Rules are evaluated in order with last-match-wins semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionRule {
    /// Tool name or wildcard `"*"`.
    pub tool: String,
    /// Resource pattern or wildcard `"*"`.
    pub resource: String,
    /// Permission effect.
    #[serde(flatten)]
    pub effect: PermissionEffect,
}

/// The effect of a permission rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionEffect {
    /// Allow the action.
    Allow,
    /// Deny the action.
    Deny,
    /// Ask the user for confirmation.
    Ask,
}

/// Definition of a named agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    /// Unique agent name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Operational mode.
    pub mode: AgentMode,
    /// System prompt text injected into the LLM context.
    pub system_prompt: String,
    /// Whether this agent is hidden from user-facing listings.
    #[serde(default)]
    pub hidden: bool,
    /// Permission rules for this agent.
    #[serde(default)]
    pub permissions: Vec<PermissionRule>,
    /// Override model; falls back to session model when `None`.
    #[serde(default)]
    pub model: Option<ModelName>,
    /// Maximum steps per turn.
    #[serde(default)]
    pub max_steps: Option<usize>,
}

impl AgentDefinition {
    /// Creates a new primary agent definition.
    #[must_use]
    pub fn primary(
        name: impl Into<String>,
        description: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            mode: AgentMode::Primary,
            system_prompt: system_prompt.into(),
            hidden: false,
            permissions: Vec::new(),
            model: None,
            max_steps: None,
        }
    }

    /// Creates a new subagent definition.
    #[must_use]
    pub fn subagent(
        name: impl Into<String>,
        description: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            mode: AgentMode::Subagent,
            system_prompt: system_prompt.into(),
            hidden: false,
            permissions: Vec::new(),
            model: None,
            max_steps: None,
        }
    }

    /// Marks the agent as hidden.
    #[must_use]
    pub fn hidden(mut self) -> Self {
        self.hidden = true;
        self
    }

    /// Sets permissions.
    #[must_use]
    pub fn with_permissions(mut self, permissions: Vec<PermissionRule>) -> Self {
        self.permissions = permissions;
        self
    }

    /// Sets the override model.
    #[must_use]
    pub fn with_model(mut self, model: ModelName) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the maximum steps.
    #[must_use]
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = Some(steps);
        self
    }
}

/// Registry for named agents.
#[derive(Clone, Default)]
pub struct AgentRegistry {
    agents: HashMap<String, AgentDefinition>,
    default_agent: Option<String>,
}

impl AgentRegistry {
    /// Creates an empty agent registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an agent definition.
    ///
    /// If a default agent has not been set, the first registered primary
    /// agent becomes the default.
    pub fn register(&mut self, agent: AgentDefinition) {
        if self.default_agent.is_none() && agent.mode == AgentMode::Primary && !agent.hidden {
            self.default_agent = Some(agent.name.clone());
        }
        self.agents.insert(agent.name.clone(), agent);
    }

    /// Sets the default agent explicitly.
    pub fn set_default(&mut self, name: impl Into<String>) {
        self.default_agent = Some(name.into());
    }

    /// Returns the agent with the given name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&AgentDefinition> {
        self.agents.get(name)
    }

    /// Returns all registered agents.
    #[must_use]
    pub fn list(&self) -> Vec<&AgentDefinition> {
        let mut agents: Vec<&AgentDefinition> = self.agents.values().collect();
        agents.sort_by_key(|a| &a.name);
        agents
    }

    /// Returns all non-hidden primary agents.
    #[must_use]
    pub fn list_selectable(&self) -> Vec<&AgentDefinition> {
        let mut agents: Vec<&AgentDefinition> = self
            .agents
            .values()
            .filter(|a| a.mode == AgentMode::Primary && !a.hidden)
            .collect();
        agents.sort_by_key(|a| &a.name);
        agents
    }

    /// Returns the default agent definition.
    #[must_use]
    pub fn default_agent(&self) -> Option<&AgentDefinition> {
        self.default_agent
            .as_ref()
            .and_then(|name| self.agents.get(name))
    }

    /// Returns true if no agents are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Returns the number of registered agents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.agents.len()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn registry_should_be_empty_when_new() {
        let registry = AgentRegistry::new();
        assert!(registry.is_empty());
        assert!(registry.default_agent().is_none());
    }

    #[test]
    fn registry_should_register_and_retrieve_agent() {
        let mut registry = AgentRegistry::new();
        let agent = AgentDefinition::primary("build", "Build agent", "You are a build agent.");
        registry.register(agent);

        assert_eq!(registry.len(), 1);
        let a = registry.get("build").unwrap();
        assert_eq!(a.name, "build");
        assert_eq!(a.mode, AgentMode::Primary);
    }

    #[test]
    fn registry_should_set_first_primary_as_default() {
        let mut registry = AgentRegistry::new();
        registry.register(AgentDefinition::primary("build", "Build", "prompt"));
        registry.register(AgentDefinition::subagent("general", "General", "prompt"));

        let default = registry.default_agent().unwrap();
        assert_eq!(default.name, "build");
    }

    #[test]
    fn registry_should_allow_explicit_default() {
        let mut registry = AgentRegistry::new();
        registry.register(AgentDefinition::primary("a", "A", "prompt"));
        registry.register(AgentDefinition::primary("b", "B", "prompt"));
        registry.set_default("b");

        let default = registry.default_agent().unwrap();
        assert_eq!(default.name, "b");
    }

    #[test]
    fn registry_should_list_all_agents() {
        let mut registry = AgentRegistry::new();
        registry.register(AgentDefinition::primary("build", "Build", "prompt"));
        registry.register(AgentDefinition::subagent("general", "General", "prompt").hidden());

        let all = registry.list();
        assert_eq!(all.len(), 2);
        let selectable = registry.list_selectable();
        assert_eq!(selectable.len(), 1);
    }

    #[test]
    fn agent_definition_builder() {
        let agent = AgentDefinition::primary("echo", "Echo agent", "You echo input.")
            .with_model(ModelName::new("gpt-4o-mini"))
            .with_max_steps(5)
            .hidden();

        assert_eq!(agent.name, "echo");
        assert!(agent.hidden);
        assert_eq!(agent.model.unwrap().as_str(), "gpt-4o-mini");
        assert_eq!(agent.max_steps, Some(5));
    }
}
