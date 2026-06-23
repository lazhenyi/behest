//! Runtime policy configuration.
//!
//! Defines limits and constraints for agent execution,
//! including iteration limits, timeouts, resource budgets,
//! and compaction strategy.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::provider::{ModelName, ProviderId};
use crate::runtime::doom_loop::DoomLoopConfig;
use crate::runtime::input::InputAdmissionConfig;
use crate::tool_output::ToolOutputConfig;

/// Compaction configuration for automatic context compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Enable automatic compaction before provider turns. Default: `true`.
    #[serde(default = "default_true")]
    pub auto: bool,
    /// Enable old tool output pruning. Default: `false`.
    #[serde(default)]
    pub prune: bool,
    /// Token headroom between context limit and compaction trigger. Default: `20_000`.
    #[serde(default = "default_buffer")]
    pub buffer_tokens: usize,
    /// Tokens to retain as recent context after compaction. Default: `8_000`.
    #[serde(default = "default_keep")]
    pub keep_tokens: usize,
    /// Number of recent turns to preserve intact. Default: `2`.
    #[serde(default = "default_tail_turns")]
    pub tail_turns: usize,
    /// Model to use for compaction. Falls back to the run's model when `None`.
    #[serde(default)]
    pub model: Option<ModelName>,
    /// Provider to use for compaction. Falls back to the run's provider when `None`.
    #[serde(default)]
    pub provider: Option<ProviderId>,
}

const fn default_true() -> bool {
    true
}

const fn default_buffer() -> usize {
    20_000
}

const fn default_keep() -> usize {
    8_000
}

const fn default_tail_turns() -> usize {
    2
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            auto: true,
            prune: false,
            buffer_tokens: 20_000,
            keep_tokens: 8_000,
            tail_turns: 2,
            model: None,
            provider: None,
        }
    }
}

impl CompactionConfig {
    /// Creates a new compaction config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Disables automatic compaction.
    #[must_use]
    pub fn with_auto_disabled(mut self) -> Self {
        self.auto = false;
        self
    }

    /// Enables tool output pruning.
    #[must_use]
    pub fn with_prune(mut self) -> Self {
        self.prune = true;
        self
    }

    /// Sets the buffer token count.
    #[must_use]
    pub fn with_buffer_tokens(mut self, tokens: usize) -> Self {
        self.buffer_tokens = tokens;
        self
    }

    /// Sets the keep token count.
    #[must_use]
    pub fn with_keep_tokens(mut self, tokens: usize) -> Self {
        self.keep_tokens = tokens;
        self
    }

    /// Sets the number of recent turns to preserve.
    #[must_use]
    pub fn with_tail_turns(mut self, turns: usize) -> Self {
        self.tail_turns = turns;
        self
    }

    /// Sets the compaction model.
    #[must_use]
    pub fn with_model(mut self, model: ModelName) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the compaction provider.
    #[must_use]
    pub fn with_provider(mut self, provider: ProviderId) -> Self {
        self.provider = Some(provider);
        self
    }
}

/// Runtime policy for agent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimePolicy {
    /// Maximum number of model call iterations per run.
    pub max_iterations: usize,
    /// Maximum total tokens per run.
    pub max_tokens: Option<usize>,
    /// Maximum concurrent tool executions.
    pub max_tool_concurrency: usize,
    /// Timeout for individual tool execution.
    pub tool_timeout: Duration,
    /// Timeout for provider calls.
    pub provider_timeout: Duration,
    /// Whether to allow tool execution failures to continue the run.
    pub continue_on_tool_failure: bool,
    /// Whether to retry on retryable provider errors.
    pub retry_on_provider_error: bool,
    /// Maximum retries for provider calls.
    pub max_retries: usize,
    /// Compaction strategy configuration.
    #[serde(default)]
    pub compaction: CompactionConfig,
    /// Tool output truncation configuration.
    #[serde(default)]
    pub tool_output: ToolOutputConfig,
    /// Doom loop detection configuration.
    #[serde(default)]
    pub doom_loop: DoomLoopConfig,
    /// Input admission pipeline configuration.
    #[serde(default)]
    pub input_admission: InputAdmissionConfig,
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            max_tokens: None,
            max_tool_concurrency: 4,
            tool_timeout: Duration::from_secs(30),
            provider_timeout: Duration::from_secs(60),
            continue_on_tool_failure: true,
            retry_on_provider_error: true,
            max_retries: 2,
            compaction: CompactionConfig::default(),
            tool_output: ToolOutputConfig::default(),
            doom_loop: DoomLoopConfig::default(),
            input_admission: InputAdmissionConfig::default(),
        }
    }
}

impl RuntimePolicy {
    /// Creates a new policy with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the maximum tokens.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the maximum tool concurrency.
    #[must_use]
    pub fn with_max_tool_concurrency(mut self, max_tool_concurrency: usize) -> Self {
        self.max_tool_concurrency = max_tool_concurrency;
        self
    }

    /// Sets the tool timeout.
    #[must_use]
    pub fn with_tool_timeout(mut self, tool_timeout: Duration) -> Self {
        self.tool_timeout = tool_timeout;
        self
    }

    /// Sets the provider timeout.
    #[must_use]
    pub fn with_provider_timeout(mut self, provider_timeout: Duration) -> Self {
        self.provider_timeout = provider_timeout;
        self
    }

    /// Sets whether to continue on tool failure.
    #[must_use]
    pub fn with_continue_on_tool_failure(mut self, continue_on_tool_failure: bool) -> Self {
        self.continue_on_tool_failure = continue_on_tool_failure;
        self
    }

    /// Sets whether to retry on provider errors.
    #[must_use]
    pub fn with_retry_on_provider_error(mut self, retry_on_provider_error: bool) -> Self {
        self.retry_on_provider_error = retry_on_provider_error;
        self
    }

    /// Sets the maximum retries.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Sets the compaction configuration.
    #[must_use]
    pub fn with_compaction(mut self, compaction: CompactionConfig) -> Self {
        self.compaction = compaction;
        self
    }

    /// Sets the tool output truncation configuration.
    #[must_use]
    pub fn with_tool_output(mut self, config: ToolOutputConfig) -> Self {
        self.tool_output = config;
        self
    }

    /// Sets the doom loop detection configuration.
    #[must_use]
    pub fn with_doom_loop(mut self, config: DoomLoopConfig) -> Self {
        self.doom_loop = config;
        self
    }

    /// Sets the input admission pipeline configuration.
    #[must_use]
    pub fn with_input_admission(mut self, config: InputAdmissionConfig) -> Self {
        self.input_admission = config;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy() {
        let policy = RuntimePolicy::default();
        assert_eq!(policy.max_iterations, 10);
        assert_eq!(policy.max_tool_concurrency, 4);
        assert!(policy.max_tokens.is_none());
        assert!(policy.continue_on_tool_failure);
    }

    #[test]
    fn policy_builder() {
        let policy = RuntimePolicy::new()
            .with_max_iterations(5)
            .with_max_tokens(1000)
            .with_max_tool_concurrency(2);

        assert_eq!(policy.max_iterations, 5);
        assert_eq!(policy.max_tokens, Some(1000));
        assert_eq!(policy.max_tool_concurrency, 2);
    }
}
