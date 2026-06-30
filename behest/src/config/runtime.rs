//! Runtime configuration section.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::runtime::policy::PromptCacheConfig;
use crate::runtime::{CompactionConfig, RuntimePolicy};

/// Runtime configuration covering policy, context limits, and event channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Execution policy (max iterations, timeouts, retries).
    #[serde(default)]
    pub policy: RuntimePolicyConfig,

    /// Maximum number of history messages before trimming. Default: 50.
    #[serde(default = "default_max_history_messages")]
    pub max_history_messages: usize,

    /// Internal broadcast channel capacity for local event subscribers. Default: 256.
    #[serde(default = "default_event_channel_capacity")]
    pub event_channel_capacity: usize,

    /// Directory for storing run recovery snapshots. `None` disables snapshotting.
    #[serde(default)]
    pub snapshot_dir: Option<String>,
}

/// Per-run execution limits and constraints.
///
/// All fields have sensible defaults; only non-default values need to be
/// set explicitly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimePolicyConfig {
    /// Maximum model call iterations per run. Default: 10.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,

    /// Maximum total tokens per run. `None` means unlimited.
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Maximum concurrent tool executions. Default: 4.
    #[serde(default = "default_max_tool_concurrency")]
    pub max_tool_concurrency: usize,

    /// Timeout for individual tool execution, in seconds. Default: 30.
    #[serde(default = "default_tool_timeout_secs")]
    pub tool_timeout_secs: u64,

    /// Timeout for provider calls, in seconds. Default: 60.
    #[serde(default = "default_provider_timeout_secs")]
    pub provider_timeout_secs: u64,

    /// Continue the run when a tool fails. Default: `true`.
    #[serde(default = "default_continue_on_tool_failure")]
    pub continue_on_tool_failure: bool,

    /// Retry on retryable provider errors. Default: `true`.
    #[serde(default = "default_retry_on_provider_error")]
    pub retry_on_provider_error: bool,

    /// Maximum retries for provider calls per attempt. Default: 2.
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            policy: RuntimePolicyConfig::default(),
            max_history_messages: default_max_history_messages(),
            event_channel_capacity: default_event_channel_capacity(),
            snapshot_dir: None,
        }
    }
}

impl Default for RuntimePolicyConfig {
    fn default() -> Self {
        Self {
            max_iterations: default_max_iterations(),
            max_tokens: None,
            max_tool_concurrency: default_max_tool_concurrency(),
            tool_timeout_secs: default_tool_timeout_secs(),
            provider_timeout_secs: default_provider_timeout_secs(),
            continue_on_tool_failure: default_continue_on_tool_failure(),
            retry_on_provider_error: default_retry_on_provider_error(),
            max_retries: default_max_retries(),
        }
    }
}

impl From<RuntimePolicyConfig> for RuntimePolicy {
    fn from(config: RuntimePolicyConfig) -> Self {
        RuntimePolicy::new()
            .with_max_iterations(config.max_iterations)
            .with_tool_timeout(Duration::from_secs(config.tool_timeout_secs))
            .with_provider_timeout(Duration::from_secs(config.provider_timeout_secs))
            .with_continue_on_tool_failure(config.continue_on_tool_failure)
            .with_retry_on_provider_error(config.retry_on_provider_error)
            .with_max_retries(config.max_retries)
            .with_max_output_recovery_attempts(3)
    }
}

const fn default_max_iterations() -> usize {
    10
}

const fn default_max_tool_concurrency() -> usize {
    4
}

const fn default_tool_timeout_secs() -> u64 {
    30
}

const fn default_provider_timeout_secs() -> u64 {
    60
}

const fn default_continue_on_tool_failure() -> bool {
    true
}

const fn default_retry_on_provider_error() -> bool {
    true
}

const fn default_max_retries() -> usize {
    2
}

const fn default_max_history_messages() -> usize {
    50
}

const fn default_event_channel_capacity() -> usize {
    256
}
