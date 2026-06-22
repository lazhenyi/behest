//! Runtime policy configuration.
//!
//! Defines limits and constraints for agent execution,
//! including iteration limits, timeouts, and resource budgets.

use std::time::Duration;

use serde::{Deserialize, Serialize};

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
