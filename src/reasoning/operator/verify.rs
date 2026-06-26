//! Verify operator — verify that a result satisfies constraints.

use super::call_llm_and_record_step;

use async_trait::async_trait;

use crate::reasoning::error::ReasoningError;
use crate::reasoning::operator::{BaseOperator, OperatorContext, OperatorKind, ReasoningOperator};
use crate::reasoning::state::TaskState;

/// Verifies that a result satisfies constraints.
pub struct VerifyOperator {
    base: BaseOperator,
}

impl VerifyOperator {
    const SYSTEM_PROMPT: &str = "Verify that the results meet all constraints. If verification fails, start your response with 'FAIL:' followed by the reason.";

    /// Creates a new `VerifyOperator`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: BaseOperator::new(OperatorKind::Verify, "Verify")
                .with_prompt("Verify the results meet all constraints and requirements."),
        }
    }

    /// Sets the prompt template.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.base.prompt = prompt.into();
        self
    }
}

impl Default for VerifyOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ReasoningOperator for VerifyOperator {
    fn kind(&self) -> OperatorKind {
        self.base.kind
    }

    fn name(&self) -> &'static str {
        self.base.name
    }

    fn prompt(&self) -> &str {
        &self.base.prompt
    }

    fn set_prompt(&mut self, prompt: String) {
        self.base.prompt = prompt;
    }

    async fn apply(
        &self,
        ctx: &OperatorContext<'_>,
        state: TaskState,
    ) -> Result<TaskState, ReasoningError> {
        let (response, state) = call_llm_and_record_step(
            ctx,
            self.base.kind,
            self.base.name,
            &self.base.prompt,
            state,
            Self::SYSTEM_PROMPT,
        )
        .await?;
        let trimmed = response.trim();
        if trimmed.starts_with("FAIL:") || trimmed.starts_with("[FAIL]") {
            let msg = trimmed
                .trim_start_matches("FAIL:")
                .trim_start_matches("[FAIL]")
                .trim()
                .to_string();
            return Err(ReasoningError::VerificationFailed { message: msg });
        }
        Ok(state)
    }
}
