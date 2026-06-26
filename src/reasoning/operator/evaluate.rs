//! Evaluate operator — evaluate quality of options or results.

use super::call_llm_and_record_step;

use async_trait::async_trait;

use crate::reasoning::error::ReasoningError;
use crate::reasoning::operator::{BaseOperator, OperatorContext, OperatorKind, ReasoningOperator};
use crate::reasoning::state::{Reflection, TaskState};

/// Evaluates quality of options or results.
pub struct EvaluateOperator {
    base: BaseOperator,
}

impl EvaluateOperator {
    const SYSTEM_PROMPT: &str = "Evaluate the quality of the current results. Identify strengths, weaknesses, and areas for improvement.";

    /// Creates a new `EvaluateOperator`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: BaseOperator::new(OperatorKind::Evaluate, "Evaluate")
                .with_prompt("Evaluate the quality and correctness of the results."),
        }
    }

    /// Sets the prompt template.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.base.prompt = prompt.into();
        self
    }
}

impl Default for EvaluateOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ReasoningOperator for EvaluateOperator {
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
        let state = state.reflect(Reflection {
            content: response,
            revised_plan: None,
        });
        Ok(state)
    }
}
