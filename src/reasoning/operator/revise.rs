//! Revise operator — revise the current plan based on reflection.

use super::call_llm_and_record_step;

use async_trait::async_trait;

use crate::reasoning::error::ReasoningError;
use crate::reasoning::operator::{BaseOperator, OperatorContext, OperatorKind, ReasoningOperator};
use crate::reasoning::state::{Plan, Reflection, TaskState};

/// Revises the current plan based on reflection.
pub struct ReviseOperator {
    base: BaseOperator,
}

impl ReviseOperator {
    const SYSTEM_PROMPT: &str = "Revise the plan based on reflection and new insights. Output the new plan rationale on the first line, then each step on a line starting with '- '.";

    /// Creates a new `ReviseOperator`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: BaseOperator::new(OperatorKind::Revise, "Revise")
                .with_prompt("Revise the plan based on reflection and new insights."),
        }
    }

    /// Sets the prompt template.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.base.prompt = prompt.into();
        self
    }
}

impl Default for ReviseOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ReasoningOperator for ReviseOperator {
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
        let mut lines = response.lines();
        let rationale = lines.next().unwrap_or("").to_string();
        let plan = lines
            .filter(|l| l.starts_with("- "))
            .fold(Plan::new(rationale), |p, l| {
                p.with_step(l.trim_start_matches("- ").to_string())
            });
        let state = state
            .reflect(Reflection {
                content: response,
                revised_plan: Some(plan.clone()),
            })
            .with_plan(plan);
        Ok(state)
    }
}
