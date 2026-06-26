//! Plan operator — produce a plan for achieving the goal.

use super::call_llm_and_record_step;

use async_trait::async_trait;

use crate::reasoning::error::ReasoningError;
use crate::reasoning::operator::{BaseOperator, OperatorContext, OperatorKind, ReasoningOperator};
use crate::reasoning::state::{Plan, TaskState};

/// Produces a plan for achieving the goal.
pub struct PlanOperator {
    base: BaseOperator,
}

impl PlanOperator {
    const SYSTEM_PROMPT: &str = "Create a detailed step-by-step plan. Output each step on a new line starting with '- '. The first line should be the plan rationale.";

    /// Creates a new `PlanOperator`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: BaseOperator::new(OperatorKind::Plan, "Plan")
                .with_prompt("Create a step-by-step plan to achieve the goal."),
        }
    }

    /// Sets the prompt template.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.base.prompt = prompt.into();
        self
    }
}

impl Default for PlanOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ReasoningOperator for PlanOperator {
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
        let state = state.with_plan(plan);
        Ok(state)
    }
}
