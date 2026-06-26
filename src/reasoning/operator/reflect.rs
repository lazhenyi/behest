//! Reflect operator — reflect on progress and identify improvements.

use super::call_llm_and_record_step;

use async_trait::async_trait;

use crate::reasoning::error::ReasoningError;
use crate::reasoning::operator::{BaseOperator, OperatorContext, OperatorKind, ReasoningOperator};
use crate::reasoning::state::{Reflection, TaskState};

/// Reflects on progress and identifies improvements.
pub struct ReflectOperator {
    base: BaseOperator,
}

impl ReflectOperator {
    const SYSTEM_PROMPT: &str = "Reflect on the current progress. What worked well? What could be improved? What should be tried next?";

    /// Creates a new `ReflectOperator`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: BaseOperator::new(OperatorKind::Reflect, "Reflect")
                .with_prompt("Reflect on the current progress and identify what can be improved."),
        }
    }

    /// Sets the prompt template.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.base.prompt = prompt.into();
        self
    }
}

impl Default for ReflectOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ReasoningOperator for ReflectOperator {
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
