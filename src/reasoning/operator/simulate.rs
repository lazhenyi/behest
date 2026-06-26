//! Simulate operator — simulate possible outcomes.

use super::call_llm_and_record_step;

use async_trait::async_trait;

use crate::reasoning::error::ReasoningError;
use crate::reasoning::operator::{BaseOperator, OperatorContext, OperatorKind, ReasoningOperator};
use crate::reasoning::state::{Artifact, ArtifactKind, TaskState};

/// Simulates possible outcomes.
pub struct SimulateOperator {
    base: BaseOperator,
}

impl SimulateOperator {
    const SYSTEM_PROMPT: &str = "Simulate possible outcomes of the proposed approach. Consider best case, worst case, and most likely scenarios.";

    /// Creates a new `SimulateOperator`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: BaseOperator::new(OperatorKind::Simulate, "Simulate")
                .with_prompt("Simulate possible outcomes to evaluate different options."),
        }
    }

    /// Sets the prompt template.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.base.prompt = prompt.into();
        self
    }
}

impl Default for SimulateOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ReasoningOperator for SimulateOperator {
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
        let state = state.add_artifact(Artifact {
            kind: ArtifactKind::Text,
            content: response,
        });
        Ok(state)
    }
}
