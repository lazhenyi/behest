//! Optimize operator — optimize an existing solution.

use super::call_llm_and_record_step;

use async_trait::async_trait;

use crate::reasoning::error::ReasoningError;
use crate::reasoning::operator::{BaseOperator, OperatorContext, OperatorKind, ReasoningOperator};
use crate::reasoning::state::{Artifact, ArtifactKind, TaskState};

/// Optimizes an existing solution.
pub struct OptimizeOperator {
    base: BaseOperator,
}

impl OptimizeOperator {
    const SYSTEM_PROMPT: &str =
        "Optimize the existing solution. Identify inefficiencies and suggest improvements.";

    /// Creates a new `OptimizeOperator`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: BaseOperator::new(OperatorKind::Optimize, "Optimize")
                .with_prompt("Optimize the solution for better performance or quality."),
        }
    }

    /// Sets the prompt template.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.base.prompt = prompt.into();
        self
    }
}

impl Default for OptimizeOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ReasoningOperator for OptimizeOperator {
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
