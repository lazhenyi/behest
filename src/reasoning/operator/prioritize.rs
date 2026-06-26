//! Prioritize operator — determine priority of tasks or options.

use super::call_llm_and_record_step;

use async_trait::async_trait;

use crate::reasoning::error::ReasoningError;
use crate::reasoning::operator::{BaseOperator, OperatorContext, OperatorKind, ReasoningOperator};
use crate::reasoning::state::{Artifact, ArtifactKind, TaskState};

/// Determines priority of tasks or options.
pub struct PrioritizeOperator {
    base: BaseOperator,
}

impl PrioritizeOperator {
    const SYSTEM_PROMPT: &str = "Determine the priority of tasks or options. Output a numbered list in order of importance.";

    /// Creates a new `PrioritizeOperator`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: BaseOperator::new(OperatorKind::Prioritize, "Prioritize")
                .with_prompt("Prioritize tasks or options based on importance and urgency."),
        }
    }

    /// Sets the prompt template.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.base.prompt = prompt.into();
        self
    }
}

impl Default for PrioritizeOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ReasoningOperator for PrioritizeOperator {
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
