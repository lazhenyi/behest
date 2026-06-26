//! Diagnose operator — diagnose problem causes.

use super::call_llm_and_record_step;

use async_trait::async_trait;

use crate::reasoning::error::ReasoningError;
use crate::reasoning::operator::{BaseOperator, OperatorContext, OperatorKind, ReasoningOperator};
use crate::reasoning::state::{Artifact, ArtifactKind, TaskState};

/// Diagnoses problem causes.
pub struct DiagnoseOperator {
    base: BaseOperator,
}

impl DiagnoseOperator {
    const SYSTEM_PROMPT: &str =
        "Diagnose the root causes of the problem. Identify symptoms and underlying issues.";

    /// Creates a new `DiagnoseOperator`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: BaseOperator::new(OperatorKind::Diagnose, "Diagnose")
                .with_prompt("Diagnose the root cause of the problem."),
        }
    }

    /// Sets the prompt template.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.base.prompt = prompt.into();
        self
    }
}

impl Default for DiagnoseOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ReasoningOperator for DiagnoseOperator {
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
