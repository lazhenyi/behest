//! Observe operator — observe the result of an action.

use super::call_llm_and_record_step;

use async_trait::async_trait;

use crate::reasoning::error::ReasoningError;
use crate::reasoning::operator::{BaseOperator, OperatorContext, OperatorKind, ReasoningOperator};
use crate::reasoning::state::{Observation, ObservationSource, TaskState};

/// Observes the result of an action.
pub struct ObserveOperator {
    base: BaseOperator,
}

impl ObserveOperator {
    const SYSTEM_PROMPT: &str =
        "Process the results of the last action. Extract key findings and observations.";

    /// Creates a new `ObserveOperator`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: BaseOperator::new(OperatorKind::Observe, "Observe")
                .with_prompt("Observe and record the results of the action taken."),
        }
    }

    /// Sets the prompt template.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.base.prompt = prompt.into();
        self
    }
}

impl Default for ObserveOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ReasoningOperator for ObserveOperator {
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
        let sequence = state.metadata.observation_sequence + 1;
        let state = state.observe(Observation {
            source: ObservationSource::Tool,
            content: response,
            sequence,
        });
        Ok(state)
    }
}
