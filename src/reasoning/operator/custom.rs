//! Custom operator — user-defined operator via closure.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;

use crate::reasoning::error::ReasoningError;
use crate::reasoning::operator::{BaseOperator, OperatorContext, OperatorKind, ReasoningOperator};
use crate::reasoning::state::TaskState;

/// Type alias for the apply closure used by [`CustomOperator`].
pub type ApplyFn = Arc<
    dyn Fn(
            &OperatorContext<'_>,
            TaskState,
        ) -> Pin<Box<dyn Future<Output = Result<TaskState, ReasoningError>> + Send>>
        + Send
        + Sync,
>;

/// A user-defined operator backed by a closure.
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use behest::reasoning::operator::{OperatorKind, ReasoningOperator};
/// use behest::reasoning::operator::custom::CustomOperator;
///
/// let op = CustomOperator::new(
///     OperatorKind::Reason,
///     "MyReason",
///     Arc::new(|_ctx, state| {
///         Box::pin(async move { Ok(state) })
///     }),
/// );
///
/// assert_eq!(op.kind(), OperatorKind::Reason);
/// assert_eq!(op.name(), "MyReason");
/// ```
pub struct CustomOperator {
    base: BaseOperator,
    apply_fn: ApplyFn,
}

impl CustomOperator {
    /// Creates a new `CustomOperator` with the given kind, name, and apply closure.
    #[must_use]
    pub fn new(kind: OperatorKind, name: &'static str, apply_fn: ApplyFn) -> Self {
        Self {
            base: BaseOperator::new(kind, name),
            apply_fn,
        }
    }

    /// Sets the prompt template.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.base.prompt = prompt.into();
        self
    }
}

#[async_trait]
impl ReasoningOperator for CustomOperator {
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
        (self.apply_fn)(ctx, state).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn custom_operator_apply() {
        let op = CustomOperator::new(
            OperatorKind::Reason,
            "TestCustom",
            Arc::new(|_ctx, state| Box::pin(async move { Ok(state) })),
        );

        assert_eq!(op.kind(), OperatorKind::Reason);
        assert_eq!(op.name(), "TestCustom");

        let ctx = OperatorContext {
            invocation: None,
            control: None,
            memory: None,
            llm: None,
        };
        let state = TaskState::new("g", "i");
        let result = op.apply(&ctx, state).await.unwrap();
        assert_eq!(result.goal.description, "g");
    }

    #[test]
    fn custom_operator_with_prompt() {
        let op = CustomOperator::new(
            OperatorKind::Analyze,
            "MyOp",
            Arc::new(|_ctx, state| Box::pin(async move { Ok(state) })),
        )
        .with_prompt("analyze this");

        assert_eq!(op.prompt(), "analyze this");
    }

    #[test]
    fn custom_operator_set_prompt() {
        let mut op = CustomOperator::new(
            OperatorKind::Analyze,
            "MyOp",
            Arc::new(|_ctx, state| Box::pin(async move { Ok(state) })),
        );
        op.set_prompt("new prompt".into());
        assert_eq!(op.prompt(), "new prompt");
    }
}
