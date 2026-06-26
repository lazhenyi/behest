//! Atomic reasoning operators and the operator trait.
//!
//! Operators are the atomic units of reasoning. Each operator receives
//! an [`OperatorContext`] and a [`TaskState`], and produces a new [`TaskState`].
//!
//! Every operator uses an [`LlmProvider`] to generate content. An optional
//! [`ChatLlmProvider`] wrapping the runtime's [`ChatProvider`] is the
//! canonical implementation.
//!
//! [`ChatLlmProvider`]: crate::reasoning::llm_provider::ChatLlmProvider
//! [`ChatProvider`]: crate::provider::ChatProvider

pub mod act;
pub mod analyze;
pub mod answer;
pub mod compose;
pub mod custom;
pub mod decompose;
pub mod diagnose;
pub mod evaluate;
pub mod explain;
pub mod generate;
pub mod hypothesize;
pub mod observe;
pub mod optimize;
pub mod plan;
pub mod prioritize;
pub mod reason;
pub mod reduce;
pub mod reflect;
pub mod revise;
pub mod search;
pub mod select;
pub mod simulate;
pub mod solve;
pub mod synthesize;
pub mod test_op;
pub mod verify;

pub use self::act::ActOperator;
pub use self::analyze::AnalyzeOperator;
pub use self::answer::AnswerOperator;
pub use self::compose::ComposeOperator;
pub use self::custom::CustomOperator;
pub use self::decompose::DecomposeOperator;
pub use self::diagnose::DiagnoseOperator;
pub use self::evaluate::EvaluateOperator;
pub use self::explain::ExplainOperator;
pub use self::generate::GenerateOperator;
pub use self::hypothesize::HypothesizeOperator;
pub use self::observe::ObserveOperator;
pub use self::optimize::OptimizeOperator;
pub use self::plan::PlanOperator;
pub use self::prioritize::PrioritizeOperator;
pub use self::reason::ReasonOperator;
pub use self::reduce::ReduceOperator;
pub use self::reflect::ReflectOperator;
pub use self::revise::ReviseOperator;
pub use self::search::SearchOperator;
pub use self::select::SelectOperator;
pub use self::simulate::SimulateOperator;
pub use self::solve::SolveOperator;
pub use self::synthesize::SynthesizeOperator;
pub use self::test_op::TestOperator;
pub use self::verify::VerifyOperator;

use std::fmt::Write;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::reasoning::error::ReasoningError;
use crate::reasoning::state::{Step, StepStatus, TaskState};
use crate::runtime::invocation::{Control, RuntimeInvocation};

// ---------------------------------------------------------------------------
// OperatorKind
// ---------------------------------------------------------------------------

/// The kind of atomic reasoning operation.
///
/// Each variant corresponds to a fundamental reasoning primitive.
/// Strategies compose these primitives into reasoning graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OperatorKind {
    /// Analyze the current state to extract structure or insights.
    Analyze,
    /// Decompose a complex goal into sub-goals.
    Decompose,
    /// Produce a plan for achieving the goal.
    Plan,
    /// Apply logical reasoning to the current state.
    Reason,
    /// Execute an action (typically a tool call).
    Act,
    /// Observe the result of an action.
    Observe,
    /// Solve a specific sub-problem.
    Solve,
    /// Verify that a result satisfies constraints.
    Verify,
    /// Reflect on progress and identify improvements.
    Reflect,
    /// Revise the current plan based on reflection.
    Revise,
    /// Select the best option from alternatives.
    Select,
    /// Compose partial results into a coherent whole.
    Compose,
    /// Reduce multiple results into a single result.
    Reduce,
    /// Explain reasoning process or results.
    Explain,
    /// Evaluate quality of options or results.
    Evaluate,
    /// Generate new options or ideas.
    Generate,
    /// Propose a hypothesis.
    Hypothesize,
    /// Test a hypothesis or solution.
    Test,
    /// Search for relevant information.
    Search,
    /// Synthesize information from multiple sources.
    Synthesize,
    /// Determine priority of tasks or options.
    Prioritize,
    /// Optimize an existing solution.
    Optimize,
    /// Simulate possible outcomes.
    Simulate,
    /// Diagnose problem causes.
    Diagnose,
    /// Produce the final answer.
    Answer,
}

impl OperatorKind {
    /// Returns `true` if this operator is a terminal operator (produces a final result).
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Answer)
    }

    /// Returns `true` if this operator typically requires tool access.
    #[must_use]
    pub const fn requires_tools(&self) -> bool {
        matches!(
            self,
            Self::Act | Self::Observe | Self::Search | Self::Test | Self::Simulate
        )
    }
}

// ---------------------------------------------------------------------------
// LlmProvider
// ---------------------------------------------------------------------------

/// LLM provider abstraction for reasoning operators.
///
/// This is a simplified interface designed for operator-level LLM calls.
/// The canonical implementation wraps [`ChatProvider`](crate::provider::ChatProvider).
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Calls the LLM with a user prompt and optional system prompt.
    ///
    /// # Errors
    ///
    /// Returns [`ReasoningError`] on provider failure.
    async fn complete(&self, prompt: &str, system: Option<&str>) -> Result<String, ReasoningError>;
}

// ---------------------------------------------------------------------------
// OperatorContext
// ---------------------------------------------------------------------------

/// Read-only context provided to operators during execution.
///
/// Operators use this to gather information (invoke sub-sessions via the
/// runtime facade, check cancellation state, access memory) but do **not**
/// control the execution flow. The operator's output is always a new
/// `TaskState` — side effects are mediated by the runtime.
pub struct OperatorContext<'a> {
    /// The runtime invocation facade — operators use this to access providers,
    /// tools, and run sub-sessions.
    pub invocation: Option<&'a RuntimeInvocation>,
    /// Cooperative cancellation/control handle.
    pub control: Option<&'a Control>,
    /// A memory store for operators that need to read/write memory.
    pub memory: Option<&'a dyn MemoryStore>,
    /// An LLM provider for generating content during operator execution.
    pub llm: Option<&'a dyn LlmProvider>,
}

/// Minimal memory store interface for operator context.
///
/// This is a read-oriented view — operators can query memory but
/// write-back is mediated by the runtime.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Retrieves a value by key.
    async fn get(&self, key: &str) -> Option<String>;

    /// Lists all keys matching a prefix.
    async fn keys(&self, prefix: &str) -> Vec<String>;
}

// ---------------------------------------------------------------------------
// ReasoningOperator
// ---------------------------------------------------------------------------

/// An atomic reasoning operator.
///
/// Operators are the building blocks of reasoning strategies. Each operator
/// implements a single, well-defined transformation on a `TaskState`.
///
/// # Provider priority
///
/// If an operator has its own provider (via `provider` field),
/// it should use that. Otherwise fall back to [`OperatorContext::llm`].
///
/// # Prompt
///
/// Each operator has a mutable prompt template. The prompt can be customized
/// via [`set_prompt`](ReasoningOperator::set_prompt) before execution.
///
/// # Design contract
///
/// - Operators MUST be **idempotent with respect to side effects**: calling
///   `apply` with the same inputs should produce the same state transformation.
/// - Operators MUST NOT mutate external state directly. If an action is needed,
///   encode it as an [`Action`](crate::reasoning::state::Action) in the output state.
/// - Operators SHOULD be small and composable. Complex reasoning patterns
///   are built by composing operators in a `ReasoningGraph`.
#[async_trait]
pub trait ReasoningOperator: Send + Sync {
    /// Returns the kind of this operator.
    fn kind(&self) -> OperatorKind;

    /// Returns a human-readable name for this operator (for logging/debugging).
    fn name(&self) -> &'static str;

    /// Returns the prompt template for this operator.
    fn prompt(&self) -> &str;

    /// Replaces the prompt template.
    fn set_prompt(&mut self, prompt: String);

    /// Applies this operator to the given state, producing a new state.
    ///
    /// # Errors
    ///
    /// Returns [`ReasoningError`] if the operator cannot produce a valid
    /// next state from the given input.
    async fn apply(
        &self,
        ctx: &OperatorContext<'_>,
        state: TaskState,
    ) -> Result<TaskState, ReasoningError>;
}

// ---------------------------------------------------------------------------
// BaseOperator
// ---------------------------------------------------------------------------

/// Shared state for all built-in operators.
///
/// Provides default implementations for prompt, provider, and name storage.
/// Concrete operator structs embed a `BaseOperator` and delegate trait methods.
pub struct BaseOperator {
    /// The operator kind discriminator.
    pub kind: OperatorKind,
    /// Human-readable operator name.
    pub name: &'static str,
    /// Mutable prompt template.
    pub prompt: String,
}

impl BaseOperator {
    /// Creates a new `BaseOperator` with the given kind and name.
    #[must_use]
    pub fn new(kind: OperatorKind, name: &'static str) -> Self {
        Self {
            kind,
            name,
            prompt: String::new(),
        }
    }

    /// Sets the prompt template, consuming self.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }
}

// ---------------------------------------------------------------------------
// Formatting & LLM call helpers
// ---------------------------------------------------------------------------

/// Formats the current reasoning state as a structured prompt string.
pub(crate) fn format_state(prompt: &str, state: &TaskState) -> String {
    let mut buf = String::new();

    let _ = writeln!(buf, "Goal: {}", state.goal.description);
    if !state.constraints.is_empty() {
        let _ = writeln!(buf);
        let _ = writeln!(buf, "Constraints:");
        for c in &state.constraints {
            let _ = writeln!(buf, "- {} ({:?})", c.description, c.kind);
        }
    }
    let _ = writeln!(buf);
    let _ = writeln!(buf, "Context: {}", state.context.input);
    if !state.context.facts.is_empty() {
        let _ = writeln!(buf, "Facts:");
        for f in &state.context.facts {
            let _ = writeln!(buf, "- {f}");
        }
    }
    if let Some(ref plan) = state.plan {
        let _ = writeln!(buf);
        let _ = writeln!(buf, "Plan: {}", plan.rationale);
        for step in &plan.steps {
            let _ = writeln!(
                buf,
                "- Step {}: {} ({:?})",
                step.id, step.description, step.status
            );
        }
    }
    if !state.observations.is_empty() {
        let _ = writeln!(buf);
        let _ = writeln!(buf, "Observations:");
        for o in &state.observations {
            let _ = writeln!(
                buf,
                "- [{}] {} (from {:?})",
                o.sequence, o.content, o.source
            );
        }
    }
    if !state.actions.is_empty() {
        let _ = writeln!(buf);
        let _ = writeln!(buf, "Actions:");
        for a in &state.actions {
            let tool_str = a.tool.as_deref().unwrap_or("reasoning");
            let _ = writeln!(buf, "- [{}] {} -> {:?}", tool_str, a.input, a.output);
        }
    }
    if !state.reflections.is_empty() {
        let _ = writeln!(buf);
        let _ = writeln!(buf, "Reflections:");
        for r in &state.reflections {
            let _ = writeln!(buf, "- {}", r.content);
        }
    }
    if !state.artifacts.is_empty() {
        let _ = writeln!(buf);
        let _ = writeln!(buf, "Artifacts:");
        for a in &state.artifacts {
            let _ = writeln!(buf, "- [{:?}] {}", a.kind, a.content);
        }
    }
    let _ = writeln!(buf);
    let _ = write!(buf, "---\n{prompt}");

    buf
}

/// Calls the LLM via [`OperatorContext`] and records a [`Step`].
///
/// Returns the LLM response text and the new state with the step recorded.
///
/// # Errors
///
/// Returns [`ReasoningError::OperatorFailed`] if no LLM is available or the
/// underlying provider fails.
pub(crate) async fn call_llm_and_record_step(
    ctx: &OperatorContext<'_>,
    kind: OperatorKind,
    name: &str,
    prompt: &str,
    state: TaskState,
    system: &str,
) -> Result<(String, TaskState), ReasoningError> {
    let llm = ctx.llm.ok_or_else(|| ReasoningError::OperatorFailed {
        operator: name.into(),
        message: "no LLM provider configured in operator context".into(),
    })?;
    let user = format_state(prompt, &state);
    let response =
        llm.complete(&user, Some(system))
            .await
            .map_err(|e| ReasoningError::OperatorFailed {
                operator: name.into(),
                message: e.to_string(),
            })?;
    let new_state = state.record_step(Step {
        operator: kind,
        input: user,
        output: Some(response.clone()),
        status: StepStatus::Completed,
    });
    Ok((response, new_state))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn operator_kind_is_terminal() {
        assert!(OperatorKind::Answer.is_terminal());
        assert!(!OperatorKind::Reason.is_terminal());
        assert!(!OperatorKind::Act.is_terminal());
        assert!(!OperatorKind::Explain.is_terminal());
        assert!(!OperatorKind::Evaluate.is_terminal());
        assert!(!OperatorKind::Generate.is_terminal());
    }

    #[test]
    fn operator_kind_requires_tools() {
        assert!(OperatorKind::Act.requires_tools());
        assert!(OperatorKind::Observe.requires_tools());
        assert!(OperatorKind::Search.requires_tools());
        assert!(OperatorKind::Test.requires_tools());
        assert!(OperatorKind::Simulate.requires_tools());
        assert!(!OperatorKind::Reason.requires_tools());
        assert!(!OperatorKind::Plan.requires_tools());
        assert!(!OperatorKind::Explain.requires_tools());
        assert!(!OperatorKind::Evaluate.requires_tools());
    }

    #[test]
    fn operator_kind_should_serialize() {
        let kind = OperatorKind::Analyze;
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, "\"Analyze\"");
        let deserialized: OperatorKind = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, OperatorKind::Analyze);
    }

    #[test]
    fn all_operator_kinds_should_serialize_roundtrip() {
        let kinds = [
            OperatorKind::Analyze,
            OperatorKind::Decompose,
            OperatorKind::Plan,
            OperatorKind::Reason,
            OperatorKind::Act,
            OperatorKind::Observe,
            OperatorKind::Solve,
            OperatorKind::Verify,
            OperatorKind::Reflect,
            OperatorKind::Revise,
            OperatorKind::Select,
            OperatorKind::Compose,
            OperatorKind::Reduce,
            OperatorKind::Explain,
            OperatorKind::Evaluate,
            OperatorKind::Generate,
            OperatorKind::Hypothesize,
            OperatorKind::Test,
            OperatorKind::Search,
            OperatorKind::Synthesize,
            OperatorKind::Prioritize,
            OperatorKind::Optimize,
            OperatorKind::Simulate,
            OperatorKind::Diagnose,
            OperatorKind::Answer,
        ];

        for kind in &kinds {
            let json = serde_json::to_string(kind).unwrap();
            let deserialized: OperatorKind = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, *kind);
        }
    }

    #[test]
    fn base_operator_new() {
        let base = BaseOperator::new(OperatorKind::Analyze, "Analyze");
        assert_eq!(base.kind, OperatorKind::Analyze);
        assert_eq!(base.name, "Analyze");
        assert!(base.prompt.is_empty());
    }
}
