//! Strongly typed task state for reasoning graphs.
//!
//! [`TaskState`] is the central data structure flowing through a reasoning
//! graph. Each operator receives a `TaskState` and produces a new one,
//! accumulating observations, actions, reflections, and artifacts.

use serde::{Deserialize, Serialize};

use crate::reasoning::operator::OperatorKind;

// ---------------------------------------------------------------------------
// TaskState
// ---------------------------------------------------------------------------

/// Complete state of a reasoning task.
///
/// A `TaskState` is **immutable by convention** — operators produce new
/// instances rather than mutating existing ones. The `transition` builder
/// facilitates ergonomic state derivation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskState {
    /// The goal to achieve.
    pub goal: Goal,
    /// Constraints that bound the solution space.
    pub constraints: Vec<Constraint>,
    /// Known context, facts, and raw input.
    pub context: Context,
    /// Current plan, if one has been produced.
    pub plan: Option<Plan>,
    /// Steps recorded during reasoning.
    pub steps: Vec<Step>,
    /// Observations gathered from tools or the environment.
    pub observations: Vec<Observation>,
    /// Actions taken during reasoning.
    pub actions: Vec<Action>,
    /// Reflections produced by the agent.
    pub reflections: Vec<Reflection>,
    /// Structured artifacts produced during reasoning.
    pub artifacts: Vec<Artifact>,
    /// Metadata about the current reasoning session.
    pub metadata: StateMetadata,
}

impl TaskState {
    /// Creates a new task state with the given goal and input.
    #[must_use]
    pub fn new(goal: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            goal: Goal::new(goal),
            constraints: Vec::new(),
            context: Context::new(input),
            plan: None,
            steps: Vec::new(),
            observations: Vec::new(),
            actions: Vec::new(),
            reflections: Vec::new(),
            artifacts: Vec::new(),
            metadata: StateMetadata::default(),
        }
    }

    /// Adds a constraint to this state.
    #[must_use]
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Adds a fact to the context.
    #[must_use]
    pub fn with_fact(mut self, fact: impl Into<String>) -> Self {
        self.context.facts.push(fact.into());
        self
    }

    /// Sets the domain hint on the context.
    #[must_use]
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.context.domain = Some(domain.into());
        self
    }

    /// Derives a new state by recording a step.
    #[must_use]
    pub fn record_step(mut self, step: Step) -> Self {
        self.steps.push(step);
        self
    }

    /// Derives a new state by recording an observation.
    #[must_use]
    pub fn observe(mut self, observation: Observation) -> Self {
        self.observations.push(observation);
        self
    }

    /// Derives a new state by recording an action.
    #[must_use]
    pub fn record_action(mut self, action: Action) -> Self {
        self.actions.push(action);
        self
    }

    /// Derives a new state by recording a reflection.
    #[must_use]
    pub fn reflect(mut self, reflection: Reflection) -> Self {
        self.reflections.push(reflection);
        self
    }

    /// Derives a new state by adding an artifact.
    #[must_use]
    pub fn add_artifact(mut self, artifact: Artifact) -> Self {
        self.artifacts.push(artifact);
        self
    }

    /// Derives a new state with an updated plan.
    #[must_use]
    pub fn with_plan(mut self, plan: Plan) -> Self {
        self.plan = Some(plan);
        self
    }

    /// Derives a new state with the iteration counter incremented.
    #[must_use]
    pub fn next_iteration(mut self) -> Self {
        self.metadata.iteration += 1;
        self
    }

    /// Returns `true` if the goal appears to be achieved based on artifacts
    /// and step outcomes.
    #[must_use]
    pub fn is_goal_achieved(&self) -> bool {
        self.artifacts
            .iter()
            .any(|a| matches!(a.kind, ArtifactKind::Answer))
    }
}

// ---------------------------------------------------------------------------
// Goal
// ---------------------------------------------------------------------------

/// The goal to be achieved by the reasoning process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// Natural language description of the goal.
    pub description: String,
    /// Measurable criteria for success.
    pub success_criteria: Vec<String>,
}

impl Goal {
    /// Creates a new goal with the given description.
    #[must_use]
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            success_criteria: Vec::new(),
        }
    }

    /// Adds a success criterion.
    #[must_use]
    pub fn with_criterion(mut self, criterion: impl Into<String>) -> Self {
        self.success_criteria.push(criterion.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Constraint
// ---------------------------------------------------------------------------

/// A constraint bounding the solution space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// Whether this constraint is hard (must satisfy) or soft (prefer to satisfy).
    pub kind: ConstraintKind,
    /// Natural language description of the constraint.
    pub description: String,
}

/// Classification of constraint strictness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// Must be satisfied — violation is a failure.
    Hard,
    /// Should be satisfied — violation is suboptimal.
    Soft,
    /// A preference, not a requirement.
    Preference,
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Known context for the reasoning task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    /// Established facts relevant to the task.
    pub facts: Vec<String>,
    /// Optional domain hint (e.g., "mathematics", "code_generation").
    pub domain: Option<String>,
    /// The raw input text from the user or caller.
    pub input: String,
}

impl Context {
    /// Creates a new context with the given input.
    #[must_use]
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            facts: Vec::new(),
            domain: None,
            input: input.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// A plan decomposing the goal into ordered steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Ordered steps in the plan.
    pub steps: Vec<PlanStep>,
    /// Explanation of why this plan was chosen.
    pub rationale: String,
}

impl Plan {
    /// Creates a new plan with the given rationale.
    #[must_use]
    pub fn new(rationale: impl Into<String>) -> Self {
        Self {
            steps: Vec::new(),
            rationale: rationale.into(),
        }
    }

    /// Adds a step to the plan.
    #[must_use]
    pub fn with_step(mut self, description: impl Into<String>) -> Self {
        let id = self.steps.len();
        self.steps.push(PlanStep {
            id,
            description: description.into(),
            status: StepStatus::Pending,
        });
        self
    }
}

/// A single step within a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Zero-based index of this step.
    pub id: usize,
    /// What this step should accomplish.
    pub description: String,
    /// Current execution status.
    pub status: StepStatus,
}

// ---------------------------------------------------------------------------
// Step
// ---------------------------------------------------------------------------

/// A recorded reasoning step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    /// The operator that produced this step.
    pub operator: OperatorKind,
    /// Input that was fed to the operator.
    pub input: String,
    /// Output produced by the operator, if any.
    pub output: Option<String>,
    /// Whether the step succeeded, failed, etc.
    pub status: StepStatus,
}

/// Execution status of a step or plan step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StepStatus {
    /// Not yet started.
    Pending,
    /// Currently executing.
    InProgress,
    /// Completed successfully.
    Completed,
    /// Failed with an error.
    Failed,
    /// Skipped (e.g., conditional branch not taken).
    Skipped,
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// An observation gathered during reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Where the observation came from.
    pub source: ObservationSource,
    /// The observed content.
    pub content: String,
    /// Monotonic sequence number (not wall-clock time).
    pub sequence: u64,
}

/// Source of an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObservationSource {
    /// Returned by a tool invocation.
    Tool,
    /// Gathered from the environment (e.g., file system, API).
    Environment,
    /// Produced by self-inspection or reflection.
    SelfInspection,
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// An action taken during reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// Name of the tool invoked, if any.
    pub tool: Option<String>,
    /// Input sent to the tool or action description.
    pub input: String,
    /// Output received, if any.
    pub output: Option<String>,
}

impl Action {
    /// Creates an action for a tool invocation.
    #[must_use]
    pub fn tool_call(tool: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            tool: Some(tool.into()),
            input: input.into(),
            output: None,
        }
    }

    /// Creates a reasoning-only action (no tool).
    #[must_use]
    pub fn reasoning(input: impl Into<String>) -> Self {
        Self {
            tool: None,
            input: input.into(),
            output: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Reflection
// ---------------------------------------------------------------------------

/// A reflection produced during the reasoning process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reflection {
    /// The reflective content — what went wrong, what to try next, etc.
    pub content: String,
    /// An optional revised plan based on this reflection.
    pub revised_plan: Option<Plan>,
}

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

/// A structured artifact produced during reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Classification of the artifact.
    pub kind: ArtifactKind,
    /// The artifact content.
    pub content: String,
}

/// Type of artifact produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArtifactKind {
    /// Source code.
    Code,
    /// Natural language text.
    Text,
    /// Structured data (JSON, YAML, etc.).
    Data,
    /// A file path or reference.
    File,
    /// An image reference.
    Image,
    /// Final answer to the goal.
    Answer,
}

// ---------------------------------------------------------------------------
// StateMetadata
// ---------------------------------------------------------------------------

/// Metadata about the reasoning session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateMetadata {
    /// Current iteration count in the reasoning loop.
    pub iteration: usize,
    /// Sequence counter for observations (monotonic).
    pub observation_sequence: u64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn task_state_new_should_set_goal_and_input() {
        let state = TaskState::new("solve equation", "x + 2 = 5");
        assert_eq!(state.goal.description, "solve equation");
        assert_eq!(state.context.input, "x + 2 = 5");
        assert!(state.constraints.is_empty());
        assert!(state.plan.is_none());
    }

    #[test]
    fn task_state_builder_chain_should_work() {
        let state = TaskState::new("goal", "input")
            .with_constraint(Constraint {
                kind: ConstraintKind::Hard,
                description: "must be numeric".into(),
            })
            .with_fact("x is unknown")
            .with_domain("mathematics")
            .with_plan(Plan::new("algebraic isolation").with_step("subtract 2"));

        assert_eq!(state.constraints.len(), 1);
        assert_eq!(state.context.facts.len(), 1);
        assert_eq!(state.context.domain.as_deref(), Some("mathematics"));
        let plan = state.plan.unwrap();
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].status, StepStatus::Pending);
    }

    #[test]
    fn task_state_record_step_should_accumulate() {
        let state = TaskState::new("g", "i").record_step(Step {
            operator: OperatorKind::Analyze,
            input: "what is this".into(),
            output: Some("a problem".into()),
            status: StepStatus::Completed,
        });

        assert_eq!(state.steps.len(), 1);
        assert_eq!(state.steps[0].operator, OperatorKind::Analyze);
    }

    #[test]
    fn task_state_is_goal_achieved_without_answer_artifact() {
        let state = TaskState::new("g", "i");
        assert!(!state.is_goal_achieved());
    }

    #[test]
    fn task_state_is_goal_achieved_with_answer_artifact() {
        let state = TaskState::new("g", "i").add_artifact(Artifact {
            kind: ArtifactKind::Answer,
            content: "42".into(),
        });
        assert!(state.is_goal_achieved());
    }

    #[test]
    fn task_state_next_iteration_should_increment() {
        let state = TaskState::new("g", "i").next_iteration().next_iteration();
        assert_eq!(state.metadata.iteration, 2);
    }

    #[test]
    fn observe_should_increment_sequence() {
        let mut state = TaskState::new("g", "i");
        state.metadata.observation_sequence = 0;
        let seq = state.metadata.observation_sequence + 1;
        let state = state.observe(Observation {
            source: ObservationSource::Tool,
            content: "result".into(),
            sequence: seq,
        });
        assert_eq!(state.observations.len(), 1);
        assert_eq!(state.observations[0].sequence, 1);
    }

    #[test]
    fn action_tool_call_and_reasoning() {
        let tool_action = Action::tool_call("calculator", "2+2");
        assert_eq!(tool_action.tool.as_deref(), Some("calculator"));
        assert!(tool_action.output.is_none());

        let reason_action = Action::reasoning("thinking...");
        assert!(reason_action.tool.is_none());
    }

    #[test]
    fn plan_with_step_should_auto_increment_ids() {
        let plan = Plan::new("rationale")
            .with_step("first")
            .with_step("second")
            .with_step("third");

        assert_eq!(plan.steps[0].id, 0);
        assert_eq!(plan.steps[1].id, 1);
        assert_eq!(plan.steps[2].id, 2);
    }

    #[test]
    fn task_state_should_be_serializable() {
        let state = TaskState::new("solve", "x = 1")
            .with_fact("x is positive")
            .add_artifact(Artifact {
                kind: ArtifactKind::Answer,
                content: "x = 1".into(),
            });

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: TaskState = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.goal.description, "solve");
        assert_eq!(deserialized.artifacts.len(), 1);
    }
}
