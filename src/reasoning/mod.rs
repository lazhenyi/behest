//! Reasoning graph runtime — unified abstraction for agent reasoning strategies.
//!
//! This module provides a common intermediate representation (IR) for diverse
//! reasoning strategies as directed graphs of atomic operators connected by
//! control flow edges.
//!
//! # Architecture
//!
//! ```text
//! Operator (atomic state transformer) → ReasoningGraph (IR) → Graph Scheduler (executor)
//! ```
//!
//! - [`operator::ReasoningOperator`] — atomic state transformation.
//! - [`control::ControlKind`] — edge semantics (pipeline, branch, loop, fan-out/in).
//! - [`graph::ReasoningGraph`] — DAG of operators + control edges.
//!
//! # Design principles
//!
//! 1. Operators can optionally hold their own [`ChatProvider`](crate::provider::ChatProvider)
//!    and a mutable prompt template.
//! 2. Users can define custom operators via [`operator::CustomOperator`].
//! 3. The execution model is the **graph scheduler** (to be implemented in the
//!    runtime layer) which traverses the graph in topological order, respecting
//!    control flow semantics.

pub mod control;
pub mod error;
pub mod graph;
pub mod llm_provider;
pub mod operator;
pub mod scheduler;
pub mod state;

pub use self::control::{BranchCondition, ControlKind, MergeStrategy};
pub use self::error::{ReasoningError, ReasoningResult};
pub use self::graph::{ControlEdge, GraphNode, ReasoningGraph, TerminationPolicy};
pub use self::operator::{
    ActOperator, AnalyzeOperator, AnswerOperator, ComposeOperator, CustomOperator,
    DecomposeOperator, DiagnoseOperator, EvaluateOperator, ExplainOperator, GenerateOperator,
    HypothesizeOperator, LlmProvider, MemoryStore, ObserveOperator, OperatorContext, OperatorKind,
    OptimizeOperator, PlanOperator, PrioritizeOperator, ReasonOperator, ReasoningOperator,
    ReduceOperator, ReflectOperator, ReviseOperator, SearchOperator, SelectOperator,
    SimulateOperator, SolveOperator, SynthesizeOperator, TestOperator, VerifyOperator,
};
pub use self::state::{
    Action, Artifact, ArtifactKind, Constraint, ConstraintKind, Context, Goal, Observation,
    ObservationSource, Plan, PlanStep, Reflection, StateMetadata, Step, StepStatus, TaskState,
};
