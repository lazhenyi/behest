//! Reasoning graph — a directed acyclic graph of operators connected by control flow.
//!
//! A [`ReasoningGraph`] is the compiled representation of a reasoning strategy.
//! It consists of [`GraphNode`]s (operators or sub-graphs) connected by
//! [`ControlEdge`]s that define the execution order and branching logic.

use std::sync::Arc;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Topo;

use crate::reasoning::control::ControlKind;
use crate::reasoning::error::{ReasoningError, ReasoningResult};
use crate::reasoning::operator::{OperatorKind, ReasoningOperator};
use crate::reasoning::state::TaskState;

// ---------------------------------------------------------------------------
// ReasoningGraph
// ---------------------------------------------------------------------------

/// A directed graph of reasoning operators connected by control flow edges.
///
/// The graph is the central data structure produced by a reasoning strategy.
/// It can be traversed by a graph scheduler that executes operators in
/// topological order, respecting control flow semantics.
///
/// The graph is always a **DAG** (directed acyclic graph). Iterative behaviour
/// (e.g., the ReAct loop) is expressed via [`loop_start`](Self::loop_start) — the runtime
/// re-executes from that node until the [`TerminationPolicy`] triggers.
pub struct ReasoningGraph {
    pub(crate) graph: DiGraph<GraphNode, ControlEdge>,
    pub(crate) termination: TerminationPolicy,
    /// Human-readable name for this graph (e.g., "ReAct", "PlanAndSolve").
    pub name: String,
    /// If set, the runtime re-executes from this node until the termination
    /// policy triggers. This is how iterative strategies (ReAct, Reflection,
    /// etc.) express loops without creating cycles in the graph.
    pub(crate) loop_start: Option<NodeIndex>,
}

impl ReasoningGraph {
    /// Creates a new empty reasoning graph with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>, termination: TerminationPolicy) -> Self {
        Self {
            graph: DiGraph::new(),
            termination,
            name: name.into(),
            loop_start: None,
        }
    }

    /// Sets the loop-start node — the runtime re-executes from this node
    /// until the termination policy triggers.
    pub fn set_loop_start(&mut self, idx: NodeIndex) {
        self.loop_start = Some(idx);
    }

    /// Returns the loop-start node, if one was set.
    #[must_use]
    pub fn loop_start(&self) -> Option<NodeIndex> {
        self.loop_start
    }

    /// Adds an operator node to the graph and returns its index.
    #[must_use]
    pub fn add_operator(&mut self, op: Arc<dyn ReasoningOperator>) -> NodeIndex {
        self.graph.add_node(GraphNode::Operator(op))
    }

    /// Adds a sub-graph node to the graph and returns its index.
    #[must_use]
    pub fn add_subgraph(&mut self, sub: ReasoningGraph) -> NodeIndex {
        self.graph.add_node(GraphNode::Subgraph(sub))
    }

    /// Connects two nodes with the given control flow kind.
    pub fn connect(&mut self, from: NodeIndex, to: NodeIndex, kind: ControlKind) {
        self.graph.add_edge(from, to, ControlEdge { kind });
    }

    /// Returns the number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns the number of edges in the graph.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Returns a reference to the node at the given index.
    #[must_use]
    pub fn node(&self, idx: NodeIndex) -> Option<&GraphNode> {
        self.graph.node_weight(idx)
    }

    /// Returns the termination policy for this graph.
    #[must_use]
    pub fn termination(&self) -> &TerminationPolicy {
        &self.termination
    }

    /// Validates the graph structure.
    ///
    /// Checks:
    /// - At least one node exists.
    /// - At least one terminal node (Answer operator) exists.
    /// - No cycles exist (the graph is a DAG).
    ///
    /// # Errors
    ///
    /// Returns [`ReasoningError::GraphValidation`] if structural problems are found.
    pub fn validate(&self) -> ReasoningResult<()> {
        if self.graph.node_count() == 0 {
            return Err(ReasoningError::GraphValidation {
                message: "graph has no nodes".into(),
            });
        }

        let has_terminal = self.graph.node_weights().any(|node| match node {
            GraphNode::Operator(op) => op.kind().is_terminal(),
            GraphNode::Subgraph(_) => false,
        });

        if !has_terminal {
            return Err(ReasoningError::GraphValidation {
                message: "graph has no terminal (Answer) operator".into(),
            });
        }

        // petgraph's is_cyclic_directed detects cycles in O(V+E).
        if petgraph::algo::is_cyclic_directed(&self.graph) {
            return Err(ReasoningError::CycleDetected);
        }

        Ok(())
    }

    /// Returns nodes in topological order.
    ///
    /// # Errors
    ///
    /// Returns [`ReasoningError::CycleDetected`] if the graph contains a cycle.
    pub fn topological_order(&self) -> ReasoningResult<Vec<NodeIndex>> {
        let mut order = Vec::with_capacity(self.graph.node_count());
        let mut topo = Topo::new(&self.graph);
        while let Some(node) = topo.next(&self.graph) {
            order.push(node);
        }

        if order.len() != self.graph.node_count() {
            return Err(ReasoningError::CycleDetected);
        }

        Ok(order)
    }

    /// Collects all operator kinds used in this graph.
    #[must_use]
    pub fn operator_kinds(&self) -> Vec<OperatorKind> {
        self.graph
            .node_weights()
            .map(|node| match node {
                GraphNode::Operator(op) => op.kind(),
                GraphNode::Subgraph(sub) => {
                    let mut kinds = sub.operator_kinds();
                    kinds.remove(0)
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// GraphNode
// ---------------------------------------------------------------------------

/// A node in a reasoning graph — either an operator or a nested sub-graph.
pub enum GraphNode {
    /// An atomic reasoning operator.
    Operator(Arc<dyn ReasoningOperator>),
    /// A nested reasoning graph.
    Subgraph(ReasoningGraph),
}

// ---------------------------------------------------------------------------
// ControlEdge
// ---------------------------------------------------------------------------

/// An edge in a reasoning graph connecting two nodes.
#[derive(Debug, Clone)]
pub struct ControlEdge {
    /// The control flow semantics of this edge.
    pub kind: ControlKind,
}

// ---------------------------------------------------------------------------
// TerminationPolicy
// ---------------------------------------------------------------------------

/// Policy controlling when a reasoning graph should stop executing.
#[derive(Debug, Clone)]
pub struct TerminationPolicy {
    /// Maximum number of iterations through the graph.
    pub max_iterations: usize,
    /// Whether to stop when the goal is achieved (detected via `TaskState::is_goal_achieved`).
    pub on_goal_achieved: bool,
}

impl TerminationPolicy {
    /// Creates a termination policy with the given max iterations.
    #[must_use]
    pub const fn with_max_iterations(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            on_goal_achieved: true,
        }
    }

    /// Creates a termination policy that stops only on max iterations.
    #[must_use]
    pub const fn fixed(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            on_goal_achieved: false,
        }
    }

    /// Checks whether execution should terminate given the current state.
    #[must_use]
    pub fn should_terminate(&self, state: &TaskState) -> bool {
        if state.metadata.iteration >= self.max_iterations {
            return true;
        }
        self.on_goal_achieved && state.is_goal_achieved()
    }
}

impl Default for TerminationPolicy {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            on_goal_achieved: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use async_trait::async_trait;

    use super::*;
    use crate::reasoning::error::ReasoningError;
    use crate::reasoning::operator::OperatorContext;
    use crate::reasoning::state::TaskState;

    struct StubOperator {
        kind: OperatorKind,
        name: &'static str,
        prompt: String,
    }

    impl StubOperator {
        fn new(kind: OperatorKind) -> Self {
            let name = match kind {
                OperatorKind::Analyze => "Analyze",
                OperatorKind::Decompose => "Decompose",
                OperatorKind::Plan => "Plan",
                OperatorKind::Reason => "Reason",
                OperatorKind::Act => "Act",
                OperatorKind::Observe => "Observe",
                OperatorKind::Solve => "Solve",
                OperatorKind::Verify => "Verify",
                OperatorKind::Reflect => "Reflect",
                OperatorKind::Revise => "Revise",
                OperatorKind::Select => "Select",
                OperatorKind::Compose => "Compose",
                OperatorKind::Reduce => "Reduce",
                OperatorKind::Explain => "Explain",
                OperatorKind::Evaluate => "Evaluate",
                OperatorKind::Generate => "Generate",
                OperatorKind::Hypothesize => "Hypothesize",
                OperatorKind::Test => "Test",
                OperatorKind::Search => "Search",
                OperatorKind::Synthesize => "Synthesize",
                OperatorKind::Prioritize => "Prioritize",
                OperatorKind::Optimize => "Optimize",
                OperatorKind::Simulate => "Simulate",
                OperatorKind::Diagnose => "Diagnose",
                OperatorKind::Answer => "Answer",
            };
            Self {
                kind,
                name,
                prompt: String::new(),
            }
        }
    }

    #[async_trait]
    impl ReasoningOperator for StubOperator {
        fn kind(&self) -> OperatorKind {
            self.kind
        }

        fn name(&self) -> &'static str {
            self.name
        }

        fn prompt(&self) -> &str {
            &self.prompt
        }

        fn set_prompt(&mut self, prompt: String) {
            self.prompt = prompt;
        }

        async fn apply(
            &self,
            _ctx: &OperatorContext<'_>,
            state: TaskState,
        ) -> Result<TaskState, ReasoningError> {
            Ok(state)
        }
    }

    #[test]
    fn graph_add_operator_and_connect() {
        let mut graph = ReasoningGraph::new("test", TerminationPolicy::default());
        let reason = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Reason)));
        let act = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Act)));
        let observe = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Observe)));
        let answer = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Answer)));

        graph.connect(reason, act, ControlKind::Pipeline);
        graph.connect(act, observe, ControlKind::Pipeline);
        graph.connect(observe, answer, ControlKind::Pipeline);

        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn graph_validate_empty_graph_fails() {
        let graph = ReasoningGraph::new("empty", TerminationPolicy::default());
        assert!(graph.validate().is_err());
    }

    #[test]
    fn graph_validate_no_terminal_fails() {
        let mut graph = ReasoningGraph::new("no-terminal", TerminationPolicy::default());
        let _ = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Reason)));
        assert!(graph.validate().is_err());
    }

    #[test]
    fn graph_validate_valid_graph_succeeds() {
        let mut graph = ReasoningGraph::new("valid", TerminationPolicy::default());
        let r = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Reason)));
        let a = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Answer)));
        graph.connect(r, a, ControlKind::Pipeline);

        assert!(graph.validate().is_ok());
    }

    #[test]
    fn graph_topological_order() {
        let mut graph = ReasoningGraph::new("topo", TerminationPolicy::default());
        let a = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Analyze)));
        let p = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Plan)));
        let s = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Solve)));
        let ans = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Answer)));

        graph.connect(a, p, ControlKind::Pipeline);
        graph.connect(p, s, ControlKind::Pipeline);
        graph.connect(s, ans, ControlKind::Pipeline);

        let order = graph.topological_order().unwrap();
        assert_eq!(order.len(), 4);
        // a must come before p, p before s, s before ans
        let pos = |n: NodeIndex| order.iter().position(|&x| x == n).unwrap();
        assert!(pos(a) < pos(p));
        assert!(pos(p) < pos(s));
        assert!(pos(s) < pos(ans));
    }

    #[test]
    fn graph_operator_kinds() {
        let mut graph = ReasoningGraph::new("kinds", TerminationPolicy::default());
        let _ = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Reason)));
        let _ = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Act)));
        let _ = graph.add_operator(Arc::new(StubOperator::new(OperatorKind::Answer)));

        let kinds = graph.operator_kinds();
        assert!(kinds.contains(&OperatorKind::Reason));
        assert!(kinds.contains(&OperatorKind::Act));
        assert!(kinds.contains(&OperatorKind::Answer));
    }

    #[test]
    fn termination_policy_should_terminate_on_max_iterations() {
        let policy = TerminationPolicy::with_max_iterations(5);
        let mut state = TaskState::new("g", "i");
        state.metadata.iteration = 5;
        assert!(policy.should_terminate(&state));
    }

    #[test]
    fn termination_policy_should_terminate_on_goal_achieved() {
        let policy = TerminationPolicy::with_max_iterations(100);
        let state = TaskState::new("g", "i").add_artifact(crate::reasoning::state::Artifact {
            kind: crate::reasoning::state::ArtifactKind::Answer,
            content: "done".into(),
        });
        assert!(policy.should_terminate(&state));
    }

    #[test]
    fn termination_policy_should_not_terminate_early() {
        let policy = TerminationPolicy::with_max_iterations(10);
        let state = TaskState::new("g", "i");
        assert!(!policy.should_terminate(&state));
    }

    #[test]
    fn termination_policy_fixed_ignores_goal() {
        let policy = TerminationPolicy::fixed(10);
        let state = TaskState::new("g", "i").add_artifact(crate::reasoning::state::Artifact {
            kind: crate::reasoning::state::ArtifactKind::Answer,
            content: "done".into(),
        });
        // Fixed policy doesn't terminate on goal, only on max iterations
        assert!(!policy.should_terminate(&state));
    }
}
