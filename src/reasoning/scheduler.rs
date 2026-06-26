//! Reasoning graph scheduler.

use crate::reasoning::control::{BranchCondition, ControlKind};
use crate::reasoning::error::{ReasoningError, ReasoningResult};
use crate::reasoning::graph::{GraphNode, ReasoningGraph};
use crate::reasoning::operator::{LlmProvider, OperatorContext};
use crate::reasoning::state::TaskState;

/// Executes a reasoning graph to completion.
///
/// Traverses the graph in topological order, executing each operator's `apply`
/// and respecting control flow semantics. Sub-graphs are executed recursively.
///
/// # Errors
///
/// Returns [`ReasoningError::MaxIterationsExceeded`] if the graph exceeds its
/// iteration limit. Returns [`ReasoningError::GraphValidation`] if a node
/// cannot be found. Passthrough from operator `apply()` failures.
pub async fn execute_graph(
    graph: &ReasoningGraph,
    llm: &dyn LlmProvider,
    mut state: TaskState,
) -> ReasoningResult<TaskState> {
    let mut iteration = 0usize;

    loop {
        if graph.termination().should_terminate(&state) {
            return Ok(state);
        }

        let nodes = graph.topological_order()?;

        for &node_idx in &nodes {
            let node = graph
                .node(node_idx)
                .ok_or_else(|| ReasoningError::GraphValidation {
                    message: "node not found during execution".into(),
                })?;

            if should_skip_node(graph, node_idx, &state) {
                continue;
            }

            match node {
                GraphNode::Operator(op) => {
                    let ctx = OperatorContext {
                        invocation: None,
                        control: None,
                        memory: None,
                        llm: Some(llm),
                    };
                    state = op.apply(&ctx, state).await?;
                }
                GraphNode::Subgraph(sub) => {
                    state = Box::pin(execute_graph(sub, llm, state)).await?;
                }
            }

            if graph.termination().should_terminate(&state) {
                return Ok(state);
            }
        }

        iteration += 1;
        if iteration >= graph.termination().max_iterations {
            return Err(ReasoningError::MaxIterationsExceeded {
                max: graph.termination().max_iterations,
            });
        }

        state = state.next_iteration();
    }
}

/// Determines whether a node should be skipped based on incoming branch edges.
fn should_skip_node(
    graph: &ReasoningGraph,
    node_idx: petgraph::graph::NodeIndex,
    state: &TaskState,
) -> bool {
    for edge in graph
        .graph
        .edges_directed(node_idx, petgraph::Direction::Incoming)
    {
        if let ControlKind::Branch { condition } = &edge.weight().kind {
            if condition_mismatch(*condition, state) {
                return true;
            }
        }
    }
    false
}

/// Returns `true` if the branch condition is NOT satisfied (so we skip).
fn condition_mismatch(condition: BranchCondition, state: &TaskState) -> bool {
    match condition {
        BranchCondition::OnSuccess | BranchCondition::OnFailure => false,
        BranchCondition::OnGoalAchieved => !state.is_goal_achieved(),
        BranchCondition::OnGoalNotAchieved => state.is_goal_achieved(),
    }
}
