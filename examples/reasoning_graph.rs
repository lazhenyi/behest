//! Demonstrates constructing a reasoning graph with operators and control edges.
//!
//! Run with:
//! ```bash
//! cargo run --example reasoning_graph
//! ```

use std::sync::Arc;

use behest::reasoning::operator::{AnswerOperator, PlanOperator, SolveOperator};
use behest::reasoning::{ControlKind, ReasoningGraph, TerminationPolicy};

fn main() {
    let mut graph = ReasoningGraph::new("plan-solve-answer", TerminationPolicy::default());

    let plan = graph.add_operator(Arc::new(PlanOperator::new()));
    let solve = graph.add_operator(Arc::new(SolveOperator::new()));
    let answer = graph.add_operator(Arc::new(AnswerOperator::new()));

    graph.connect(plan, solve, ControlKind::Pipeline);
    graph.connect(solve, answer, ControlKind::Pipeline);

    println!("Reasoning graph: \"{}\"", graph.name);
    println!("  nodes: {}", graph.node_count());
    println!("  edges: {}", graph.edge_count());

    println!("\nOperators:");
    let indices = [plan, solve, answer];
    let labels = ["Plan", "Solve", "Answer"];
    for (i, idx) in indices.iter().enumerate() {
        if let Some(node) = graph.node(*idx) {
            let kind = match node {
                behest::reasoning::GraphNode::Operator(op) => format!("{:?}", op.kind()),
                behest::reasoning::GraphNode::Subgraph(sg) => format!("subgraph: {}", sg.name),
            };
            println!("  [{}] {} — {kind}", idx.index(), labels[i]);
        }
    }

    println!("\nControl flow:");
    println!("  Plan ({}) --> Solve ({})", plan.index(), solve.index());
    println!(
        "  Solve ({}) --> Answer ({})",
        solve.index(),
        answer.index()
    );

    let term = graph.termination();
    println!(
        "\nTermination: max_iterations={}, on_goal_achieved={}",
        term.max_iterations, term.on_goal_achieved
    );

    match graph.validate() {
        Ok(()) => println!("Graph validation: OK"),
        Err(e) => println!("Graph validation failed: {e}"),
    }
}
