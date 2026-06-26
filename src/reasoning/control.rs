//! Control flow primitives for reasoning graphs.
//!
//! [`ControlKind`] defines the edges in a reasoning graph — how operators
//! connect, whether they run in parallel, loop, or branch.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ControlKind
// ---------------------------------------------------------------------------

/// Type of control flow between operators in a reasoning graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ControlKind {
    /// Sequential execution — run the next operator after the current one completes.
    Pipeline,
    /// Conditional branching based on a predicate.
    Branch {
        /// The condition that determines which branch to take.
        condition: BranchCondition,
    },
    /// Loop back to a previous operator.
    Loop {
        /// Maximum number of iterations before forced exit.
        max_iterations: usize,
    },
    /// Fan-out: spawn parallel executions.
    FanOut {
        /// Maximum degree of parallelism.
        parallelism: usize,
    },
    /// Fan-in: merge parallel results.
    FanIn {
        /// Strategy for merging multiple results into one.
        strategy: MergeStrategy,
    },
    /// Race: take the first result from parallel executions.
    Race,
    /// Retry: re-execute the target operator on failure.
    Retry {
        /// Maximum number of retry attempts.
        max_attempts: usize,
    },
    /// Execute a nested sub-graph.
    Subgraph,
}

/// Condition that determines which branch to take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BranchCondition {
    /// Branch if the previous operator succeeded.
    OnSuccess,
    /// Branch if the previous operator failed.
    OnFailure,
    /// Branch if the goal is achieved.
    OnGoalAchieved,
    /// Branch if the goal is NOT achieved.
    OnGoalNotAchieved,
}

/// Strategy for merging parallel results at a fan-in point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Take the first successful result.
    First,
    /// Collect all results.
    All,
    /// Take the best result (requires a scoring function in the runtime).
    Best,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn control_kind_should_serialize_roundtrip() {
        let kinds = [
            ControlKind::Pipeline,
            ControlKind::Branch {
                condition: BranchCondition::OnSuccess,
            },
            ControlKind::Loop { max_iterations: 10 },
            ControlKind::FanOut { parallelism: 4 },
            ControlKind::FanIn {
                strategy: MergeStrategy::All,
            },
            ControlKind::Race,
            ControlKind::Retry { max_attempts: 3 },
            ControlKind::Subgraph,
        ];

        for kind in &kinds {
            let json = serde_json::to_string(kind).unwrap();
            let deserialized: ControlKind = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, *kind);
        }
    }

    #[test]
    fn branch_condition_variants() {
        assert_ne!(BranchCondition::OnSuccess, BranchCondition::OnFailure);
        assert_ne!(
            BranchCondition::OnGoalAchieved,
            BranchCondition::OnGoalNotAchieved
        );
    }

    #[test]
    fn merge_strategy_variants() {
        assert_ne!(MergeStrategy::First, MergeStrategy::All);
        assert_ne!(MergeStrategy::All, MergeStrategy::Best);
    }
}
