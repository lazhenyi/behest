//! Turn-based state machine for agent control flow.
//!
//! Replaces flat loop `break`/`continue`/`return` with explicit
//! [`TurnState`] → [`TurnOutcome`] → [`TurnAction`] transitions
//! driven by a pure [`TurnTransition::resolve`] function.
//!
//! This makes the agent loop's control flow auditable, testable, and
//! extensible without touching the core executor.

use crate::provider::FinishReason;

use super::run::RunStatus;

use serde::{Deserialize, Serialize};

/// States in the agent turn cycle.
///
/// Each iteration of the agent loop moves through a subset of these states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnState {
    /// Validating iteration count and token budget before any work.
    CheckingPolicy,
    /// Running proactive compaction then building the chat context.
    BuildingContext,
    /// Calling the model provider (streaming or complete).
    CallingModel,
    /// Running reactive compaction after a context overflow.
    Compacting,
    /// Parsing the model response and deciding whether to call tools.
    ProcessingResponse,
    /// Executing tool calls returned by the model.
    ExecutingTools,
    /// Persisting tool results and preparing for the next turn.
    Persisting,
}

/// Outcome of a turn — what the state produced.
///
/// Drives the transition to the next state.
#[derive(Debug, Clone)]
pub enum TurnOutcome {
    /// State executed successfully.
    Success,
    /// Policy limit reached (iteration or token).
    PolicyExceeded {
        /// Human-readable reason.
        reason: String,
    },
    /// Provider returned a context overflow error.
    ContextOverflow,
    /// Provider returned a non-retryable error.
    ProviderError {
        /// Error message.
        message: String,
    },
    /// Context or storage error.
    PipelineError {
        /// Error message.
        message: String,
    },
    /// Model response has no tool calls — agent is done.
    NoToolCalls,
    /// Model finish reason is not ToolCalls — agent is done.
    NotToolCalls {
        /// The finish reason that stopped the loop.
        finish_reason: FinishReason,
    },
    /// Tool execution failed and policy does not allow continuation.
    ToolFailure {
        /// Error message.
        message: String,
    },
    /// Model output was truncated (FinishReason::Length). Recovery may be attempted.
    OutputTruncated,
}

/// Action the loop executor should take after a transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnAction {
    /// Proceed to the next state in the normal flow.
    Continue {
        /// Next state to enter.
        next: TurnState,
    },
    /// Break the loop and complete the run successfully.
    BreakLoop,
    /// Run compaction then retry CallingModel.
    CompactAndRetry,
    /// Terminate the run with an error.
    Fail {
        /// Error message.
        reason: String,
    },
}

/// Pure resolver mapping `(TurnState, TurnOutcome)` → `TurnAction`.
///
/// This function encodes the complete control-flow logic. It is
/// stateless and can be tested exhaustively without the runtime.
pub struct TurnTransition;

impl TurnTransition {
    /// Resolves the next action given the current state and outcome.
    #[must_use]
    pub fn resolve(state: TurnState, outcome: &TurnOutcome) -> TurnAction {
        match (state, outcome) {
            // ── CheckingPolicy ──────────────────────────────────────
            (TurnState::CheckingPolicy, TurnOutcome::PolicyExceeded { reason }) => {
                TurnAction::Fail {
                    reason: reason.clone(),
                }
            }
            (TurnState::CheckingPolicy, TurnOutcome::Success) => TurnAction::Continue {
                next: TurnState::BuildingContext,
            },

            // ── BuildingContext / Compacting ────────────────────────
            (TurnState::BuildingContext | TurnState::Compacting, TurnOutcome::Success)
            | (TurnState::ProcessingResponse, TurnOutcome::OutputTruncated) => {
                TurnAction::Continue {
                    next: TurnState::CallingModel,
                }
            }
            (
                TurnState::BuildingContext | TurnState::Compacting,
                TurnOutcome::PipelineError { .. },
            ) => TurnAction::Fail {
                reason: "context build failed".to_string(),
            },

            // ── CallingModel ────────────────────────────────────────
            (TurnState::CallingModel, TurnOutcome::Success) => TurnAction::Continue {
                next: TurnState::ProcessingResponse,
            },
            (TurnState::CallingModel, TurnOutcome::ContextOverflow) => TurnAction::CompactAndRetry,
            (TurnState::CallingModel, TurnOutcome::ProviderError { .. }) => TurnAction::Fail {
                reason: "provider error".to_string(),
            },

            // ── ProcessingResponse ──────────────────────────────────
            (
                TurnState::ProcessingResponse,
                TurnOutcome::NoToolCalls | TurnOutcome::NotToolCalls { .. },
            ) => TurnAction::BreakLoop,
            (TurnState::ProcessingResponse, TurnOutcome::Success) => TurnAction::Continue {
                next: TurnState::ExecutingTools,
            },

            // ── ExecutingTools ──────────────────────────────────────
            (TurnState::ExecutingTools, TurnOutcome::Success) => TurnAction::Continue {
                next: TurnState::Persisting,
            },
            (TurnState::ExecutingTools, TurnOutcome::ToolFailure { .. }) => TurnAction::Fail {
                reason: "tool execution failed".to_string(),
            },

            // ── Persisting ──────────────────────────────────────────
            (TurnState::Persisting, TurnOutcome::Success) => TurnAction::Continue {
                next: TurnState::CheckingPolicy,
            },
            (TurnState::Persisting, TurnOutcome::PipelineError { .. }) => TurnAction::Fail {
                reason: "persistence failed".to_string(),
            },

            // ── Fallback: unexpected combinations ────────────────────
            (_, outcome) => TurnAction::Fail {
                reason: format!("unexpected outcome {outcome:?} in state {state:?}"),
            },
        }
    }

    /// Derives the [`RunStatus`] implied by a given turn state.
    #[must_use]
    pub fn status_for(state: TurnState) -> RunStatus {
        match state {
            TurnState::CheckingPolicy | TurnState::Compacting | TurnState::BuildingContext => {
                RunStatus::BuildingContext
            }
            TurnState::CallingModel | TurnState::ProcessingResponse => RunStatus::CallingModel,
            TurnState::ExecutingTools => RunStatus::WaitingForTools,
            TurnState::Persisting => RunStatus::Persisting,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── CheckingPolicy ──────────────────────────────────────────────

    #[test]
    fn policy_exceeded_fails() {
        let action = TurnTransition::resolve(
            TurnState::CheckingPolicy,
            &TurnOutcome::PolicyExceeded {
                reason: "iteration".into(),
            },
        );
        assert_eq!(
            action,
            TurnAction::Fail {
                reason: "iteration".into()
            }
        );
    }

    #[test]
    fn policy_ok_proceeds_to_context() {
        let action = TurnTransition::resolve(TurnState::CheckingPolicy, &TurnOutcome::Success);
        assert_eq!(
            action,
            TurnAction::Continue {
                next: TurnState::BuildingContext
            }
        );
    }

    // ── CallingModel ────────────────────────────────────────────────

    #[test]
    fn context_overflow_compacts_and_retries() {
        let action =
            TurnTransition::resolve(TurnState::CallingModel, &TurnOutcome::ContextOverflow);
        assert_eq!(action, TurnAction::CompactAndRetry);
    }

    #[test]
    fn model_success_moves_to_processing() {
        let action = TurnTransition::resolve(TurnState::CallingModel, &TurnOutcome::Success);
        assert_eq!(
            action,
            TurnAction::Continue {
                next: TurnState::ProcessingResponse
            }
        );
    }

    #[test]
    fn provider_error_fails() {
        let action = TurnTransition::resolve(
            TurnState::CallingModel,
            &TurnOutcome::ProviderError {
                message: "boom".into(),
            },
        );
        assert_eq!(
            action,
            TurnAction::Fail {
                reason: "provider error".into()
            }
        );
    }

    // ── Compacting ──────────────────────────────────────────────────

    #[test]
    fn compaction_success_returns_to_model() {
        let action = TurnTransition::resolve(TurnState::Compacting, &TurnOutcome::Success);
        assert_eq!(
            action,
            TurnAction::Continue {
                next: TurnState::CallingModel
            }
        );
    }

    // ── ProcessingResponse ──────────────────────────────────────────

    #[test]
    fn no_tool_calls_breaks_loop() {
        let action =
            TurnTransition::resolve(TurnState::ProcessingResponse, &TurnOutcome::NoToolCalls);
        assert_eq!(action, TurnAction::BreakLoop);
    }

    #[test]
    fn not_tool_calls_finish_breaks() {
        let action = TurnTransition::resolve(
            TurnState::ProcessingResponse,
            &TurnOutcome::NotToolCalls {
                finish_reason: FinishReason::Stop,
            },
        );
        assert_eq!(action, TurnAction::BreakLoop);
    }

    #[test]
    fn tool_calls_moves_to_execution() {
        let action = TurnTransition::resolve(TurnState::ProcessingResponse, &TurnOutcome::Success);
        assert_eq!(
            action,
            TurnAction::Continue {
                next: TurnState::ExecutingTools
            }
        );
    }

    #[test]
    fn output_truncated_continues_to_model() {
        let action =
            TurnTransition::resolve(TurnState::ProcessingResponse, &TurnOutcome::OutputTruncated);
        assert_eq!(
            action,
            TurnAction::Continue {
                next: TurnState::CallingModel
            }
        );
    }

    // ── Persisting to loop ──────────────────────────────────────────

    #[test]
    fn persisting_success_returns_to_check() {
        let action = TurnTransition::resolve(TurnState::Persisting, &TurnOutcome::Success);
        assert_eq!(
            action,
            TurnAction::Continue {
                next: TurnState::CheckingPolicy
            }
        );
    }

    // ── Status mapping ──────────────────────────────────────────────

    #[test]
    fn status_maps_correctly() {
        assert_eq!(
            TurnTransition::status_for(TurnState::BuildingContext),
            RunStatus::BuildingContext
        );
        assert_eq!(
            TurnTransition::status_for(TurnState::CallingModel),
            RunStatus::CallingModel
        );
        assert_eq!(
            TurnTransition::status_for(TurnState::ExecutingTools),
            RunStatus::WaitingForTools
        );
        assert_eq!(
            TurnTransition::status_for(TurnState::Persisting),
            RunStatus::Persisting
        );
    }
}
