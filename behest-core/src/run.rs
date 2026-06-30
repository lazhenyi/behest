//! Sans-IO run state machine for the agent prompt loop.
//!
//! The run loop is modeled as a pure state machine that receives input events,
//! transitions between states, and produces output actions. This design is:
//!
//! - **Runtime-agnostic**: no dependency on tokio or any async runtime.
//! - **Serializable**: the entire machine state can be serialized for
//!   checkpoint/resume across process boundaries.
//! - **Testable**: transitions can be verified without network, model API,
//!   or actual tool execution.
//!
//! The [`transition`] function is the core: `(state, input, config) -> (new_state, actions)`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::{ChatRequest, ContentPart, FinishReason, Message, TokenUsage};
use crate::tool_types::ToolCall;

/// Shared cancellation message to avoid string duplication.
const CANCELLED_MSG: &str = "cancelled";

/// The state of a single agent run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunState {
    /// Initial state, ready to accept a user message.
    Idle,
    /// Building the context (system prompt, history, RAG) for the model call.
    BuildingContext,
    /// Waiting for the model to respond (non-streaming).
    CallingModel,
    /// Receiving streaming chunks from the model.
    StreamingModel,
    /// Processing the completed model response.
    ProcessingResponse,
    /// Executing tool calls requested by the model.
    ExecutingTools,
    /// Waiting for human approval of a tool call.
    WaitingApproval,
    /// Compacting conversation memory.
    Compacting,
    /// Demoting old messages to long-term storage.
    Demoting,
    /// Run completed successfully.
    Finished,
    /// Run was aborted due to an error or cancellation.
    Aborted,
}

/// Input events that drive the state machine forward.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RunInput {
    /// A user message has been received.
    UserMessageReceived {
        /// Content parts of the user message.
        content: Vec<ContentPart>,
    },
    /// A streaming text chunk was received from the model.
    ModelChunkReceived {
        /// Incremental text delta.
        delta: String,
    },
    /// The model completed its response (non-streaming or end of stream).
    ModelCompleted {
        /// The complete assistant message.
        message: Message,
        /// Token usage for this model call.
        usage: TokenUsage,
    },
    /// Tool calls were requested by the model.
    ToolCallRequested {
        /// The tool calls to execute.
        calls: Vec<ToolCall>,
    },
    /// A tool execution completed.
    ToolResultReceived {
        /// The call ID this result corresponds to.
        call_id: String,
        /// The tool output.
        output: Value,
    },
    /// Human approval was granted for a pending tool call.
    ToolApprovalGranted {
        /// The call ID that was approved.
        call_id: String,
    },
    /// Human approval was rejected for a pending tool call.
    ToolApprovalRejected {
        /// The call ID that was rejected.
        call_id: String,
        /// Optional reason for rejection.
        reason: Option<String>,
    },
    /// Memory compaction completed.
    MemoryCompactionCompleted {
        /// Summary text produced by the compactor.
        summary: String,
    },
    /// A runtime-level error occurred.
    RuntimeErrorReceived {
        /// Description of the error.
        error: String,
    },
    /// Cancellation was requested.
    CancellationRequested,
}

/// Actions produced by the state machine that must be executed by the runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RunAction {
    /// Request a model completion (non-streaming or streaming).
    RequestModel {
        /// The chat request to send.
        request: ChatRequest,
    },
    /// Execute one or more tool calls.
    ExecuteTool {
        /// The tool calls to execute.
        calls: Vec<ToolCall>,
    },
    /// Request human approval before executing a tool.
    RequestToolApproval {
        /// The tool call requiring approval.
        call: ToolCall,
        /// Human-readable reason approval is needed.
        reason: String,
    },
    /// Compact conversation memory (summarize old messages).
    CompactMemory,
    /// Demote old messages to long-term storage.
    DemoteMemory,
    /// The run has finished.
    FinishRun {
        /// Reason for finishing.
        reason: FinishReason,
        /// Total token usage for the run.
        usage: TokenUsage,
    },
    /// The run was aborted.
    AbortRun {
        /// Error that caused the abort.
        error: String,
    },
    /// No action needed (e.g., during streaming).
    Noop,
}

/// Configuration for the run state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    /// Maximum number of tool-calling iterations (model → tool → model loops).
    pub max_iterations: usize,
    /// Maximum total tokens across all model calls.
    pub max_tokens: Option<usize>,
    /// Whether to continue when a tool fails.
    pub continue_on_tool_failure: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            max_tokens: None,
            continue_on_tool_failure: true,
        }
    }
}

/// The core transition function.
///
/// Given the current state, an input event, and configuration, returns the next
/// state and the actions the runtime should execute.
///
/// # Determinism
///
/// This function is pure: for the same `(state, input, config)`, it always
/// produces the same `(next_state, actions)`. This property enables:
/// - Deterministic replay from event logs
/// - Snapshot-based crash recovery
/// - Property-based testing
#[must_use]
pub fn transition(
    state: &RunState,
    input: RunInput,
    _config: &RunConfig,
) -> (RunState, Vec<RunAction>) {
    match (state, input) {
        // ---- Idle ----
        (RunState::Idle, RunInput::UserMessageReceived { .. }) => {
            (RunState::BuildingContext, vec![RunAction::Noop])
        }
        (RunState::Idle, RunInput::CancellationRequested) => (
            RunState::Aborted,
            vec![RunAction::AbortRun {
                error: "cancelled before start".to_string(),
            }],
        ),

        // ---- BuildingContext ----
        (RunState::BuildingContext, RunInput::UserMessageReceived { .. }) => {
            // Context built implicitly by the runtime adapter; transition to model call.
            // The runtime adapter constructs the ChatRequest and feeds it back.
            (RunState::CallingModel, vec![RunAction::Noop])
        }

        // ---- CallingModel ----
        (RunState::CallingModel, RunInput::ModelChunkReceived { .. }) => {
            (RunState::StreamingModel, vec![RunAction::Noop])
        }
        (RunState::CallingModel, RunInput::ModelCompleted { message, usage: _ }) => {
            if !message.tool_calls().is_empty() {
                let calls = message.tool_calls().to_vec();
                (
                    RunState::ExecutingTools,
                    vec![RunAction::ExecuteTool { calls }],
                )
            } else {
                (
                    RunState::Finished,
                    vec![RunAction::FinishRun {
                        reason: FinishReason::Stop,
                        usage: TokenUsage::new(0, 0),
                    }],
                )
            }
        }
        (RunState::CallingModel, RunInput::RuntimeErrorReceived { error }) => {
            (RunState::Aborted, vec![RunAction::AbortRun { error }])
        }
        (RunState::CallingModel, RunInput::CancellationRequested) => (
            RunState::Aborted,
            vec![RunAction::AbortRun {
                error: CANCELLED_MSG.to_string(),
            }],
        ),

        // ---- StreamingModel ----
        (RunState::StreamingModel, RunInput::ModelChunkReceived { .. }) => {
            (RunState::StreamingModel, vec![RunAction::Noop])
        }
        (RunState::StreamingModel, RunInput::ModelCompleted { message, usage }) => {
            if !message.tool_calls().is_empty() {
                let calls = message.tool_calls().to_vec();
                (
                    RunState::ExecutingTools,
                    vec![RunAction::ExecuteTool { calls }],
                )
            } else {
                (
                    RunState::Finished,
                    vec![RunAction::FinishRun {
                        reason: FinishReason::Stop,
                        usage,
                    }],
                )
            }
        }
        (RunState::StreamingModel, RunInput::RuntimeErrorReceived { error }) => {
            (RunState::Aborted, vec![RunAction::AbortRun { error }])
        }
        (RunState::StreamingModel, RunInput::CancellationRequested) => (
            RunState::Aborted,
            vec![RunAction::AbortRun {
                error: CANCELLED_MSG.to_string(),
            }],
        ),

        // ---- ProcessingResponse ----
        (RunState::ProcessingResponse, RunInput::ToolCallRequested { calls }) => (
            RunState::ExecutingTools,
            vec![RunAction::ExecuteTool { calls }],
        ),

        // ---- ExecutingTools ----
        (RunState::ExecutingTools, RunInput::ToolResultReceived { .. }) => {
            // After all tools complete, the runtime adapter should transition
            // back to BuildingContext for the next model call.
            // Individual tool results are collected by the runtime.
            (RunState::ExecutingTools, vec![RunAction::Noop])
        }
        (RunState::ExecutingTools, RunInput::RuntimeErrorReceived { .. }) => {
            (RunState::BuildingContext, vec![RunAction::Noop])
        }

        // ---- WaitingApproval ----
        (RunState::WaitingApproval, RunInput::ToolApprovalGranted { call_id }) => (
            RunState::ExecutingTools,
            vec![RunAction::ExecuteTool {
                calls: vec![ToolCall::new(call_id, String::new(), Value::Null)],
            }],
        ),
        (RunState::WaitingApproval, RunInput::ToolApprovalRejected { call_id: _, reason }) => {
            // Tool was rejected; feed rejection back to the model
            let _error_msg = reason.unwrap_or_else(|| "tool execution was rejected".to_string());
            (RunState::BuildingContext, vec![RunAction::Noop])
        }

        // ---- Compacting ----
        (RunState::Compacting, RunInput::MemoryCompactionCompleted { .. }) => {
            (RunState::BuildingContext, vec![RunAction::Noop])
        }

        // ---- Terminal states ----
        (RunState::Finished, _) | (RunState::Aborted, _) => {
            // No transitions from terminal states
            (state.clone(), vec![])
        }

        // ---- Catch-all for unhandled combinations ----
        _ => (state.clone(), vec![RunAction::Noop]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ContentPart;

    fn make_config() -> RunConfig {
        RunConfig::default()
    }

    #[test]
    fn idle_to_building_context_on_user_message() {
        let (next, actions) = transition(
            &RunState::Idle,
            RunInput::UserMessageReceived {
                content: vec![ContentPart::text("hello")],
            },
            &make_config(),
        );
        assert_eq!(next, RunState::BuildingContext);
        assert!(matches!(actions.as_slice(), [RunAction::Noop]));
    }

    #[test]
    fn idle_cancellation_aborts() {
        let (next, actions) = transition(
            &RunState::Idle,
            RunInput::CancellationRequested,
            &make_config(),
        );
        assert_eq!(next, RunState::Aborted);
        assert!(matches!(actions.as_slice(), [RunAction::AbortRun { .. }]));
    }

    #[test]
    fn calling_model_with_tool_calls_transitions_to_executing() {
        let msg = Message::Assistant {
            content: vec![],
            tool_calls: vec![ToolCall::new("call_1", "search", Value::Null)],
        };
        let (next, actions) = transition(
            &RunState::CallingModel,
            RunInput::ModelCompleted {
                message: msg,
                usage: TokenUsage::new(100, 50),
            },
            &make_config(),
        );
        assert_eq!(next, RunState::ExecutingTools);
        assert!(matches!(
            actions.as_slice(),
            [RunAction::ExecuteTool { .. }]
        ));
    }

    #[test]
    fn calling_model_without_tools_finishes() {
        let msg = Message::assistant_text("done");
        let (next, actions) = transition(
            &RunState::CallingModel,
            RunInput::ModelCompleted {
                message: msg,
                usage: TokenUsage::new(100, 50),
            },
            &make_config(),
        );
        assert_eq!(next, RunState::Finished);
        assert!(matches!(actions.as_slice(), [RunAction::FinishRun { .. }]));
    }

    #[test]
    fn terminal_states_no_transition() {
        for state in &[RunState::Finished, RunState::Aborted] {
            let (next, actions) = transition(
                state,
                RunInput::UserMessageReceived {
                    content: vec![ContentPart::text("hello")],
                },
                &make_config(),
            );
            assert_eq!(next, *state);
            assert!(actions.is_empty());
        }
    }

    #[test]
    fn approval_granted_executes_tool() {
        let (next, actions) = transition(
            &RunState::WaitingApproval,
            RunInput::ToolApprovalGranted {
                call_id: "call_1".to_string(),
            },
            &make_config(),
        );
        assert_eq!(next, RunState::ExecutingTools);
        assert!(matches!(
            actions.as_slice(),
            [RunAction::ExecuteTool { .. }]
        ));
    }

    #[test]
    fn approval_rejected_goes_to_building_context() {
        let (next, _actions) = transition(
            &RunState::WaitingApproval,
            RunInput::ToolApprovalRejected {
                call_id: "call_1".to_string(),
                reason: Some("too dangerous".to_string()),
            },
            &make_config(),
        );
        assert_eq!(next, RunState::BuildingContext);
    }

    #[test]
    fn deterministic_same_input_same_output() {
        let state = RunState::Idle;
        let config = make_config();
        let input = RunInput::UserMessageReceived {
            content: vec![ContentPart::text("hello")],
        };

        let (next1, actions1) = transition(&state, input.clone(), &config);
        let (next2, actions2) = transition(&state, input, &config);

        assert_eq!(next1, next2);
        // Compare actions by debug representation
        assert_eq!(format!("{actions1:?}"), format!("{actions2:?}"));
    }

    #[test]
    fn streaming_chunks_keep_streaming_state() {
        let (next, actions) = transition(
            &RunState::StreamingModel,
            RunInput::ModelChunkReceived {
                delta: "hello".to_string(),
            },
            &make_config(),
        );
        assert_eq!(next, RunState::StreamingModel);
        assert!(matches!(actions.as_slice(), [RunAction::Noop]));
    }
}
