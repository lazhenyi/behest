//! Human-in-the-loop tool approval for the behest agent runtime.
//!
//! Controls execution of sensitive, destructive, or high-cost tool calls
//! by requiring human approval before execution.
//!
//! # Key types
//!
//! - [`ApprovalPolicy`]: controls when approval is required
//! - [`PendingApproval`]: a tool call waiting for human decision
//! - [`ApprovalDecision`]: the outcome of an approval request
//! - [`ApprovalGate`]: manages pending approvals with timeout support

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use behest_core::tool_types::ToolCall;
use tokio::sync::Mutex;

/// Controls when tool calls require human approval.
#[derive(Debug, Clone, Default)]
pub enum ApprovalPolicy {
    /// Never require approval.
    AlwaysAllow,
    /// Always require approval for all tools.
    AlwaysDeny,
    /// Ask the user for each tool that requires approval.
    AskUser,
    /// Automatically approve read-only tools, ask for all others.
    #[default]
    AutoForReadOnly,
}

impl ApprovalPolicy {
    /// Returns `true` if the given tool requires approval under this policy.
    #[must_use]
    pub fn requires_approval(&self, tool_requires_approval: bool, tool_is_read_only: bool) -> bool {
        match self {
            Self::AlwaysAllow => false,
            Self::AlwaysDeny => true,
            Self::AskUser => tool_requires_approval,
            Self::AutoForReadOnly => tool_requires_approval && !tool_is_read_only,
        }
    }
}

/// A tool call waiting for human approval.
#[derive(Debug, Clone)]
pub struct PendingApproval {
    /// The call awaiting approval.
    pub call: ToolCall,
    /// Human-readable reason for the approval request.
    pub reason: String,
    /// When the approval was requested.
    pub requested_at: Instant,
    /// Optional timeout, after which the approval is considered denied.
    pub timeout: Option<Duration>,
}

impl Default for PendingApproval {
    fn default() -> Self {
        Self::new(
            ToolCall::new("", "", serde_json::Value::Null),
            String::new(),
            None,
        )
    }
}

impl PendingApproval {
    /// Creates a new pending approval request.
    #[must_use]
    pub fn new(call: ToolCall, reason: String, timeout: Option<Duration>) -> Self {
        Self {
            call,
            reason,
            requested_at: Instant::now(),
            timeout,
        }
    }

    /// Returns `true` if the approval has timed out.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.timeout
            .is_some_and(|t| self.requested_at.elapsed() >= t)
    }
}

/// The outcome of an approval request.
#[derive(Debug, Clone)]
pub enum ApprovalDecision {
    /// The tool call was approved for execution.
    Approve,
    /// The tool call was rejected.
    Reject {
        /// Optional reason for rejection (shown to the model).
        reason: Option<String>,
    },
}

/// Manages pending approvals with timeout support.
///
/// Thread-safe: can be shared between the run loop and an external
/// approval UI or API endpoint.
pub struct ApprovalGate {
    pending: Arc<Mutex<HashMap<String, PendingApproval>>>,
    decisions: Arc<Mutex<HashMap<String, ApprovalDecision>>>,
}

impl ApprovalGate {
    /// Creates a new approval gate.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pending: Arc::new(Mutex::new(HashMap::new())),
            decisions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Submits a tool call for approval.
    ///
    /// Returns `true` if the call was registered (not a duplicate).
    pub async fn request(&self, call: ToolCall, reason: String, timeout: Option<Duration>) -> bool {
        let mut pending = self.pending.lock().await;
        if pending.contains_key(&call.id) {
            return false;
        }
        pending.insert(call.id.clone(), PendingApproval::new(call, reason, timeout));
        true
    }

    /// Records an approval decision for a pending call.
    ///
    /// Returns `true` if the call was found and the decision was recorded.
    pub async fn decide(&self, call_id: &str, decision: ApprovalDecision) -> bool {
        let mut pending = self.pending.lock().await;
        if pending.remove(call_id).is_some() {
            let mut decisions = self.decisions.lock().await;
            decisions.insert(call_id.to_string(), decision);
            true
        } else {
            false
        }
    }

    /// Checks if a call has been approved.
    pub async fn is_approved(&self, call_id: &str) -> bool {
        let decisions = self.decisions.lock().await;
        matches!(decisions.get(call_id), Some(ApprovalDecision::Approve))
    }

    /// Returns any expired approvals, removing them from pending.
    pub async fn expire_timed_out(&self) -> Vec<String> {
        let mut pending = self.pending.lock().await;
        let expired: Vec<String> = pending
            .iter()
            .filter(|(_, p)| p.is_expired())
            .map(|(id, _)| id.clone())
            .collect();
        for id in &expired {
            pending.remove(id);
        }
        expired
    }

    /// Returns the number of pending approvals.
    pub async fn pending_count(&self) -> usize {
        self.pending.lock().await.len()
    }
}

impl Default for ApprovalGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use behest_core::tool_types::ToolCall;
    use serde_json::Value;

    fn make_call(id: &str) -> ToolCall {
        ToolCall::new(id, "test_tool", Value::Null)
    }

    #[test]
    fn policy_always_allow_never_requires() {
        assert!(!ApprovalPolicy::AlwaysAllow.requires_approval(true, false));
        assert!(!ApprovalPolicy::AlwaysAllow.requires_approval(true, true));
    }

    #[test]
    fn policy_always_deny_always_requires() {
        assert!(ApprovalPolicy::AlwaysDeny.requires_approval(false, true));
    }

    #[test]
    fn policy_auto_read_only_skips_ro_tools() {
        assert!(ApprovalPolicy::AutoForReadOnly.requires_approval(true, false));
        assert!(!ApprovalPolicy::AutoForReadOnly.requires_approval(true, true));
    }

    #[test]
    fn pending_approval_expiry() {
        let call = make_call("call_1");
        let pending =
            PendingApproval::new(call, "test".to_string(), Some(Duration::from_millis(1)));
        std::thread::sleep(Duration::from_millis(2));
        assert!(pending.is_expired());
    }

    #[tokio::test]
    async fn approval_gate_request_and_approve() {
        let gate = ApprovalGate::new();
        let call = make_call("call_1");

        assert!(gate.request(call.clone(), "test".to_string(), None).await);
        assert!(!gate.request(call.clone(), "test".to_string(), None).await); // duplicate

        assert!(gate.decide("call_1", ApprovalDecision::Approve).await);
        assert!(gate.is_approved("call_1").await);
    }

    #[tokio::test]
    async fn approval_gate_reject() {
        let gate = ApprovalGate::new();
        let call = make_call("call_2");

        gate.request(call, "test".to_string(), None).await;
        gate.decide(
            "call_2",
            ApprovalDecision::Reject {
                reason: Some("too expensive".to_string()),
            },
        )
        .await;
        assert!(!gate.is_approved("call_2").await);
    }

    #[tokio::test]
    async fn approval_gate_nonexistent_call() {
        let gate = ApprovalGate::new();
        assert!(!gate.decide("nonexistent", ApprovalDecision::Approve).await);
        assert!(!gate.is_approved("nonexistent").await);
    }

    #[tokio::test]
    async fn approval_gate_timeout() {
        let gate = ApprovalGate::new();
        let call = make_call("call_3");

        gate.request(call, "test".to_string(), Some(Duration::from_millis(1)))
            .await;
        tokio::time::sleep(Duration::from_millis(5)).await;

        let expired = gate.expire_timed_out().await;
        assert_eq!(expired, vec!["call_3"]);
        assert_eq!(gate.pending_count().await, 0);
    }
}
