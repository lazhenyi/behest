//! Drain-aware replace protocol primitives for
//! [`ExtensionPoint`](super::extension::ExtensionPoint).
//!
//! [`ReplaceToken`] is a one-shot, two-state handle
//! (`Pending` / `Committed`) handed out by
//! [`ExtensionPoint::begin_replace`](super::extension::ExtensionPoint::begin_replace).
//! A paired
//! [`ExtensionPoint::complete_replace`](super::extension::ExtensionPoint::complete_replace)
//! call commits the token, waits for the previous `Arc<T>` to drain to
//! the registry's single reference, and then writes the new value.
//!
//! The token is intentionally not coupled to a specific `ExtensionPoint`
//! instance: callers can move it freely, hand it to another task, or
//! store it alongside a [`ShutdownToken`](super::lifecycle::ShutdownToken)
//! to enforce a deadline.

#![allow(clippy::pedantic)]
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use thiserror::Error;

/// Default drain timeout used by
/// [`ExtensionPoint::begin_replace`](super::extension::ExtensionPoint::begin_replace)
/// when no explicit deadline is supplied.
pub const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// State of a [`ReplaceToken`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReplaceState {
    /// The token has been handed out but not yet committed.
    Pending,
    /// The token was committed by a successful `complete_replace` call.
    Committed,
}

impl ReplaceState {
    fn from_bool(committed: bool) -> Self {
        if committed {
            Self::Committed
        } else {
            Self::Pending
        }
    }
}

/// Errors raised by the drain-aware replace protocol.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReplaceError {
    /// `complete_replace` was called more than once with the same token.
    #[error("replace token was already committed")]
    AlreadyCommitted,
}

/// One-shot, shared-state handle for a drain-aware replace operation.
///
/// A `ReplaceToken` is created by
/// [`ExtensionPoint::begin_replace`](super::extension::ExtensionPoint::begin_replace)
/// and consumed by the corresponding
/// [`ExtensionPoint::complete_replace`](super::extension::ExtensionPoint::complete_replace)
/// call. Cloning the token is cheap; all clones share the same internal
/// state and see the same transitions.
#[derive(Debug, Clone)]
pub struct ReplaceToken {
    state: Arc<AtomicBool>,
    timeout: Duration,
}

impl ReplaceToken {
    /// Construct a fresh `Pending` token with the given drain timeout.
    #[must_use]
    pub fn new(timeout: Duration) -> Self {
        Self {
            state: Arc::new(AtomicBool::new(false)),
            timeout,
        }
    }

    /// Current state of the token.
    #[must_use]
    pub fn state(&self) -> ReplaceState {
        ReplaceState::from_bool(self.state.load(Ordering::SeqCst))
    }

    /// Drain timeout configured at construction time.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Atomically transition `Pending -> Committed`. Returns the error
    /// that corresponds to the prior state if the transition cannot be
    /// made.
    pub(crate) fn try_commit(&self) -> Result<(), ReplaceError> {
        if self.state.swap(true, Ordering::SeqCst) {
            Err(ReplaceError::AlreadyCommitted)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_token_is_pending() {
        let t = ReplaceToken::new(Duration::from_secs(1));
        assert_eq!(t.state(), ReplaceState::Pending);
        assert_eq!(t.timeout(), Duration::from_secs(1));
    }

    #[test]
    fn try_commit_pending_returns_ok() {
        let t = ReplaceToken::new(Duration::from_secs(1));
        assert!(t.try_commit().is_ok());
        assert_eq!(t.state(), ReplaceState::Committed);
    }

    #[test]
    fn try_commit_twice_returns_already_committed() {
        let t = ReplaceToken::new(Duration::from_secs(1));
        assert!(t.try_commit().is_ok());
        match t.try_commit() {
            Err(ReplaceError::AlreadyCommitted) => {}
            other => panic!("expected AlreadyCommitted, got {other:?}"),
        }
    }

    #[test]
    fn clones_share_state() {
        let a = ReplaceToken::new(Duration::from_millis(500));
        let b = a.clone();
        assert!(a.try_commit().is_ok());
        assert_eq!(b.state(), ReplaceState::Committed);
    }

    #[test]
    fn default_drain_timeout_is_thirty_seconds() {
        assert_eq!(DEFAULT_DRAIN_TIMEOUT, Duration::from_secs(30));
    }
}
