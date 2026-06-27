//! Drain-aware replace protocol primitives for
//! [`ExtensionPoint`](super::extension::ExtensionPoint).
//!
//! [`ReplaceToken`] is a one-shot, three-state handle
//! (`Pending` / `Committed` / `Aborted`) handed out by
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
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use thiserror::Error;

/// Default drain timeout used by
/// [`ExtensionPoint::begin_replace`](super::extension::ExtensionPoint::begin_replace)
/// when no explicit deadline is supplied.
pub const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

const STATE_PENDING: u8 = 0;
const STATE_COMMITTED: u8 = 1;
const STATE_ABORTED: u8 = 2;

/// State of a [`ReplaceToken`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReplaceState {
    /// The token has been handed out but neither committed nor aborted.
    Pending,
    /// The token was committed by a successful `complete_replace` call.
    Committed,
    /// The token was aborted; the paired `complete_replace` call will
    /// not write the new value.
    Aborted,
}

impl ReplaceState {
    fn from_u8(value: u8) -> Self {
        match value {
            STATE_PENDING => Self::Pending,
            STATE_COMMITTED => Self::Committed,
            STATE_ABORTED => Self::Aborted,
            // Future state values fall back to Aborted for forward
            // compatibility.
            _ => Self::Aborted,
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
    /// The token was aborted before `complete_replace` was able to
    /// commit it.
    #[error("replace was aborted before completion")]
    Aborted,
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
    state: Arc<AtomicU8>,
    timeout: Duration,
}

impl ReplaceToken {
    /// Construct a fresh `Pending` token with the given drain timeout.
    #[must_use]
    pub fn new(timeout: Duration) -> Self {
        Self {
            state: Arc::new(AtomicU8::new(STATE_PENDING)),
            timeout,
        }
    }

    /// Current state of the token.
    #[must_use]
    pub fn state(&self) -> ReplaceState {
        ReplaceState::from_u8(self.state.load(Ordering::SeqCst))
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
        let prior = self.state.swap(STATE_COMMITTED, Ordering::SeqCst);
        match ReplaceState::from_u8(prior) {
            ReplaceState::Pending => Ok(()),
            ReplaceState::Committed => Err(ReplaceError::AlreadyCommitted),
            ReplaceState::Aborted => Err(ReplaceError::Aborted),
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
