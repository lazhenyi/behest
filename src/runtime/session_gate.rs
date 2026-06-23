//! Per-session concurrency gate.
//!
//! Prevents concurrent runs from interleaving writes to the same session.
//! Each session is protected by its own [`tokio::sync::Mutex`]; a run
//! acquires the lock at entry and holds it until the run completes or
//! is cancelled.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

/// RAII guard that holds a per-session lock for the duration of a run.
///
/// Dropping this guard releases the session lock, allowing the next
/// queued run to proceed.
pub struct SessionGuard {
    _inner: tokio::sync::OwnedMutexGuard<()>,
}

impl SessionGuard {
    /// Creates a new guard wrapping an acquired mutex lock.
    fn new(inner: tokio::sync::OwnedMutexGuard<()>) -> Self {
        Self { _inner: inner }
    }
}

impl std::fmt::Debug for SessionGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionGuard").finish_non_exhaustive()
    }
}

/// Registry of per-session mutexes.
///
/// Uses a `RwLock<HashMap>` to lazily create mutex entries.  The outer
/// `RwLock` is held briefly (read or write) to look up / insert a
/// mutex *handle*; the actual session serialization happens via the
/// inner `Mutex`.
#[derive(Debug, Clone, Default)]
pub struct SessionGate {
    locks: Arc<RwLock<HashMap<Uuid, Arc<Mutex<()>>>>>,
}

impl SessionGate {
    /// Creates a new empty gate.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquires a session lock, returning an RAII guard.
    ///
    /// If the session is already held by another run, this returns
    /// immediately with an error rather than waiting — concurrent
    /// writes to the same session are a programming mistake, not a
    /// normal contention scenario.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::SessionBusy`] if the lock is already held.
    pub async fn acquire(&self, session_id: Uuid) -> Result<SessionGuard, SessionBusy> {
        let mutex = {
            let read = self.locks.read().await;
            read.get(&session_id).cloned()
        };

        let mutex = if let Some(m) = mutex {
            m
        } else {
            let mut write = self.locks.write().await;
            write
                .entry(session_id)
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };

        // Try to acquire the lock without blocking.  If another run
        // already holds it, return an error immediately.
        let guard = mutex
            .try_lock_owned()
            .map_err(|_| SessionBusy { session_id })?;

        Ok(SessionGuard::new(guard))
    }
}

/// Error returned when a session is already being processed by another run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionBusy {
    /// The session that was already locked.
    pub session_id: Uuid,
}

impl std::fmt::Display for SessionBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "session {} is busy — another run is active",
            self.session_id
        )
    }
}

impl std::error::Error for SessionBusy {}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn acquire_and_release() {
        let gate = SessionGate::new();
        let sid = Uuid::new_v4();

        let guard = gate.acquire(sid).await.expect("should acquire");
        // Guard is alive — session is locked.
        drop(guard);

        // After drop, the same session can be acquired again.
        let _guard2 = gate
            .acquire(sid)
            .await
            .expect("should re-acquire after drop");
    }

    #[tokio::test]
    async fn concurrent_acquire_fails() {
        let gate = SessionGate::new();
        let sid = Uuid::new_v4();

        let guard = gate.acquire(sid).await.expect("first acquire");
        let result = gate.acquire(sid).await;
        assert!(result.is_err(), "second acquire should fail");
        assert_eq!(result.unwrap_err().session_id, sid);

        drop(guard);

        // After release, acquire succeeds.
        let _guard2 = gate
            .acquire(sid)
            .await
            .expect("should succeed after release");
    }

    #[tokio::test]
    async fn independent_sessions_no_contention() {
        let gate = SessionGate::new();
        let sid1 = Uuid::new_v4();
        let sid2 = Uuid::new_v4();

        let guard1 = gate.acquire(sid1).await.expect("acquire sid1");
        let guard2 = gate.acquire(sid2).await.expect("acquire sid2");

        // Both acquired — no contention.
        drop(guard1);
        drop(guard2);
    }

    #[tokio::test]
    async fn sequential_runs_serialize() {
        let gate = SessionGate::new();
        let sid = Uuid::new_v4();

        let g1 = gate.acquire(sid).await.expect("first");
        // Simulate first run finishing.
        drop(g1);

        let g2 = gate.acquire(sid).await.expect("second");
        drop(g2);
    }
}
