//! Transport-neutral runtime stream primitives.
//!
//! This module defines the envelope, routing key, and stream type used by the
//! runtime stream abstraction. The design is inspired by the Socket.IO Adapter
//! rules (room-based fanout, per-room ordering, best-effort live delivery), but
//! it is **not** a Socket.IO implementation and carries no transport coupling.
//!
//! - [`RuntimeStreamAdapter`](super::stream_adapter::RuntimeStreamAdapter)
//!   performs best-effort live fanout only.
//! - [`RuntimeEventStore`](super::event_store::RuntimeEventStore) is the
//!   authoritative replay source.
//! - Delivery is at-least-once; consumers deduplicate via [`RuntimeEventId`]
//!   or [`RuntimeEventEnvelope::seq`].
//! - Authorization is **not** modeled here; rooms express fanout routing only
//!   and must be gated by the transport/service layer before subscription.

use std::fmt;
use std::pin::Pin;

use chrono::{DateTime, Utc};
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use super::run::RunId;
use behest_event::AgentEvent;
use behest_provider::ProviderId;

/// Globally unique identifier for a single runtime event.
///
/// Use [`RuntimeEventId::new`] to mint a fresh id (UUIDv7, time-ordered). It is
/// intentionally a newtype so callers cannot accidentally pass an arbitrary
/// [`Uuid`] where an event identity is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuntimeEventId(Uuid);

impl RuntimeEventId {
    /// Mints a new time-ordered event id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// Wraps an existing [`Uuid`] as a [`RuntimeEventId`].
    #[must_use]
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Views the underlying [`Uuid`].
    #[must_use]
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for RuntimeEventId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RuntimeEventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<RuntimeEventId> for Uuid {
    fn from(value: RuntimeEventId) -> Self {
        value.0
    }
}

/// Envelope wrapping an [`AgentEvent`] with cross-instance routing metadata.
///
/// `seq` is monotonic per `run_id`; it is the unit clients use to resume a
/// stream after a disconnect (`run_id + seq`). `event_id` is globally unique
/// and is the canonical deduplication key when the same envelope is delivered
/// more than once.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEventEnvelope {
    /// Globally unique event identity.
    pub event_id: RuntimeEventId,
    /// Per-run monotonic sequence number.
    pub seq: u64,
    /// Run this event belongs to.
    pub run_id: RunId,
    /// Session this event belongs to, when known from `RunStarted`.
    pub session_id: Option<Uuid>,
    /// The wrapped runtime event.
    pub event: AgentEvent,
    /// When the envelope was emitted by the runtime bridge.
    pub emitted_at: DateTime<Utc>,
}

impl RuntimeEventEnvelope {
    /// Returns the run identifier carried by the wrapped event.
    #[must_use]
    pub fn run_id(&self) -> RunId {
        self.event.run_id()
    }

    /// Delegates to [`AgentEvent::is_terminal`].
    ///
    /// True for `RunCompleted` / `RunFailed` / `RunCancelled`.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.event.is_terminal()
    }
}

/// Fanout routing key, inspired by a Socket.IO room but transport-neutral.
///
/// A room expresses **where** an event should be live-fanned-out; it carries no
/// authorization semantics. A transport/service layer must authorize
/// subscriptions before they reach [`RuntimeRoom`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuntimeRoom {
    /// All events for a specific run.
    Run(RunId),
    /// All events for a specific session.
    Session(Uuid),
    /// All events for a specific provider.
    Provider(ProviderId),
}

impl fmt::Display for RuntimeRoom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeRoom::Run(run_id) => write!(f, "run:{run_id}"),
            RuntimeRoom::Session(session_id) => write!(f, "session:{session_id}"),
            RuntimeRoom::Provider(provider_id) => write!(f, "provider:{provider_id}"),
        }
    }
}

/// Owned, type-erased stream of runtime event envelopes.
///
/// Adapters return this from `subscribe`; it yields `Err` on transient lag or
/// channel closure and `Ok(envelope)` for delivered events.
pub type BoxRuntimeEventStream =
    Pin<Box<dyn Stream<Item = Result<RuntimeEventEnvelope, RuntimeStreamError>> + Send + 'static>>;

/// Errors raised by a [`RuntimeStreamAdapter`](super::stream_adapter::RuntimeStreamAdapter).
///
/// The adapter only models best-effort live fanout, so failures here are
/// recoverable: callers are expected to fall back to the event store for
/// replay and deduplicate by `event_id`/`seq`.
#[derive(Debug, Error)]
pub enum RuntimeStreamError {
    /// A live publish could not be delivered to any live consumer.
    #[error("runtime stream publish failed: {message}")]
    Publish {
        /// Human-readable diagnostic.
        message: String,
    },
    /// A subscriber lagged behind the live stream and skipped events.
    ///
    /// The adapter surface this as an error so consumers can decide whether to
    /// tolerate the gap or reconcile from the event store.
    #[error("runtime stream subscriber lagged, skipped {skipped} events")]
    Lagged {
        /// Number of events the receiver skipped.
        skipped: u64,
    },
    /// A subscription could not be established.
    #[error("runtime stream subscribe failed: {message}")]
    Subscribe {
        /// Human-readable diagnostic.
        message: String,
    },
    /// The live stream has been closed and will yield no further events.
    #[error("runtime stream closed")]
    Closed,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn run_room_display_is_stable() {
        let id = RunId::from_uuid(Uuid::nil());
        assert_eq!(RuntimeRoom::Run(id).to_string(), format!("run:{id}"));
    }

    #[test]
    fn session_room_display_is_stable() {
        let id = Uuid::nil();
        assert_eq!(
            RuntimeRoom::Session(id).to_string(),
            format!("session:{id}")
        );
    }

    #[test]
    fn provider_room_display_is_stable() {
        let id = ProviderId::new("acme");
        let expected = format!("provider:{id}");
        assert_eq!(RuntimeRoom::Provider(id).to_string(), expected);
    }
}
