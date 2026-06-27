//! Health status primitives shared by every Component and Transport.
//!
//! [`HealthStatus`] is intentionally minimal: a tri-state (healthy / degraded /
//! unhealthy), each carrying a free-form JSON detail payload. The intent is to
//! expose a uniform shape across providers, stores, transports and the agent
//! runtime itself, so external orchestrators (k8s probes, load balancers, the
//! future `behestctl` CLI) can introspect the system without coupling to
//! internal types.

#![allow(clippy::pedantic)]
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Tri-state health classification of a component or transport.
///
/// All variants carry an optional JSON detail payload for observability. The
/// payload is intentionally not typed — components are free to attach
/// arbitrary diagnostic data (latency, connection pool size, last error
/// timestamp, etc.) without forcing the registry to model every shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum HealthStatus {
    /// Component is operating within its expected parameters.
    Healthy {
        /// Optional free-form diagnostic detail.
        detail: Option<Value>,
    },
    /// Component is operational but reporting a non-fatal issue
    /// (e.g. elevated latency, retry pressure, degraded upstream).
    Degraded {
        /// Human-readable reason for the degraded classification.
        reason: String,
        /// Optional free-form diagnostic detail.
        detail: Option<Value>,
    },
    /// Component is unable to serve requests and must be replaced or
    /// restarted. Operators should treat this as a hard failure.
    Unhealthy {
        /// Human-readable reason for the unhealthy classification.
        reason: String,
        /// Optional free-form diagnostic detail.
        detail: Option<Value>,
    },
}

impl HealthStatus {
    /// Construct a [`HealthStatus::Healthy`] with no detail.
    #[must_use]
    pub fn healthy() -> Self {
        Self::Healthy { detail: None }
    }

    /// Construct a [`HealthStatus::Healthy`] with a JSON detail payload.
    #[must_use]
    pub fn healthy_with(detail: Value) -> Self {
        Self::Healthy {
            detail: Some(detail),
        }
    }

    /// Construct a [`HealthStatus::Degraded`] with a reason.
    #[must_use]
    pub fn degraded(reason: impl Into<String>) -> Self {
        Self::Degraded {
            reason: reason.into(),
            detail: None,
        }
    }

    /// Construct a [`HealthStatus::Degraded`] with a reason and detail.
    #[must_use]
    pub fn degraded_with(reason: impl Into<String>, detail: Value) -> Self {
        Self::Degraded {
            reason: reason.into(),
            detail: Some(detail),
        }
    }

    /// Construct a [`HealthStatus::Unhealthy`] with a reason.
    #[must_use]
    pub fn unhealthy(reason: impl Into<String>) -> Self {
        Self::Unhealthy {
            reason: reason.into(),
            detail: None,
        }
    }

    /// Construct a [`HealthStatus::Unhealthy`] with a reason and detail.
    #[must_use]
    pub fn unhealthy_with(reason: impl Into<String>, detail: Value) -> Self {
        Self::Unhealthy {
            reason: reason.into(),
            detail: Some(detail),
        }
    }

    /// Attach or replace the JSON detail payload.
    #[must_use]
    pub fn with_detail(mut self, detail: Value) -> Self {
        match &mut self {
            Self::Healthy { detail: d }
            | Self::Degraded { detail: d, .. }
            | Self::Unhealthy { detail: d, .. } => *d = Some(detail),
        }
        self
    }

    /// Returns `true` if the status is `Healthy`.
    #[must_use]
    pub const fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy { .. })
    }

    /// Returns `true` if the status is `Healthy` or `Degraded`.
    ///
    /// This is the operational gate most load balancers should use:
    /// `Degraded` components are still serving traffic, just sub-optimally.
    #[must_use]
    pub const fn is_operational(&self) -> bool {
        !matches!(self, Self::Unhealthy { .. })
    }

    /// Short string label for logs and metrics.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Healthy { .. } => "healthy",
            Self::Degraded { .. } => "degraded",
            Self::Unhealthy { .. } => "unhealthy",
        }
    }

    /// Convert to a JSON object suitable for embedding in `/healthz`
    /// responses and OTLP log records.
    #[must_use]
    pub fn to_json(&self) -> Value {
        match self {
            Self::Healthy { detail } => json!({ "status": "healthy", "detail": detail }),
            Self::Degraded { reason, detail } => {
                json!({ "status": "degraded", "reason": reason, "detail": detail })
            }
            Self::Unhealthy { reason, detail } => {
                json!({ "status": "unhealthy", "reason": reason, "detail": detail })
            }
        }
    }
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self::healthy()
    }
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy { detail: Some(d) } => write!(f, "healthy({d})"),
            Self::Healthy { detail: None } => f.write_str("healthy"),
            Self::Degraded {
                reason,
                detail: Some(d),
            } => write!(f, "degraded({reason}; {d})"),
            Self::Degraded {
                reason,
                detail: None,
            } => write!(f, "degraded({reason})"),
            Self::Unhealthy {
                reason,
                detail: Some(d),
            } => {
                write!(f, "unhealthy({reason}; {d})")
            }
            Self::Unhealthy {
                reason,
                detail: None,
            } => {
                write!(f, "unhealthy({reason})")
            }
            // Forward-compatible catch-all for future variants added
            // under the `#[non_exhaustive]` attribute.
            #[allow(unreachable_patterns)]
            _ => f.write_str("unknown"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_default_and_label() {
        let s = HealthStatus::healthy();
        assert!(s.is_healthy());
        assert!(s.is_operational());
        assert_eq!(s.label(), "healthy");
    }

    #[test]
    fn degraded_is_operational_but_unhealthy_is_not() {
        let d = HealthStatus::degraded("slow");
        assert!(!d.is_healthy());
        assert!(d.is_operational());
        assert_eq!(d.label(), "degraded");

        let u = HealthStatus::unhealthy("down");
        assert!(!u.is_healthy());
        assert!(!u.is_operational());
        assert_eq!(u.label(), "unhealthy");
    }

    #[test]
    fn with_detail_attaches_payload() {
        let s = HealthStatus::healthy().with_detail(json!({ "latency_ms": 12 }));
        match s {
            HealthStatus::Healthy { detail: Some(v) } => {
                assert_eq!(v["latency_ms"], 12);
            }
            _ => panic!("expected Healthy with detail"),
        }
    }

    #[test]
    fn to_json_shape_is_stable() {
        let s = HealthStatus::unhealthy_with("redis-down", json!({ "host": "127.0.0.1" }));
        let j = s.to_json();
        assert_eq!(j["status"], "unhealthy");
        assert_eq!(j["reason"], "redis-down");
        assert_eq!(j["detail"]["host"], "127.0.0.1");
    }
}
