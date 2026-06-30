//! Prompt caching control types.
//!
//! Provider-neutral abstractions for marking message content and tool
//! definitions as eligible for prompt caching. Adapters translate these
//! markers to provider-specific wire formats (e.g. Anthropic's
//! `cache_control` blocks) or rely on the provider's automatic caching
//! (e.g. OpenAI's prefix cache).
//!
//! # Stability model
//!
//! - `CacheControl` is a per-`ContentPart` and per-`ToolSpec` marker.
//! - A marker on the last content block of a stable region (system
//!   prompt, tool definitions, conversation tail) instructs the provider
//!   to create a cache entry covering everything from the start of the
//!   prompt up to and including that block.
//! - Markers have no effect on providers that do not support caching;
//!   they are silently ignored.
//!
//! # Example
//!
//! ```rust
//! use behest_core::cache::CacheControl;
//! use behest_core::message::{ContentPart, Message};
//!
//! let ctrl = CacheControl::ephemeral();
//! let part = ContentPart::text("You are a research assistant.").with_cache_control(ctrl);
//! let message = Message::system_text("You are a research assistant.")
//!     .mark_cache_breakpoint();
//! ```

use serde::{Deserialize, Serialize};

/// Provider-neutral cache control marker.
///
/// Attach to a [`crate::message::ContentPart`] or [`crate::tool_types::ToolSpec`]
/// to request that the provider cache the prefix ending at this marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheControl {
    /// Cache strategy. Currently only `Ephemeral` is supported across all
    /// integrated providers.
    pub kind: CacheControlKind,
    /// Time-to-live for the cache entry.
    pub ttl: CacheTtl,
}

impl CacheControl {
    /// Returns an `Ephemeral` cache control with the default 5-minute TTL.
    #[must_use]
    pub const fn ephemeral() -> Self {
        Self {
            kind: CacheControlKind::Ephemeral,
            ttl: CacheTtl::FiveMinutes,
        }
    }

    /// Returns an `Ephemeral` cache control with the specified TTL.
    #[must_use]
    pub const fn with_ttl(mut self, ttl: CacheTtl) -> Self {
        self.ttl = ttl;
        self
    }

    /// Returns the wire-format TTL string (e.g. `"5m"`, `"1h"`).
    #[must_use]
    pub const fn ttl_wire(&self) -> &'static str {
        match self.ttl {
            CacheTtl::FiveMinutes => "5m",
            CacheTtl::OneHour => "1h",
        }
    }
}

impl Default for CacheControl {
    fn default() -> Self {
        Self::ephemeral()
    }
}

/// Cache strategy kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CacheControlKind {
    /// Short-lived cache that the provider may evict.
    Ephemeral,
}

/// Time-to-live for a cache entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CacheTtl {
    /// 5-minute TTL. Cheapest cache writes (1.25× base input price).
    #[default]
    FiveMinutes,
    /// 1-hour TTL. More expensive cache writes (2× base input price) but
    /// better hit rate for long-lived sessions.
    OneHour,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn cache_control_default_is_ephemeral_five_minutes() {
        let ctrl = CacheControl::default();
        assert_eq!(ctrl.kind, CacheControlKind::Ephemeral);
        assert_eq!(ctrl.ttl, CacheTtl::FiveMinutes);
    }

    #[test]
    fn cache_control_with_ttl_overrides() {
        let ctrl = CacheControl::ephemeral().with_ttl(CacheTtl::OneHour);
        assert_eq!(ctrl.ttl, CacheTtl::OneHour);
    }

    #[test]
    fn cache_control_ttl_wire_strings() {
        assert_eq!(CacheControl::ephemeral().ttl_wire(), "5m");
        assert_eq!(
            CacheControl::ephemeral()
                .with_ttl(CacheTtl::OneHour)
                .ttl_wire(),
            "1h"
        );
    }
}
