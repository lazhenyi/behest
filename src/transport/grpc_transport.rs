//! [`GrpcTransport`]: a [`Transport`] implementation wrapping a
//! tonic gRPC server.
//!
//! The binary constructs a `GrpcTransport` by providing the listen
//! address and a fully configured tonic [`Router`].
//! The transport then serves the router until the shutdown token fires.
//!
//! # Example
//!
//! ```rust,ignore
//! use behest::transport::grpc_transport::GrpcTransport;
//! use behest::transport::TransportHub;
//!
//! let health = tonic_health::server::health_reporter();
//! let router = tonic::transport::Server::builder()
//!     .add_service(health);
//! let transport = GrpcTransport::new("0.0.0.0:50051", router);
//! let hub = TransportHub::new();
//! hub.add("grpc", std::sync::Arc::new(transport)).unwrap();
//! ```

#![allow(clippy::pedantic)]

use std::net::SocketAddr;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tonic::transport::server::Router;

use crate::health::HealthStatus;
use crate::runtime::lifecycle::ShutdownToken;
use crate::transport::Transport;

/// Configuration for a gRPC transport. Currently holds only the
/// listen address; the tonic router is supplied at construction
/// time.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GrpcTransportConfig {
    /// Listen address (e.g. `"0.0.0.0:50051"`).
    pub addr: String,
}

/// Errors raised by the gRPC transport.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GrpcTransportError {
    /// The address string could not be parsed.
    #[error("invalid listen address: {0}")]
    InvalidAddress(#[from] std::net::AddrParseError),
    /// The tonic server returned an error during serve.
    #[error("gRPC serve error: {0}")]
    Serve(String),
    /// The router was already consumed by a previous serve call.
    #[error("gRPC router already consumed")]
    AlreadyServed,
}

/// A [`Transport`] that wraps a tonic gRPC server.
///
/// Construct via [`GrpcTransport::new`], supplying the listen
/// address and a fully configured tonic [`Router`]. The router is
/// consumed on the first call to `serve`; subsequent calls return
/// [`GrpcTransportError::AlreadyServed`].
pub struct GrpcTransport {
    addr: SocketAddr,
    router: Mutex<Option<Router>>,
}

impl GrpcTransport {
    /// Construct a new gRPC transport from a listen address and a
    /// tonic router.
    ///
    /// The router should contain all services and interceptors
    /// needed. The transport will call
    /// `Router::serve_with_shutdown` on it when started.
    ///
    /// # Errors
    /// Returns [`GrpcTransportError::InvalidAddress`] if the
    /// address string cannot be parsed as a [`SocketAddr`].
    pub fn new(addr: impl AsRef<str>, router: Router) -> Result<Self, GrpcTransportError> {
        let addr: SocketAddr = addr.as_ref().parse()?;
        Ok(Self {
            addr,
            router: Mutex::new(Some(router)),
        })
    }

    /// Borrow the configured listen address.
    #[must_use]
    pub fn listen_addr(&self) -> SocketAddr {
        self.addr
    }
}

#[async_trait]
impl Transport for GrpcTransport {
    const NAME: &'static str = "transport.grpc";
    type Config = GrpcTransportConfig;
    type Error = GrpcTransportError;

    async fn serve(&self, shutdown: ShutdownToken) -> Result<(), Self::Error> {
        let router = self
            .router
            .lock()
            .await
            .take()
            .ok_or(GrpcTransportError::AlreadyServed)?;

        router
            .serve_with_shutdown(self.addr, async move {
                shutdown.wait().await;
            })
            .await
            .map_err(|e| GrpcTransportError::Serve(e.to_string()))
    }

    async fn health(&self) -> HealthStatus {
        HealthStatus::healthy()
    }

    fn addr(&self) -> Option<String> {
        Some(self.addr.to_string())
    }
}

impl std::fmt::Debug for GrpcTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrpcTransport")
            .field("addr", &self.addr)
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn test_router() -> Router {
        let (_reporter, health_svc) = tonic_health::server::health_reporter();
        tonic::transport::Server::builder().add_service(health_svc)
    }

    #[test]
    fn new_parses_socket_addr() {
        let transport =
            GrpcTransport::new("127.0.0.1:50051", test_router()).expect("should parse address");
        assert_eq!(transport.listen_addr().port(), 50051);
        assert_eq!(transport.addr(), Some("127.0.0.1:50051".to_string()));
    }

    #[test]
    fn new_rejects_invalid_address() {
        let result = GrpcTransport::new("not-a-valid-addr", test_router());
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn health_returns_healthy() {
        let transport =
            GrpcTransport::new("127.0.0.1:0", test_router()).expect("should parse address");
        let h = transport.health().await;
        assert!(h.is_healthy());
    }
}
