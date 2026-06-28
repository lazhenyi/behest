//! Transport layer for the composable runtime.
//!
//! A [`Transport`] is a long-lived network-facing component that exposes
//! a runtime over some protocol. The composable runtime model treats
//! gRPC, HTTP+SSE, and future transports as interchangeable plug-in
//! implementations of the same trait, each registered into a
//! [`TransportHub`] and started together.
//!
//! # Architecture
//!
//! ```text
//!   TransportHub
//!        │
//!        ├── gRPC transport   (default, feature = "server")
//!        ├── HTTP+SSE transport (future)
//!        └── WebSocket transport (future)
//! ```
//!
//! Each transport is an `Arc<dyn Transport>` and can be hot-swapped at
//! runtime via [`TransportHub::add`] (which accepts a new instance
//! and registers it under a name). Existing callers continue to hold
//! their `Arc`, so a swap is non-disruptive for in-flight requests.

#![allow(clippy::pedantic)]

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::future::BoxFuture;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use thiserror::Error;

use crate::health::HealthStatus;
use crate::runtime::lifecycle::ShutdownToken;

pub mod hub;

#[cfg(feature = "server")]
pub mod grpc;

#[cfg(feature = "server")]
pub mod grpc_transport;

pub use hub::TransportHub;

#[cfg(feature = "server")]
pub use grpc_transport::GrpcTransport;

/// Errors raised by transport operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TransportError {
    /// Tried to register a transport with a name that is already in use.
    #[error("transport `{name}` already registered")]
    AlreadyRegistered {
        /// The conflicting name.
        name: String,
    },
    /// Tried to look up a transport that is not registered.
    #[error("transport `{name}` not found")]
    NotFound {
        /// The missing name.
        name: String,
    },
    /// The transport's serve future returned an error.
    #[error("transport `{name}` failed: {message}")]
    Serve {
        /// Name of the failing transport.
        name: String,
        /// Human-readable error.
        message: String,
    },
    /// Internal lock acquisition failed.
    #[error("transport hub lock poisoned")]
    LockPoisoned,
}

/// Transport configuration shape. Implementations deserialize their
/// own concrete type using `serde_json::from_value` or similar.
pub trait TransportConfig: DeserializeOwned + JsonSchema + Send + Sync + 'static {}

/// Blanket implementation for any compatible type.
impl<T> TransportConfig for T where T: DeserializeOwned + JsonSchema + Send + Sync + 'static {}

/// A long-lived network-facing component that exposes a runtime over
/// some protocol (gRPC, HTTP+SSE, WebSocket, etc.).
///
/// Implementations are constructed externally and wrapped in an
/// `Arc<dyn Transport>`. The hub then drives the lifecycle via
/// [`Transport::serve`].
///
/// # Lifecycle
///
/// ```text
///   construct
///       ↓
///   serve(shutdown) ──► runs until shutdown
///       ↓
///   health() — probe at any time
/// ```
///
/// `serve` is the only blocking call. Implementations should respect
/// the supplied [`ShutdownToken`] and return `Ok(())` when it fires.
#[async_trait]
pub trait Transport: Send + Sync + 'static {
    /// Stable identifier for the transport kind (e.g. `"transport.grpc"`).
    const NAME: &'static str;

    /// Configuration shape. Implementations deserialize this from
    /// [`crate::config::TransportConfig::config`].
    type Config: TransportConfig;

    /// Error type for [`Transport::serve`].
    type Error: std::error::Error + Send + Sync + 'static;

    /// Begin serving. The returned future resolves when the transport
    /// has stopped, either because the shutdown token fired or
    /// because the transport encountered a fatal error.
    async fn serve(&self, shutdown: ShutdownToken) -> Result<(), Self::Error>;

    /// Non-mutating health probe. Default is [`HealthStatus::healthy`].
    async fn health(&self) -> HealthStatus {
        HealthStatus::healthy()
    }

    /// Optional textual representation of the bound address
    /// (e.g. `"0.0.0.0:50051"`). Default is `None`.
    fn addr(&self) -> Option<String> {
        None
    }
}

/// Type-erased [`Transport`] handle, used by [`TransportHub`] for
/// heterogeneous storage.
#[async_trait]
pub trait AnyTransport: Send + Sync + 'static {
    /// Returns the kind identifier (e.g. `"transport.grpc"`).
    fn kind(&self) -> &'static str;

    /// Returns a `TypeId` of the concrete transport type.
    fn type_id(&self) -> std::any::TypeId;

    /// Downcasts the underlying `Arc<T>` by type.
    fn as_any_arc(&self) -> Arc<dyn Any + Send + Sync>;

    /// Begin serving. See [`Transport::serve`].
    fn serve(
        self: Arc<Self>,
        shutdown: ShutdownToken,
    ) -> BoxFuture<'static, Result<(), TransportError>>;

    /// Health probe. See [`Transport::health`].
    fn health(&self) -> BoxFuture<'static, HealthStatus>;

    /// Returns the bound address, if known.
    fn addr(&self) -> Option<String> {
        None
    }
}

/// Adapter from a typed [`Transport`] to [`AnyTransport`].
pub struct TypedAnyTransport<T: Transport + ?Sized> {
    inner: Arc<T>,
}

impl<T: Transport> TypedAnyTransport<T> {
    /// Wrap a typed transport as a type-erased [`AnyTransport`].
    #[must_use]
    pub fn new(inner: Arc<T>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<T: Transport> AnyTransport for TypedAnyTransport<T> {
    fn kind(&self) -> &'static str {
        T::NAME
    }

    fn type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<T>()
    }

    fn as_any_arc(&self) -> Arc<dyn Any + Send + Sync> {
        self.inner.clone()
    }

    fn serve(
        self: Arc<Self>,
        shutdown: ShutdownToken,
    ) -> BoxFuture<'static, Result<(), TransportError>> {
        let inner = self.inner.clone();
        Box::pin(async move {
            inner
                .serve(shutdown)
                .await
                .map_err(|e| TransportError::Serve {
                    name: T::NAME.to_string(),
                    message: e.to_string(),
                })
        })
    }

    fn health(&self) -> BoxFuture<'static, HealthStatus> {
        let inner = self.inner.clone();
        Box::pin(async move { inner.health().await })
    }

    fn addr(&self) -> Option<String> {
        self.inner.addr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::lifecycle::ShutdownToken;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct EchoTransport {
        served: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Transport for EchoTransport {
        const NAME: &'static str = "transport.test.echo";
        type Config = serde_json::Value;
        type Error = std::io::Error;

        async fn serve(&self, shutdown: ShutdownToken) -> Result<(), Self::Error> {
            shutdown.wait().await;
            self.served.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn typed_any_transport_serves_until_shutdown() {
        let served = Arc::new(AtomicUsize::new(0));
        let transport: Arc<EchoTransport> = Arc::new(EchoTransport {
            served: served.clone(),
        });
        let any: Arc<dyn AnyTransport> = Arc::new(TypedAnyTransport::new(transport));
        let shutdown = ShutdownToken::new();
        let shutdown2 = shutdown.clone();
        let any2 = any.clone();
        let handle = tokio::spawn(async move { any2.serve(shutdown2).await });
        // Give the transport time to register.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        shutdown.signal_shutdown();
        let result = handle.await;
        assert!(result.is_ok());
        assert_eq!(served.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn typed_any_transport_kind_and_addr() {
        struct AddrTransport;
        #[async_trait]
        impl Transport for AddrTransport {
            const NAME: &'static str = "transport.test.addr";
            type Config = serde_json::Value;
            type Error = std::io::Error;
            async fn serve(&self, _s: ShutdownToken) -> Result<(), Self::Error> {
                Ok(())
            }
            fn addr(&self) -> Option<String> {
                Some("127.0.0.1:0".to_string())
            }
        }
        let any: Arc<dyn AnyTransport> = Arc::new(TypedAnyTransport::new(Arc::new(AddrTransport)));
        assert_eq!(any.kind(), "transport.test.addr");
        assert_eq!(any.addr().as_deref(), Some("127.0.0.1:0"));
    }
}
