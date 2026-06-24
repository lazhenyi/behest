//! gRPC server configuration section.

use serde::{Deserialize, Serialize};

/// gRPC server configuration.
///
/// Controls listen address, TLS, authentication, and concurrency limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcConfig {
    /// Socket address to listen on.
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,

    /// TLS configuration. When `None`, the server runs in plaintext mode.
    #[serde(default)]
    pub tls: Option<GrpcTlsConfig>,

    /// Bearer token for authentication. When `None`, no auth is required.
    #[serde(default)]
    pub auth_token: Option<String>,

    /// Maximum number of concurrent runs. When `None`, no limit is enforced.
    #[serde(default)]
    pub max_concurrent_runs: Option<usize>,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        Self {
            listen_addr: default_listen_addr(),
            tls: None,
            auth_token: None,
            max_concurrent_runs: None,
        }
    }
}

/// TLS configuration for the gRPC server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcTlsConfig {
    /// Path to the server certificate (PEM).
    pub cert_path: String,
    /// Path to the server private key (PEM).
    pub key_path: String,
    /// Path to the client CA certificate (PEM) for mTLS.
    #[serde(default)]
    pub client_ca_path: Option<String>,
}

fn default_listen_addr() -> String {
    "[::1]:50051".to_owned()
}
