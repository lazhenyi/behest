//! gRPC server configuration section.

use serde::{Deserialize, Serialize};

/// gRPC server configuration.
///
/// Controls listen address and TLS certificate paths (batch 3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcConfig {
    /// Socket address to listen on.
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        Self {
            listen_addr: default_listen_addr(),
        }
    }
}

fn default_listen_addr() -> String {
    "[::1]:50051".to_owned()
}
