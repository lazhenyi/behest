//! gRPC server adapter for the agent runtime.
//!
//! Compiles protobuf definitions from `src/grpc/proto/` and
//! implements the generated service traits.
//!
//! Enable with `features = ["server"]`.

// Generated protobuf code.
#[allow(clippy::all, clippy::pedantic, clippy::restriction)]
#[allow(missing_docs)]
pub mod pb {
    tonic::include_proto!("agent.v1");
}

pub mod event;
pub mod provider;
pub mod run;
pub mod session;
pub mod state;
pub mod tool;
pub mod usage;
