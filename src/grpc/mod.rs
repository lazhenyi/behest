//! gRPC server adapter for the agent runtime.
//!
//! Compiles protobuf definitions from `src/grpc/proto/` and
//! implements the generated service traits for admin, agent,
//! artifact, auth, chat, compaction, context, embedding, event,
//! provider, run, session, snapshot, tool, and usage services.
//!
//! Enable with `features = ["server"]`.

// Generated protobuf code.
#[allow(clippy::all, clippy::pedantic, clippy::restriction)]
#[allow(missing_docs)]
pub mod pb {
    tonic::include_proto!("agent.v1");
}

pub mod admin;
pub mod agent_grpc;
pub mod artifact;
pub mod auth;
pub mod chat;
pub mod compaction;
pub mod context;
pub mod embedding;
pub mod event;
pub mod provider;
pub mod run;
pub mod session;
pub mod snapshot;
pub mod state;
pub mod tool;
pub mod usage;

/// Converts a [`chrono::DateTime`] to a protobuf timestamp.
///
/// # Panics
///
/// Panics if the nanosecond component exceeds [`i32::MAX`].
pub(crate) fn to_prost_timestamp(dt: chrono::DateTime<chrono::Utc>) -> prost_types::Timestamp {
    prost_types::Timestamp {
        seconds: dt.timestamp(),
        nanos: i32::try_from(dt.timestamp_subsec_nanos()).unwrap_or(0),
    }
}

/// Converts a crate [`Error`](crate::Error) to a gRPC [`Status`] with
/// semantically appropriate status codes.
pub(crate) fn error_to_status(err: crate::Error) -> tonic::Status {
    use crate::Error;
    use tonic::{Code, Status};

    match err {
        Error::Provider(ref e) => provider_error_to_status(e),
        Error::Tool(ref e) => tool_error_to_status(e),
        Error::Context(ref e) => context_error_to_status(e),
        Error::Storage(ref e) => storage_error_to_status(e),
        Error::Config(msg) => Status::new(Code::InvalidArgument, msg),
    }
}

/// Maps a [`ProviderError`](crate::ProviderError) to a gRPC [`Status`].
///
/// Converts provider-specific errors (authentication, rate limiting, timeout,
/// transport, etc.) into semantically appropriate gRPC status codes.
pub(super) fn provider_error_to_status(err: &crate::ProviderError) -> tonic::Status {
    use crate::ProviderError;
    use tonic::{Code, Status};

    let code = match err {
        ProviderError::Authentication { .. } => Code::Unauthenticated,
        ProviderError::BadRequest { .. } => Code::InvalidArgument,
        ProviderError::RateLimited { .. } => Code::ResourceExhausted,
        ProviderError::Timeout { .. } => Code::DeadlineExceeded,
        ProviderError::Overloaded { .. } | ProviderError::Transport { .. } => Code::Unavailable,
        ProviderError::Unsupported { .. } => Code::Unimplemented,
        ProviderError::Decode { .. } => Code::Internal,
        ProviderError::Provider { status, .. } => match status {
            Some(400) => Code::InvalidArgument,
            Some(401) => Code::Unauthenticated,
            Some(403) => Code::PermissionDenied,
            Some(404) => Code::NotFound,
            Some(429) => Code::ResourceExhausted,
            Some(500..=599) => Code::Unavailable,
            _ => Code::Internal,
        },
    };
    Status::new(code, err.to_string())
}

fn tool_error_to_status(err: &crate::ToolError) -> tonic::Status {
    use crate::ToolError;
    use tonic::{Code, Status};

    let code = match err {
        ToolError::NotFound { .. } => Code::NotFound,
        ToolError::Execution { .. } => Code::Internal,
        ToolError::InvalidArguments { .. } => Code::InvalidArgument,
        ToolError::NotImplemented { .. } => Code::Unimplemented,
    };
    Status::new(code, err.to_string())
}

fn context_error_to_status(err: &crate::ContextError) -> tonic::Status {
    use crate::ContextError;
    use tonic::{Code, Status};

    let code = match err {
        ContextError::AdapterFailed { .. } => Code::Internal,
        ContextError::InvalidInput { .. } => Code::InvalidArgument,
        ContextError::AdapterNotFound { .. } => Code::NotFound,
    };
    Status::new(code, err.to_string())
}

/// Converts a [`RuntimeError`](crate::runtime::RuntimeError) to a gRPC [`Status`]
/// with semantically appropriate status codes.
pub(crate) fn runtime_error_to_status(err: &crate::runtime::RuntimeError) -> tonic::Status {
    use crate::runtime::RuntimeError;
    use tonic::{Code, Status};

    let code = match err {
        RuntimeError::ProviderNotFound(_)
        | RuntimeError::SessionNotFound(_)
        | RuntimeError::RunNotFound(_) => Code::NotFound,
        RuntimeError::InvalidRunState { .. } | RuntimeError::DoomLoopDetected { .. } => {
            Code::FailedPrecondition
        }
        RuntimeError::IterationLimitExceeded(_)
        | RuntimeError::TokenBudgetExceeded { .. }
        | RuntimeError::ContextOverflow { .. } => Code::ResourceExhausted,
        RuntimeError::SessionBusy(_) => Code::Aborted,
        RuntimeError::ToolTimeout { .. } => Code::DeadlineExceeded,
        RuntimeError::Provider(e) => return provider_error_to_status(e),
        RuntimeError::Context(e) => return context_error_to_status(e),
        RuntimeError::Storage(e) => return storage_error_to_status(e),
        RuntimeError::Tool(e) => return tool_error_to_status(e),
        RuntimeError::InputRejected { .. } => Code::InvalidArgument,
        RuntimeError::RecoveryFailed(_) | RuntimeError::InputAdmissionFailed(_) => Code::Internal,
    };
    Status::new(code, err.to_string())
}

fn storage_error_to_status(err: &crate::StorageError) -> tonic::Status {
    use crate::StorageError;
    use tonic::{Code, Status};

    let code = match err {
        StorageError::NotFound { .. } => Code::NotFound,
        StorageError::ConnectionFailed { .. } => Code::Unavailable,
        StorageError::SerializationFailed { .. }
        | StorageError::BackendError { .. }
        | StorageError::MigrationFailed { .. }
        | StorageError::DataCorruption { .. } => Code::Internal,
    };
    Status::new(code, err.to_string())
}
