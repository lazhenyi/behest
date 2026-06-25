//! Bearer token authentication interceptor for gRPC services.

use tonic::{Request, Status};

/// Interceptor that validates Bearer tokens on incoming gRPC requests.
///
/// When an `expected_token` is configured, every request must include
/// an `authorization` metadata header with value `Bearer <token>`.
/// Pass `None` at construction to disable authentication entirely.
#[derive(Clone)]
pub struct AuthInterceptor {
    expected_token: Option<String>,
}

impl AuthInterceptor {
    /// Creates a new interceptor with an optional expected bearer token.
    ///
    /// Pass `None` to disable authentication for all requests.
    #[must_use]
    pub fn new(expected_token: Option<String>) -> Self {
        Self { expected_token }
    }
}

impl tonic::service::Interceptor for AuthInterceptor {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        let Some(ref expected) = self.expected_token else {
            return Ok(request);
        };

        let auth = request
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));

        match auth {
            Some(token) if token == expected => Ok(request),
            Some(_) => Err(Status::unauthenticated("invalid bearer token")),
            None => Err(Status::unauthenticated(
                "missing authorization header with Bearer token",
            )),
        }
    }
}
