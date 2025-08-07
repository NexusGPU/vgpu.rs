//! HTTP API for querying pod resource information
//!
//! This module provides a REST API for clients to query pod resource allocations
//! including TFLOPS and VRAM usage. Authentication is handled via JWT tokens
//! containing Kubernetes pod and namespace information.
//!
//! # API Endpoints
//!
//! - `GET /api/v1/pods` - Get resource information for the authenticated pod
//!
//! # Authentication
//!
//! All requests must include a JWT token in the Authorization header:
//! ```
//! Authorization: Bearer <JWT_TOKEN>
//! ```
//!
//! The JWT payload must contain Kubernetes information in the following format:
//! ```json
//! {
//!   "kubernetes.io": {
//!     "namespace": "default",
//!     "pod": {
//!       "name": "my-pod",
//!       "uid": "pod-uuid"
//!     },
//!     "node": {
//!       "name": "node-name",
//!       "uid": "node-uuid"
//!     },
//!     "serviceaccount": {
//!       "name": "default",
//!       "uid": "sa-uuid"
//!     }
//!   },
//!   "nbf": 1751311081,
//!   "sub": "user-subject"
//! }
//! ```

use core::error::Error;

// Re-export common API types
pub use api_types::PodInfo;
pub use api_types::PodInfoResponse;
pub use api_types::ProcessInfo;
pub use api_types::ProcessInitResponse;

// Re-export other necessary types
pub use api_types::JwtAuthConfig;
pub use api_types::JwtPayload;
pub use api_types::LimiterCommand;
pub use api_types::LimiterCommandResponse;
pub use api_types::LimiterCommandType;

pub mod auth;
pub mod handlers;
pub mod server;

/// API errors
#[derive(Debug, derive_more::Display)]
pub enum ApiError {
    #[display("Pod not found: {pod_name} in namespace {namespace}")]
    #[allow(dead_code)]
    PodNotFound { pod_name: String, namespace: String },
    #[display("Server error: {message}")]
    ServerError { message: String },
    #[display("Authentication failed: {reason}")]
    #[allow(dead_code)]
    AuthenticationFailed { reason: String },
    #[display("Invalid JWT token: {reason}")]
    InvalidJwtToken { reason: String },
    #[display("Missing authorization header")]
    #[allow(dead_code)]
    MissingAuthHeader,
}

impl Error for ApiError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_error_display_formatting() {
        let pod_not_found = ApiError::PodNotFound {
            pod_name: "my-pod".to_string(),
            namespace: "default".to_string(),
        };
        assert_eq!(
            pod_not_found.to_string(),
            "Pod not found: my-pod in namespace default"
        );

        let server_error = ApiError::ServerError {
            message: "Internal server error".to_string(),
        };
        assert_eq!(
            server_error.to_string(),
            "Server error: Internal server error"
        );

        let auth_failed = ApiError::AuthenticationFailed {
            reason: "Invalid credentials".to_string(),
        };
        assert_eq!(
            auth_failed.to_string(),
            "Authentication failed: Invalid credentials"
        );

        let invalid_jwt = ApiError::InvalidJwtToken {
            reason: "Token expired".to_string(),
        };
        assert_eq!(invalid_jwt.to_string(), "Invalid JWT token: Token expired");

        let missing_header = ApiError::MissingAuthHeader;
        assert_eq!(missing_header.to_string(), "Missing authorization header");
    }
}
