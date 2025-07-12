use core::error::Error;

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
