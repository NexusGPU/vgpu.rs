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
