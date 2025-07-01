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
    fn pod_not_found_error_displays_correctly() {
        // Arrange
        let error = ApiError::PodNotFound {
            pod_name: "test-pod".to_string(),
            namespace: "default".to_string(),
        };

        // Act
        let display_message = format!("{error}");

        // Assert
        assert_eq!(
            display_message, "Pod not found: test-pod in namespace default",
            "Pod not found error should display pod name and namespace"
        );
    }

    #[test]
    fn server_error_displays_correctly() {
        // Arrange
        let error = ApiError::ServerError {
            message: "Database connection failed".to_string(),
        };

        // Act
        let display_message = format!("{error}");

        // Assert
        assert_eq!(
            display_message, "Server error: Database connection failed",
            "Server error should display the provided message"
        );
    }

    #[test]
    fn authentication_failed_error_displays_correctly() {
        // Arrange
        let error = ApiError::AuthenticationFailed {
            reason: "Invalid signature".to_string(),
        };

        // Act
        let display_message = format!("{error}");

        // Assert
        assert_eq!(
            display_message, "Authentication failed: Invalid signature",
            "Authentication failed error should display the reason"
        );
    }

    #[test]
    fn invalid_jwt_token_error_displays_correctly() {
        // Arrange
        let error = ApiError::InvalidJwtToken {
            reason: "Malformed payload".to_string(),
        };

        // Act
        let display_message = format!("{error}");

        // Assert
        assert_eq!(
            display_message, "Invalid JWT token: Malformed payload",
            "Invalid JWT token error should display the reason"
        );
    }

    #[test]
    fn missing_auth_header_error_displays_correctly() {
        // Arrange
        let error = ApiError::MissingAuthHeader;

        // Act
        let display_message = format!("{error}");

        // Assert
        assert_eq!(
            display_message, "Missing authorization header",
            "Missing auth header error should display fixed message"
        );
    }

    #[test]
    fn api_error_implements_error_trait() {
        // Arrange
        let error = ApiError::ServerError {
            message: "test error".to_string(),
        };

        // Act - Test that it can be used as a trait object
        let error_trait: &dyn Error = &error;
        let debug_output = format!("{error_trait:?}");

        // Assert
        assert!(
            debug_output.contains("ServerError"),
            "Error should be debuggable as trait object"
        );
    }

    #[test]
    fn api_error_can_be_debugged() {
        // Arrange
        let error = ApiError::PodNotFound {
            pod_name: "debug-pod".to_string(),
            namespace: "debug-ns".to_string(),
        };

        // Act
        let debug_output = format!("{error:?}");

        // Assert
        assert!(
            debug_output.contains("PodNotFound"),
            "Debug output should contain variant name"
        );
        assert!(
            debug_output.contains("debug-pod"),
            "Debug output should contain pod name"
        );
        assert!(
            debug_output.contains("debug-ns"),
            "Debug output should contain namespace"
        );
    }
}
