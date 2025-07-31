use std::env;
use std::fs;

use anyhow::Result;
use api_types::PodInfoResponse;
use api_types::ProcessInitResponse;
use error_stack::Report;
use error_stack::ResultExt;
use reqwest::blocking::Client;
use reqwest::Method;

/// Result containing device configuration
#[derive(Debug, Clone)]
pub struct DeviceConfigResult {
    pub gpu_uuids: Vec<String>,
    pub host_pid: u32,
}

/// Worker operation type
#[derive(Debug, Clone, Copy)]
pub enum WorkerOperation {
    /// Get pod information (GET request) - no container_name needed
    GetInfo,
    /// Initialize worker process (POST request) - requires container_name
    Initialize,
}

impl WorkerOperation {
    fn http_method(&self) -> Method {
        match self {
            WorkerOperation::GetInfo => Method::GET,
            WorkerOperation::Initialize => Method::POST,
        }
    }

    fn requires_container_name(&self) -> bool {
        matches!(self, WorkerOperation::Initialize)
    }

    fn endpoint_path(&self) -> &'static str {
        match self {
            WorkerOperation::GetInfo => "/api/v1/pod",
            WorkerOperation::Initialize => "/api/v1/process",
        }
    }
}

/// Error types for configuration operations
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("HTTP request failed")]
    HttpRequest,
    #[error("OIDC authentication failed")]
    OidcAuth,
    #[error("JSON parsing failed")]
    JsonParsing,
}

/// Get worker device configuration from hypervisor (GET request)
pub fn get_worker_config(
    hypervisor_ip: &str,
    hypervisor_port: &str,
) -> Result<DeviceConfigResult, Report<ConfigError>> {
    request_worker_info(
        hypervisor_ip,
        hypervisor_port,
        None,
        WorkerOperation::GetInfo,
    )
}

/// Initialize worker with hypervisor (POST request)
pub fn init_worker(
    hypervisor_ip: &str,
    hypervisor_port: &str,
    container_name: &str,
) -> Result<DeviceConfigResult, Report<ConfigError>> {
    // Initialize the process to get host_pid and GPU UUIDs
    let process_config = request_worker_info(
        hypervisor_ip,
        hypervisor_port,
        Some(container_name),
        WorkerOperation::Initialize,
    )?;

    // Return the process configuration directly
    Ok(process_config)
}

/// Request worker information from hypervisor API using Kubernetes service account token
fn request_worker_info(
    hypervisor_ip: &str,
    hypervisor_port: &str,
    container_name: Option<&str>,
    operation: WorkerOperation,
) -> Result<DeviceConfigResult, Report<ConfigError>> {
    // Get Kubernetes service account token
    let token = get_k8s_service_account_token().change_context(ConfigError::OidcAuth)?;

    // Get container PID (this process's PID)
    let container_pid = std::process::id();

    // Create HTTP client
    let client = Client::new();
    let url = format!(
        "http://{hypervisor_ip}:{hypervisor_port}{}",
        operation.endpoint_path()
    );

    tracing::debug!(
        "Requesting worker info from: {} (operation: {:?})",
        url,
        operation
    );

    // Record request start time
    let request_start = std::time::Instant::now();

    // Prepare query parameters based on operation type
    let pid_string = container_pid.to_string();
    let mut query_params = vec![("container_pid", pid_string.as_str())];

    // Only add container_name for operations that require it
    let container_name_for_request = if operation.requires_container_name() {
        container_name.unwrap_or_else(|| {
            tracing::warn!("container_name not provided for {:?} operation", operation);
            ""
        })
    } else {
        // For GET requests, get from environment if available
        &env::var("CONTAINER_NAME").unwrap_or_else(|_| {
            tracing::debug!("CONTAINER_NAME environment variable not set for GET request");
            String::new()
        })
    };

    if !container_name_for_request.is_empty() {
        query_params.push(("container_name", container_name_for_request));
    }

    let response = client
        .request(operation.http_method(), &url)
        .bearer_auth(&token)
        .query(&query_params)
        .send()
        .change_context(ConfigError::HttpRequest)?;

    if !response.status().is_success() {
        tracing::error!("HTTP request failed with status: {}", response.status());
        return Err(Report::new(ConfigError::HttpRequest));
    }

    let request_duration = request_start.elapsed();

    // Handle different response types based on operation
    let result = match operation {
        WorkerOperation::GetInfo => {
            // Parse PodInfoResponse for GET requests
            let pod_response: PodInfoResponse =
                response.json().change_context(ConfigError::JsonParsing)?;

            if !pod_response.success {
                tracing::error!("Hypervisor API returned error: {}", pod_response.message);
                return Err(Report::new(ConfigError::HttpRequest));
            }

            let pod_info = pod_response.data.ok_or_else(|| {
                tracing::error!("No pod data returned from hypervisor API");
                Report::new(ConfigError::JsonParsing)
            })?;

            // For GetInfo, we don't have host_pid, so use container_pid as placeholder
            DeviceConfigResult {
                host_pid: container_pid, // Use container_pid as placeholder for GetInfo
                gpu_uuids: pod_info.gpu_uuids,
            }
        }
        WorkerOperation::Initialize => {
            // Parse ProcessInitResponse for POST requests
            let process_response: ProcessInitResponse =
                response.json().change_context(ConfigError::JsonParsing)?;

            if !process_response.success {
                tracing::error!(
                    "Hypervisor API returned error: {}",
                    process_response.message
                );
                return Err(Report::new(ConfigError::HttpRequest));
            }

            let process_info = process_response.data.ok_or_else(|| {
                tracing::error!("No process data returned from hypervisor API");
                Report::new(ConfigError::JsonParsing)
            })?;

            DeviceConfigResult {
                host_pid: process_info.host_pid,
                // there is no gpu_uuids in the process response
                gpu_uuids: vec![],
            }
        }
    };

    tracing::info!(
        "Successfully processed hypervisor response: operation: {:?}, request_time: {:?}, host_pid: {}",
        operation,
        request_duration,
        result.host_pid
    );

    Ok(result)
}

/// Get Kubernetes service account token from the pod
fn get_k8s_service_account_token() -> Result<String, Report<ConfigError>> {
    // Read the service account token directly from the mounted file
    const SERVICE_ACCOUNT_TOKEN_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";

    tracing::debug!(
        "Reading Kubernetes service account token from: {}",
        SERVICE_ACCOUNT_TOKEN_PATH
    );

    let token = fs::read_to_string(SERVICE_ACCOUNT_TOKEN_PATH).map_err(|e| {
        tracing::error!("Failed to read service account token: {}", e);
        Report::new(ConfigError::OidcAuth)
    })?;

    let token = token.trim().to_string();

    if token.is_empty() {
        tracing::error!("Service account token is empty");
        return Err(Report::new(ConfigError::OidcAuth));
    }

    tracing::debug!("Successfully read Kubernetes service account token");

    Ok(token)
}

/// Get hypervisor configuration from environment variables
pub fn get_hypervisor_config() -> Option<(String, String)> {
    let hypervisor_ip = env::var("HYPERVISOR_IP").ok()?;
    let hypervisor_port = env::var("HYPERVISOR_PORT").ok()?;
    Some((hypervisor_ip, hypervisor_port))
}

/// Get container name from environment variable
pub fn get_container_name() -> Option<String> {
    env::var("CONTAINER_NAME").ok()
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    fn get_k8s_service_account_token_from_path(path: &str) -> Result<String, Report<ConfigError>> {
        tracing::debug!("Reading Kubernetes service account token from: {}", path);

        let token = fs::read_to_string(path).map_err(|e| {
            tracing::error!("Failed to read service account token: {}", e);
            Report::new(ConfigError::OidcAuth)
        })?;

        let token = token.trim().to_string();

        if token.is_empty() {
            tracing::error!("Service account token is empty");
            return Err(Report::new(ConfigError::OidcAuth));
        }

        tracing::debug!("Successfully read Kubernetes service account token");

        Ok(token)
    }

    #[test]
    fn test_read_token_successfully() {
        // Create a temporary file with a token
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let test_token = "test-token-123";
        writeln!(temp_file, "{test_token}").expect("Failed to write to temp file");

        let token_path = temp_file.path().to_str().unwrap();
        let result = get_k8s_service_account_token_from_path(token_path);

        assert!(result.is_ok(), "Should successfully read token");
        assert_eq!(
            result.unwrap(),
            test_token,
            "Token should match expected value"
        );
    }

    #[test]
    fn test_read_token_with_whitespace() {
        // Create a temporary file with a token that has surrounding whitespace
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let test_token = "test-token-with-spaces";
        writeln!(temp_file, "  {test_token}  \n").expect("Failed to write to temp file");

        let token_path = temp_file.path().to_str().unwrap();
        let result = get_k8s_service_account_token_from_path(token_path);

        assert!(result.is_ok(), "Should successfully read token");
        assert_eq!(result.unwrap(), test_token, "Token should be trimmed");
    }

    #[test]
    fn test_token_file_not_found() {
        let non_existent_path = "/path/that/does/not/exist/token";
        let result = get_k8s_service_account_token_from_path(non_existent_path);

        assert!(result.is_err(), "Should return error for non-existent file");

        let error = result.unwrap_err();
        assert!(
            matches!(error.current_context(), ConfigError::OidcAuth),
            "Should return OidcAuth error"
        );
    }

    #[test]
    fn test_empty_token_file() {
        // Create a temporary file with empty content
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        // Don't write anything to the file, leaving it empty

        let token_path = temp_file.path().to_str().unwrap();
        let result = get_k8s_service_account_token_from_path(token_path);

        assert!(result.is_err(), "Should return error for empty file");

        let error = result.unwrap_err();
        assert!(
            matches!(error.current_context(), ConfigError::OidcAuth),
            "Should return OidcAuth error"
        );
    }

    #[test]
    fn test_whitespace_only_token_file() {
        // Create a temporary file with only whitespace
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(temp_file, "   \n\t  \n").expect("Failed to write to temp file");

        let token_path = temp_file.path().to_str().unwrap();
        let result = get_k8s_service_account_token_from_path(token_path);

        assert!(
            result.is_err(),
            "Should return error for whitespace-only file"
        );

        let error = result.unwrap_err();
        assert!(
            matches!(error.current_context(), ConfigError::OidcAuth),
            "Should return OidcAuth error"
        );
    }
}
