use std::env;
use std::fs;

use api_types::WorkerQueryResponse;
use error_stack::Report;
use error_stack::ResultExt;
use reqwest::blocking::Client;

/// Configuration for a CUDA device
#[derive(Debug, Clone, Default)]
pub(crate) struct DeviceLimit {
    /// Utilization percentage limit (0-100)
    #[allow(dead_code)]
    pub up_limit: u32,
    /// Memory limit in bytes (0 means no limit)
    #[allow(dead_code)]
    pub mem_limit: u64,
}

/// Configuration result containing device configs and host PID
#[derive(Debug, Clone)]
pub struct DeviceConfigResult {
    #[allow(dead_code)]
    pub device_limit: DeviceLimit,
    pub gpu_uuids: Vec<String>,
    pub host_pid: u32,
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

pub fn get_device_configs(
    hypervisor_ip: &str,
    hypervisor_port: &str,
) -> Result<DeviceConfigResult, Report<ConfigError>> {
    let container_name = env::var("CONTAINER_NAME").unwrap_or_else(|_| {
        tracing::warn!("CONTAINER_NAME environment variable not set, using empty string");
        String::new()
    });

    fetch_device_configs_from_hypervisor(hypervisor_ip, hypervisor_port, &container_name)
}

/// Fetch device configurations from hypervisor API using Kubernetes service account token
fn fetch_device_configs_from_hypervisor(
    hypervisor_ip: &str,
    hypervisor_port: &str,
    container_name: &str,
) -> Result<DeviceConfigResult, Report<ConfigError>> {
    // Get Kubernetes service account token
    let token = get_k8s_service_account_token().change_context(ConfigError::OidcAuth)?;

    // Get container PID (this process's PID)
    let container_pid = std::process::id();

    // Create HTTP client
    let client = Client::new();
    let url = format!("http://{hypervisor_ip}:{hypervisor_port}/api/v1/worker");

    tracing::debug!("Fetching pod information from: {}", url);

    // Record request start time
    let request_start = std::time::Instant::now();

    // Make HTTP request with Bearer token and container_pid query parameter
    let pid_string = container_pid.to_string();
    let query_params: &[(&str, &str)] = &[
        ("container_pid", &pid_string),
        ("container_name", container_name),
    ];
    let response = client
        .get(&url)
        .bearer_auth(&token)
        .query(query_params)
        .send()
        .change_context(ConfigError::HttpRequest)?;

    if !response.status().is_success() {
        tracing::error!("HTTP request failed with status: {}", response.status());
        return Err(Report::new(ConfigError::HttpRequest));
    }

    // Parse response
    let worker_info_response: WorkerQueryResponse =
        response.json().change_context(ConfigError::JsonParsing)?;

    if !worker_info_response.success {
        tracing::error!(
            "Hypervisor API returned error: {}",
            worker_info_response.message
        );
        return Err(Report::new(ConfigError::HttpRequest));
    }

    let worker_info = worker_info_response.data.ok_or_else(|| {
        tracing::error!("No pod data returned from hypervisor API");
        Report::new(ConfigError::JsonParsing)
    })?;

    // Convert PodResourceInfo to DeviceConfig
    // Note: We need to extract device configuration from the pod resource info
    // For now, we'll create a single device config based on the resource limits
    let device_limit = if let (Some(tflops_limit), Some(vram_limit)) =
        (worker_info.tflops_limit, worker_info.vram_limit)
    {
        DeviceLimit {
            up_limit: tflops_limit as u32,
            mem_limit: vram_limit,
        }
    } else {
        tracing::warn!("Pod resource info does not contain required limits");
        DeviceLimit::default()
    };

    let request_duration = request_start.elapsed();
    tracing::info!(
        "Successfully fetched device limit from hypervisor: {device_limit:?}, request_time: {request_duration:?}",
    );

    Ok(DeviceConfigResult {
        device_limit,
        host_pid: worker_info.host_pid,
        gpu_uuids: worker_info.gpu_uuids.unwrap_or_default(),
    })
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

/// Get Kubernetes service account token from a custom path (for testing)
#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

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
