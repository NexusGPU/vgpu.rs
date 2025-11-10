use core::time::Duration;
use std::env;
use std::fs;

use api_types::{PodInfoResponse, ProcessInitResponse};
use error_stack::{Report, ResultExt};
use reqwest::blocking::Client;

/// Service account token path in Kubernetes pods
const SERVICE_ACCOUNT_TOKEN_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";

/// Default HTTP request timeout
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Pod configuration returned from GET /api/v1/pod
#[derive(Debug, Clone)]
pub struct PodConfig {
    /// GPU UUIDs assigned to the pod
    pub gpu_uuids: Vec<String>,
    /// Whether compute sharding is enabled
    pub compute_shard: bool,
}

/// Process configuration returned from POST /api/v1/process
#[derive(Debug, Clone)]
pub struct ProcessConfig {
    /// Host PID of the process
    pub host_pid: u32,
}

/// Error types for configuration operations
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Failed to read service account token")]
    TokenRead,
    #[error("Service account token is empty")]
    EmptyToken,
    #[error("HTTP request failed")]
    HttpRequest,
    #[error("Invalid response from hypervisor")]
    InvalidResponse,
    #[error("Missing required data in response")]
    MissingData,
    #[error("JSON parsing failed")]
    JsonParsing,
}

/// Get pod configuration from hypervisor
///
/// Makes a GET request to /api/v1/pod to retrieve GPU assignments and compute shard status.
///
/// # Errors
///
/// Returns [`ConfigError::TokenRead`] if the service account token cannot be read.
/// Returns [`ConfigError::HttpRequest`] if the HTTP request fails.
/// Returns [`ConfigError::JsonParsing`] if the response cannot be parsed.
/// Returns [`ConfigError::MissingData`] if the response is missing required data.
#[tracing::instrument(skip(hypervisor_ip, hypervisor_port), fields(url))]
pub fn get_worker_config(
    hypervisor_ip: impl AsRef<str>,
    hypervisor_port: impl AsRef<str>,
) -> Result<PodConfig, Report<ConfigError>> {
    request_pod_info(hypervisor_ip.as_ref(), hypervisor_port.as_ref())
}

/// Initialize worker process with hypervisor
///
/// Makes a POST request to /api/v1/process to register the worker process
/// and retrieve the host PID.
///
/// # Errors
///
/// Returns [`ConfigError::TokenRead`] if the service account token cannot be read.
/// Returns [`ConfigError::HttpRequest`] if the HTTP request fails.
/// Returns [`ConfigError::JsonParsing`] if the response cannot be parsed.
/// Returns [`ConfigError::MissingData`] if the response is missing required data.
#[tracing::instrument(skip(hypervisor_ip, hypervisor_port), fields(url, container_name))]
pub fn init_worker(
    hypervisor_ip: impl AsRef<str>,
    hypervisor_port: impl AsRef<str>,
    container_name: impl AsRef<str>,
) -> Result<ProcessConfig, Report<ConfigError>> {
    request_process_init(
        hypervisor_ip.as_ref(),
        hypervisor_port.as_ref(),
        container_name.as_ref(),
    )
}

/// Request pod information from hypervisor API
#[tracing::instrument(level = "debug", fields(container_pid))]
fn request_pod_info(
    hypervisor_ip: &str,
    hypervisor_port: &str,
) -> Result<PodConfig, Report<ConfigError>> {
    let token = read_service_account_token()?;
    let container_pid = std::process::id();
    let container_name = env::var("CONTAINER_NAME").unwrap_or_default();

    tracing::Span::current().record("container_pid", container_pid);

    let client = build_http_client();
    let url = format!("http://{hypervisor_ip}:{hypervisor_port}/api/v1/pod");

    tracing::debug!(url = %url, "Requesting pod information");

    let request_start = std::time::Instant::now();

    let mut request = client
        .get(&url)
        .bearer_auth(&token)
        .query(&[("container_pid", container_pid.to_string())]);

    if !container_name.is_empty() {
        request = request.query(&[("container_name", container_name)]);
    }

    let response = request
        .send()
        .change_context(ConfigError::HttpRequest)
        .attach_with(|| format!("Failed to send GET request to {url}"))?;

    if !response.status().is_success() {
        return Err(Report::new(ConfigError::HttpRequest).attach(format!(
            "HTTP request failed with status: {}",
            response.status()
        )));
    }

    let pod_response: PodInfoResponse = response
        .json()
        .change_context(ConfigError::JsonParsing)
        .attach("Failed to parse PodInfoResponse")?;

    if !pod_response.success {
        return Err(Report::new(ConfigError::InvalidResponse)
            .attach(format!("Hypervisor API error: {}", pod_response.message)));
    }

    let pod_info = pod_response
        .data
        .ok_or_else(|| Report::new(ConfigError::MissingData).attach("No pod data in response"))?;

    let request_duration = request_start.elapsed();

    tracing::info!(
        duration_ms = request_duration.as_millis(),
        gpu_count = pod_info.gpu_uuids.len(),
        compute_shard = pod_info.compute_shard,
        "Successfully retrieved pod configuration"
    );

    Ok(PodConfig {
        gpu_uuids: pod_info.gpu_uuids,
        compute_shard: pod_info.compute_shard,
    })
}

/// Request process initialization from hypervisor API
#[tracing::instrument(level = "debug", fields(container_pid, host_pid))]
fn request_process_init(
    hypervisor_ip: &str,
    hypervisor_port: &str,
    container_name: &str,
) -> Result<ProcessConfig, Report<ConfigError>> {
    let token = read_service_account_token()?;
    let container_pid = std::process::id();

    tracing::Span::current().record("container_pid", container_pid);

    let client = build_http_client();
    let url = format!("http://{hypervisor_ip}:{hypervisor_port}/api/v1/process");

    tracing::debug!(url = %url, container_name = %container_name, "Initializing worker process");

    let request_start = std::time::Instant::now();

    let response = client
        .post(&url)
        .bearer_auth(&token)
        .query(&[
            ("container_pid", container_pid.to_string()),
            ("container_name", container_name.to_string()),
        ])
        .send()
        .change_context(ConfigError::HttpRequest)
        .attach_with(|| format!("Failed to send POST request to {url}"))?;

    if !response.status().is_success() {
        return Err(Report::new(ConfigError::HttpRequest).attach(format!(
            "HTTP request failed with status: {}",
            response.status()
        )));
    }

    let process_response: ProcessInitResponse = response
        .json()
        .change_context(ConfigError::JsonParsing)
        .attach("Failed to parse ProcessInitResponse")?;

    if !process_response.success {
        return Err(Report::new(ConfigError::InvalidResponse).attach(format!(
            "Hypervisor API error: {}",
            process_response.message
        )));
    }

    let process_info = process_response.data.ok_or_else(|| {
        Report::new(ConfigError::MissingData).attach("No process data in response")
    })?;

    let request_duration = request_start.elapsed();

    tracing::Span::current().record("host_pid", process_info.host_pid);

    tracing::info!(
        duration_ms = request_duration.as_millis(),
        host_pid = process_info.host_pid,
        "Successfully initialized worker process"
    );

    Ok(ProcessConfig {
        host_pid: process_info.host_pid,
    })
}

/// Read and validate Kubernetes service account token
#[tracing::instrument(level = "debug")]
fn read_service_account_token() -> Result<String, Report<ConfigError>> {
    let token = fs::read_to_string(SERVICE_ACCOUNT_TOKEN_PATH)
        .change_context(ConfigError::TokenRead)
        .attach_with(|| format!("Failed to read token from {SERVICE_ACCOUNT_TOKEN_PATH}"))?;

    let token = token.trim();

    if token.is_empty() {
        return Err(Report::new(ConfigError::EmptyToken).attach("Service account token is empty"));
    }

    tracing::debug!("Successfully read service account token");

    Ok(token.to_string())
}

/// Build HTTP client with default configuration
fn build_http_client() -> Client {
    Client::builder()
        .timeout(DEFAULT_REQUEST_TIMEOUT)
        .build()
        .expect("should build HTTP client")
}

/// Get hypervisor configuration from environment variables
///
/// Returns `None` if either `HYPERVISOR_IP` or `HYPERVISOR_PORT` is not set.
pub fn get_hypervisor_config() -> Option<(String, String)> {
    let hypervisor_ip = env::var("HYPERVISOR_IP").ok()?;
    let hypervisor_port = env::var("HYPERVISOR_PORT").ok()?;
    Some((hypervisor_ip, hypervisor_port))
}

/// Get container name from environment variable
///
/// Returns `None` if `CONTAINER_NAME` is not set.
pub fn get_container_name() -> Option<String> {
    env::var("CONTAINER_NAME").ok()
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use serial_test::serial;
    use tempfile::NamedTempFile;

    use super::*;

    fn read_token_from_path(path: &str) -> Result<String, Report<ConfigError>> {
        let token = fs::read_to_string(path)
            .change_context(ConfigError::TokenRead)
            .attach_with(|| format!("Failed to read token from {path}"))?;

        let token = token.trim();

        if token.is_empty() {
            return Err(
                Report::new(ConfigError::EmptyToken).attach("Service account token is empty")
            );
        }

        Ok(token.to_string())
    }

    #[test]
    fn read_token_successfully() {
        let mut temp_file = NamedTempFile::new().expect("should create temp file");
        let test_token = "test-token-123";
        writeln!(temp_file, "{test_token}").expect("should write to temp file");

        let token_path = temp_file.path().to_str().unwrap();
        let result = read_token_from_path(token_path);

        assert!(result.is_ok(), "Should successfully read token");
        assert_eq!(
            result.unwrap(),
            test_token,
            "Token should match expected value"
        );
    }

    #[test]
    fn read_token_with_whitespace() {
        let mut temp_file = NamedTempFile::new().expect("should create temp file");
        let test_token = "test-token-with-spaces";
        writeln!(temp_file, "  {test_token}  \n").expect("should write to temp file");

        let token_path = temp_file.path().to_str().unwrap();
        let result = read_token_from_path(token_path);

        assert!(result.is_ok(), "Should successfully read token");
        assert_eq!(result.unwrap(), test_token, "Token should be trimmed");
    }

    #[test]
    fn token_file_not_found() {
        let non_existent_path = "/path/that/does/not/exist/token";
        let result = read_token_from_path(non_existent_path);

        assert!(result.is_err(), "Should return error for non-existent file");

        let error = result.unwrap_err();
        assert!(
            matches!(error.current_context(), ConfigError::TokenRead),
            "Should return TokenRead error"
        );
    }

    #[test]
    fn empty_token_file() {
        let temp_file = NamedTempFile::new().expect("should create temp file");

        let token_path = temp_file.path().to_str().unwrap();
        let result = read_token_from_path(token_path);

        assert!(result.is_err(), "Should return error for empty file");

        let error = result.unwrap_err();
        assert!(
            matches!(error.current_context(), ConfigError::EmptyToken),
            "Should return EmptyToken error"
        );
    }

    #[test]
    fn whitespace_only_token_file() {
        let mut temp_file = NamedTempFile::new().expect("should create temp file");
        writeln!(temp_file, "   \n\t  \n").expect("should write to temp file");

        let token_path = temp_file.path().to_str().unwrap();
        let result = read_token_from_path(token_path);

        assert!(
            result.is_err(),
            "Should return error for whitespace-only file"
        );

        let error = result.unwrap_err();
        assert!(
            matches!(error.current_context(), ConfigError::EmptyToken),
            "Should return EmptyToken error"
        );
    }

    #[test]
    #[serial]
    fn hypervisor_config_with_env_vars() {
        env::set_var("HYPERVISOR_IP", "127.0.0.1");
        env::set_var("HYPERVISOR_PORT", "8080");

        let config = get_hypervisor_config();

        assert!(
            config.is_some(),
            "Should return config when env vars are set"
        );
        let (ip, port) = config.unwrap();
        assert_eq!(ip, "127.0.0.1");
        assert_eq!(port, "8080");

        env::remove_var("HYPERVISOR_IP");
        env::remove_var("HYPERVISOR_PORT");
    }

    #[test]
    #[serial]
    fn hypervisor_config_without_env_vars() {
        env::remove_var("HYPERVISOR_IP");
        env::remove_var("HYPERVISOR_PORT");

        let config = get_hypervisor_config();

        assert!(
            config.is_none(),
            "Should return None when env vars are not set"
        );
    }

    #[test]
    #[serial]
    fn container_name_with_env_var() {
        env::set_var("CONTAINER_NAME", "test-container");

        let name = get_container_name();

        assert!(name.is_some(), "Should return name when env var is set");
        assert_eq!(name.unwrap(), "test-container");

        env::remove_var("CONTAINER_NAME");
    }

    #[test]
    #[serial]
    fn container_name_without_env_var() {
        env::remove_var("CONTAINER_NAME");

        let name = get_container_name();

        assert!(name.is_none(), "Should return None when env var is not set");
    }
}
