use core::time::Duration;
use std::env;
use std::fs;

use api_types::{AutoFreezeConfig, PodInfoResponse, ProcessInitResponse};
use error_stack::{Report, ResultExt};
use reqwest::blocking::Client;

const SERVICE_ACCOUNT_TOKEN_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone)]
pub struct PodConfig {
    pub gpu_uuids: Vec<String>,
    pub compute_shard: bool,
    pub auto_freeze: Option<AutoFreezeConfig>,
}

#[derive(Debug, Clone)]
pub struct ProcessConfig {
    pub host_pid: u32,
}

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

#[tracing::instrument(skip(hypervisor_ip, hypervisor_port), fields(url))]
pub fn get_worker_config(
    hypervisor_ip: impl AsRef<str>,
    hypervisor_port: impl AsRef<str>,
) -> Result<PodConfig, Report<ConfigError>> {
    request_pod_info(hypervisor_ip.as_ref(), hypervisor_port.as_ref())
}

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
        .pod_info
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
        auto_freeze: pod_info.auto_freeze,
    })
}

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

    let process_info = process_response.process_info.ok_or_else(|| {
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

fn build_http_client() -> Client {
    Client::builder()
        .timeout(DEFAULT_REQUEST_TIMEOUT)
        .build()
        .expect("should build HTTP client")
}

pub fn get_hypervisor_config() -> Option<(String, String)> {
    let hypervisor_ip = env::var("HYPERVISOR_IP").ok()?;
    let hypervisor_port = env::var("HYPERVISOR_PORT").ok()?;
    Some((hypervisor_ip, hypervisor_port))
}

pub fn get_container_name() -> Option<String> {
    env::var("CONTAINER_NAME").ok()
}

pub fn parse_duration(duration_str: &str) -> Result<Duration, Report<ConfigError>> {
    let duration_str = duration_str.trim();

    if duration_str.is_empty() {
        return Err(Report::new(ConfigError::InvalidResponse).attach("Duration string is empty"));
    }

    humantime::parse_duration(duration_str)
        .change_context(ConfigError::InvalidResponse)
        .attach_with(|| format!("Failed to parse duration: {duration_str}"))
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;

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

    #[test]
    fn parse_duration_seconds() {
        let result = super::parse_duration("30s");
        assert!(result.is_ok(), "Should parse seconds successfully");
        assert_eq!(
            result.unwrap().as_secs(),
            30,
            "Should parse 30 seconds correctly"
        );
    }

    #[test]
    fn parse_duration_minutes() {
        let result = super::parse_duration("5m");
        assert!(result.is_ok(), "Should parse minutes successfully");
        assert_eq!(
            result.unwrap().as_secs(),
            300,
            "Should parse 5 minutes correctly"
        );
    }

    #[test]
    fn parse_duration_hours() {
        let result = super::parse_duration("2h");
        assert!(result.is_ok(), "Should parse hours successfully");
        assert_eq!(
            result.unwrap().as_secs(),
            7200,
            "Should parse 2 hours correctly"
        );
    }

    #[test]
    fn parse_duration_complex() {
        let result = super::parse_duration("1h 30m");
        assert!(result.is_ok(), "Should parse complex duration successfully");
        assert_eq!(
            result.unwrap().as_secs(),
            5400,
            "Should parse '1h 30m' correctly"
        );
    }

    #[test]
    fn parse_duration_empty_string() {
        let result = super::parse_duration("");
        assert!(result.is_err(), "Should fail on empty string");
    }

    #[test]
    fn parse_duration_invalid_format() {
        let result = super::parse_duration("invalid");
        assert!(result.is_err(), "Should fail on invalid format");
    }
}
