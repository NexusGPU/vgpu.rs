use std::collections::HashSet;
use std::env;
use std::fs;

use api_types::WorkerQueryResponse;
use error_stack::Report;
use error_stack::ResultExt;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;
use reqwest::blocking::Client;

use crate::limiter::DeviceConfig;

/// Configuration result containing device configs and host PID
#[derive(Debug, Clone)]
pub struct DeviceConfigResult {
    pub device_configs: Vec<DeviceConfig>,
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
    #[error("NVML error")]
    Nvml(NvmlError),
}

pub fn get_device_configs(
    nvml: &Nvml,
    hypervisor_ip: &str,
    hypervisor_port: &str,
) -> Result<DeviceConfigResult, Report<ConfigError>> {
    let container_name = env::var("CONTAINER_NAME").unwrap_or_else(|_| {
        tracing::warn!("CONTAINER_NAME environment variable not set, using empty string");
        String::new()
    });

    fetch_device_configs_from_hypervisor(nvml, hypervisor_ip, hypervisor_port, &container_name)
}

/// Fetch device configurations from hypervisor API using Kubernetes service account token
fn fetch_device_configs_from_hypervisor(
    nvml: &Nvml,
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
    let device_configs = if let (Some(tflops_limit), Some(vram_limit)) =
        (worker_info.tflops_limit, worker_info.vram_limit)
    {
        let config = DeviceConfig::new(
            0,                   // Default to device 0, may need to be configurable
            tflops_limit as u32, // Convert TFLOPS to up_limit
            vram_limit,
            2048, // Default total CUDA cores, will be calculated by hypervisor
        );
        vec![config]
    } else {
        tracing::warn!("Pod resource info does not contain required limits");
        vec![]
    };

    tracing::info!(
        "Successfully fetched {} device configurations from hypervisor",
        device_configs.len()
    );

    if let Some(gpu_uuids) = &worker_info.gpu_uuids {
        if !gpu_uuids.is_empty() {
            let lower_case_uuids: HashSet<_> = gpu_uuids.iter().map(|u| u.to_lowercase()).collect();
            let device_count = nvml
                .device_count()
                .map_err(|e| Report::new(ConfigError::Nvml(e)))?;

            let mut device_indices = Vec::new();
            for i in 0..device_count {
                let device = nvml
                    .device_by_index(i)
                    .map_err(|e| Report::new(ConfigError::Nvml(e)))?;
                let uuid = device
                    .uuid()
                    .map_err(|e| Report::new(ConfigError::Nvml(e)))?;
                if lower_case_uuids.contains(&uuid.to_lowercase()) {
                    device_indices.push(i.to_string());
                }
            }

            if !device_indices.is_empty() {
                let visible_devices = device_indices.join(",");
                tracing::info!(
                    "Setting CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES to {}",
                    &visible_devices
                );
                env::set_var("CUDA_VISIBLE_DEVICES", &visible_devices);
                env::set_var("NVIDIA_VISIBLE_DEVICES", &visible_devices);
            }
        }
    }

    Ok(DeviceConfigResult {
        device_configs,
        host_pid: worker_info.host_pid,
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
