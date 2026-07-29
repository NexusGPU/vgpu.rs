use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use erl::{DeviceController, DeviceControllerConfig};
use nvml_wrapper::Nvml;
use poem::{
    handler, web::Data, web::Json as PoemJson, web::Query as PoemQuery, EndpointExt, Route,
};
use serde::Deserialize;
use tokio::task::JoinHandle;
use tracing::{debug, error, info};
use utils::shared_memory::{
    erl_adapter::ErlSharedMemoryAdapter, handle::SharedMemoryHandle, DeviceConfig,
};

use api_types::{PodInfo, PodInfoResponse, ProcessInfo, ProcessInitResponse};

/// Local mode configuration
#[derive(Clone, Debug)]
pub struct LocalConfig {
    /// GPU device indices to manage
    pub devices: Vec<usize>,
    /// Target utilization (0.0 - 1.0)
    pub target_utilization: f64,
    /// Shared memory path
    pub shm_path: PathBuf,
    /// HTTP API port
    pub api_port: u16,
    /// Compute shard mode
    pub compute_shard: bool,
    /// Isolation level
    pub isolation: Option<String>,
    /// Update interval in milliseconds
    pub update_interval_ms: u64,
    /// Initial refill rate (tokens/sec)
    pub initial_rate: f64,
    /// ERL controller config
    pub controller_config: DeviceControllerConfig,
}

/// State for local API server
#[derive(Clone, Debug)]
struct LocalApiState {
    config: LocalConfig,
    gpu_uuids: Vec<String>,
}

/// Query parameters for pod info
#[derive(Debug, Deserialize)]
struct PodInfoQuery {
    #[allow(dead_code)]
    container_pid: Option<u32>,
    #[allow(dead_code)]
    container_name: Option<String>,
}

/// Query parameters for process init
#[derive(Debug, Deserialize)]
struct ProcessInitQuery {
    #[allow(dead_code)]
    container_pid: u32,
    #[allow(dead_code)]
    container_name: String,
}

/// Run hypervisor in local testing mode
pub async fn run_local_mode(config: LocalConfig) -> Result<()> {
    info!("Starting hypervisor local mode");
    info!("Target devices: {:?}", config.devices);
    info!(
        "Target utilization: {:.1}%",
        config.target_utilization * 100.0
    );
    info!("API port: {}", config.api_port);
    info!("Shared memory path: {}", config.shm_path.display());

    // 1. Initialize NVML
    let nvml = Nvml::init().context("Failed to initialize NVML")?;
    info!("NVML initialized successfully");

    // 2. Create shared memory
    let device_configs =
        create_device_configs(&nvml, &config.devices).context("Failed to create device configs")?;

    let shm_handle = SharedMemoryHandle::create(&config.shm_path, &device_configs)
        .context("Failed to create shared memory")?;
    let shm_handle = Arc::new(shm_handle);

    info!("Shared memory created: {}", config.shm_path.display());

    // 3. Start HTTP API server
    let api_state = LocalApiState {
        config: config.clone(),
        gpu_uuids: device_configs
            .iter()
            .map(|d| d.device_uuid.clone())
            .collect(),
    };

    let api_task: JoinHandle<Result<()>> = tokio::spawn(async move {
        start_local_api_server(config.api_port, api_state)
            .await
            .context("API server error")
    });

    // 4. Start ERL controller loop
    let erl_task: JoinHandle<Result<()>> = tokio::spawn(async move {
        run_erl_control_loop(nvml, shm_handle, config)
            .await
            .context("ERL controller error")
    });

    info!("Local mode started successfully");

    // Wait for either task to complete (or fail)
    tokio::select! {
        result = api_task => {
            match result {
                Ok(Ok(())) => info!("API server stopped"),
                Ok(Err(e)) => error!("API server error: {:#}", e),
                Err(e) => error!("API server task panic: {}", e),
            }
        }
        result = erl_task => {
            match result {
                Ok(Ok(())) => info!("ERL controller stopped"),
                Ok(Err(e)) => error!("ERL controller error: {:#}", e),
                Err(e) => error!("ERL controller task panic: {}", e),
            }
        }
    }

    Ok(())
}

/// Start HTTP API server for local mode
async fn start_local_api_server(port: u16, state: LocalApiState) -> Result<()> {
    let state = Arc::new(state);

    let app = Route::new()
        .at("/api/v1/pod", poem::get(handle_pod_info))
        .at("/api/v1/process", poem::post(handle_process_init))
        .data(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    info!("Local API server listening on {}", addr);

    poem::Server::new(poem::listener::TcpListener::bind(addr))
        .run(app)
        .await
        .context("API server error")?;

    Ok(())
}

/// Handle pod info requests
#[handler]
async fn handle_pod_info(
    state: Data<&Arc<LocalApiState>>,
    PoemQuery(_params): PoemQuery<PodInfoQuery>,
) -> PoemJson<PodInfoResponse> {
    debug!("Handling pod info request");

    PoemJson(PodInfoResponse {
        success: true,
        message: "Local mode - pod info".to_string(),
        pod_info: Some(PodInfo {
            pod_name: "local-pod".to_string(),
            namespace: "local".to_string(),
            gpu_uuids: state.gpu_uuids.clone(),
            compute_shard: state.config.compute_shard,
            isolation: state.config.isolation.clone(),
            auto_freeze: None,
            qos_level: None,
            tflops_limit: None,
            vram_limit: None,
        }),
    })
}

/// Handle process init requests
#[handler]
async fn handle_process_init(
    Data(_state): Data<&Arc<LocalApiState>>,
    PoemQuery(params): PoemQuery<ProcessInitQuery>,
) -> PoemJson<ProcessInitResponse> {
    debug!("Handling process init request");

    // In local mode, we don't track processes
    // Return a dummy host_pid
    PoemJson(ProcessInitResponse {
        success: true,
        message: "Local mode - process init".to_string(),
        process_info: Some(ProcessInfo {
            host_pid: std::process::id(),
            container_pid: params.container_pid,
            container_name: params.container_name,
            namespace: "local".to_string(),
            pod_name: "local-pod".to_string(),
        }),
    })
}

/// Run ERL controller loop
async fn run_erl_control_loop(
    nvml: Nvml,
    shm_handle: Arc<SharedMemoryHandle>,
    config: LocalConfig,
) -> Result<()> {
    info!("Starting ERL controller loop");

    // Create DeviceController for each device
    let mut controllers = Vec::new();
    for &device_idx in &config.devices {
        // Create separate backend for each controller
        let backend = ErlSharedMemoryAdapter::new(Arc::clone(&shm_handle));
        let controller =
            DeviceController::new(backend, device_idx, config.controller_config.clone())
                .map_err(|e| anyhow::anyhow!("{}", e))?;
        controllers.push((device_idx, controller));
        info!("Initialized ERL controller for device {}", device_idx);
    }

    // Main control loop
    let interval = Duration::from_millis(config.update_interval_ms);
    let mut last_update = Instant::now();

    info!("ERL control loop started (interval: {:?})", interval);

    loop {
        tokio::time::sleep(interval).await;

        let now = Instant::now();
        let delta = now.duration_since(last_update);
        last_update = now;

        let heartbeat = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        shm_handle.get_state().update_heartbeat(heartbeat);

        for (device_idx, controller) in &mut controllers {
            // Get current GPU utilization from NVML
            let device = match nvml.device_by_index(*device_idx as u32) {
                Ok(dev) => dev,
                Err(e) => {
                    error!(device = device_idx, error = %e, "Failed to get device");
                    continue;
                }
            };

            let utilization = match device.utilization_rates() {
                Ok(rates) => rates.gpu as f64 / 100.0,
                Err(e) => {
                    error!(device = device_idx, error = %e, "Failed to get utilization");
                    continue;
                }
            };

            // Update controller (which refills tokens)
            match controller.update(utilization, delta.as_secs_f64()) {
                Ok(state) => {
                    debug!(
                        device = device_idx,
                        util = %format!("{:.1}%", utilization * 100.0),
                        target = %format!("{:.1}%", config.target_utilization * 100.0),
                        rate = %format!("{:.1}/s", state.current_rate),
                        capacity = %format!("{:.1}", state.current_capacity),
                        "ERL controller update"
                    );
                }
                Err(e) => {
                    error!(device = device_idx, error = %e, "Failed to update controller");
                }
            }
        }
    }
}

/// Create device configs from NVML
fn create_device_configs(nvml: &Nvml, devices: &[usize]) -> Result<Vec<DeviceConfig>> {
    let mut configs = Vec::new();

    for &idx in devices {
        let device = nvml
            .device_by_index(idx as u32)
            .context(format!("Failed to get device {}", idx))?;

        let uuid = device
            .uuid()
            .context(format!("Failed to get UUID for device {}", idx))?;

        let memory_info = device
            .memory_info()
            .context(format!("Failed to get memory info for device {}", idx))?;

        let sm_count = device
            .num_cores()
            .context(format!("Failed to get SM count for device {}", idx))?;

        let config = DeviceConfig {
            device_idx: idx as u32,
            device_uuid: uuid.clone(),
            up_limit: 80, // Default 80% utilization limit
            mem_limit: memory_info.total,
            sm_count: sm_count as u32,
            max_thread_per_sm: 1536, // Typical value for most GPUs
            total_cuda_cores: sm_count as u32 * 64, // Rough estimate
        };

        info!(
            "Device {} configured: UUID={}, Memory={}GB, SMs={}",
            idx,
            uuid,
            memory_info.total / (1024 * 1024 * 1024),
            sm_count
        );

        configs.push(config);
    }

    Ok(configs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use erl::DeviceControllerConfig;

    #[test]
    fn test_local_config_creation() {
        let config = LocalConfig {
            devices: vec![0, 1],
            target_utilization: 0.75,
            shm_path: PathBuf::from("/tmp/test"),
            api_port: 8080,
            compute_shard: true,
            isolation: Some("soft".to_string()),
            update_interval_ms: 100,
            initial_rate: 50.0,
            controller_config: DeviceControllerConfig::default(),
        };

        assert_eq!(config.devices, vec![0, 1]);
        assert_eq!(config.target_utilization, 0.75);
        assert_eq!(config.api_port, 8080);
        assert!(config.compute_shard);
        assert_eq!(config.isolation, Some("soft".to_string()));
    }

    #[test]
    fn test_local_api_state_creation() {
        let config = LocalConfig {
            devices: vec![0],
            target_utilization: 0.5,
            shm_path: PathBuf::from("/tmp/test"),
            api_port: 8080,
            compute_shard: false,
            isolation: None,
            update_interval_ms: 100,
            initial_rate: 50.0,
            controller_config: DeviceControllerConfig::default(),
        };

        let gpu_uuids = vec!["GPU-test-uuid".to_string()];
        let state = LocalApiState {
            config: config.clone(),
            gpu_uuids: gpu_uuids.clone(),
        };

        assert_eq!(state.gpu_uuids, gpu_uuids);
        assert_eq!(state.config.devices, vec![0]);
    }

    #[test]
    fn test_pod_info_response_structure() {
        let config = LocalConfig {
            devices: vec![0],
            target_utilization: 0.7,
            shm_path: PathBuf::from("/tmp/test"),
            api_port: 8080,
            compute_shard: true,
            isolation: Some("hard".to_string()),
            update_interval_ms: 100,
            initial_rate: 50.0,
            controller_config: DeviceControllerConfig::default(),
        };

        let gpu_uuids = vec!["GPU-test-uuid".to_string()];

        let pod_info = PodInfo {
            pod_name: "local-pod".to_string(),
            namespace: "local".to_string(),
            gpu_uuids: gpu_uuids.clone(),
            compute_shard: config.compute_shard,
            isolation: config.isolation.clone(),
            auto_freeze: None,
            qos_level: None,
            tflops_limit: None,
            vram_limit: None,
        };

        assert_eq!(pod_info.pod_name, "local-pod");
        assert_eq!(pod_info.namespace, "local");
        assert_eq!(pod_info.gpu_uuids, gpu_uuids);
        assert!(pod_info.compute_shard);
        assert_eq!(pod_info.isolation, Some("hard".to_string()));
    }

    #[test]
    fn test_process_info_response_structure() {
        let process_info = ProcessInfo {
            host_pid: 12345,
            container_pid: 1,
            container_name: "test-container".to_string(),
            namespace: "local".to_string(),
            pod_name: "local-pod".to_string(),
        };

        assert_eq!(process_info.host_pid, 12345);
        assert_eq!(process_info.container_pid, 1);
        assert_eq!(process_info.container_name, "test-container");
        assert_eq!(process_info.namespace, "local");
        assert_eq!(process_info.pod_name, "local-pod");
    }
}
