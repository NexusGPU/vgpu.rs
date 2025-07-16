//! Limiter Coordinator Module

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use nvml_wrapper::error::NvmlError;
use tokio::task::JoinHandle;
use tokio::time::interval;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;
use utils::shared_memory::DeviceConfig;
use utils::shared_memory::ThreadSafeSharedMemoryManager;

/// Tracks device usage at the pod level.
#[derive(Debug, Clone)]
pub struct PodDeviceUsage {
    pub device_configs: Vec<DeviceConfig>,
    /// Information about containers in this pod that are using the device: container_name -> (container_pid, host_pid)
    pub active_containers: HashMap<String, (u32, u32)>,
}

impl PodDeviceUsage {
    fn new(device_configs: Vec<DeviceConfig>) -> Self {
        Self {
            device_configs,
            active_containers: HashMap::new(),
        }
    }

    fn add_container(&mut self, container_name: String, container_pid: u32, host_pid: u32) {
        self.active_containers
            .insert(container_name, (container_pid, host_pid));
    }

    fn remove_container(&mut self, container_name: &str) -> bool {
        self.active_containers.remove(container_name);
        self.active_containers.is_empty()
    }

    /// Gets all host_pids in the pod.
    pub fn get_host_pids(&self) -> Vec<u32> {
        self.active_containers
            .values()
            .map(|(_, host_pid)| *host_pid)
            .collect()
    }
}

/// GPU utilization statistics.
#[derive(Debug, Default, Clone, Copy)]
struct Utilization {
    /// Utilization of the current process.
    user_current: u32,
    /// Total system utilization.
    sys_current: u32,
}

/// Normalization function for codec utilization.
const fn codec_normalize(x: u32) -> u32 {
    x * 85 / 100
}

/// Limiter coordinator.
pub struct LimiterCoordinator {
    /// Shared memory manager.
    shared_memory_manager: Arc<ThreadSafeSharedMemoryManager>,
    /// Active pod device usage: pod_identifier -> PodDeviceUsage
    active_pods: Arc<RwLock<HashMap<String, PodDeviceUsage>>>,
    /// Monitoring task handles for each device: device_idx -> JoinHandle
    device_watcher_tasks: RwLock<HashMap<u32, JoinHandle<()>>>,
    /// Monitoring interval.
    watch_interval: Duration,
    /// Number of GPU devices.
    device_count: u32,
}

impl LimiterCoordinator {
    pub fn new(watch_interval: Duration, device_count: u32) -> Self {
        Self {
            shared_memory_manager: Arc::new(ThreadSafeSharedMemoryManager::new()),
            active_pods: Arc::new(RwLock::new(HashMap::new())),
            device_watcher_tasks: RwLock::new(HashMap::new()),
            watch_interval,
            device_count,
        }
    }

    /// Run the coordinator with cancellation support
    pub async fn run(&self, cancellation_token: CancellationToken) {
        tracing::info!(
            "Starting LimiterCoordinator with {} GPU devices",
            self.device_count
        );

        // Start monitoring tasks
        self.start_watcher_with_cancellation(cancellation_token.clone())
            .await;

        // Wait for cancellation
        cancellation_token.cancelled().await;

        tracing::info!("LimiterCoordinator received cancellation signal, stopping all tasks");

        // Stop all monitoring tasks
        self.stop_all_tasks().await;

        tracing::info!("LimiterCoordinator stopped");
    }

    /// Stop all monitoring tasks
    async fn stop_all_tasks(&self) {
        let mut watcher_tasks = self.device_watcher_tasks.write().unwrap();
        for (device_idx, task) in watcher_tasks.drain() {
            tracing::debug!("Stopping watcher task for device {}", device_idx);
            task.abort();
        }
    }

    /// Starts a monitoring task for each GPU device with cancellation support
    async fn start_watcher_with_cancellation(&self, cancellation_token: CancellationToken) {
        let active_pods = self.active_pods.clone();

        // Start a monitoring task for each GPU device.
        for device_idx in 0..self.device_count {
            let watch_interval = self.watch_interval;
            let active_pods = active_pods.clone();
            let token = cancellation_token.clone();

            let task = tokio::spawn(async move {
                let mut device_states: HashMap<String, (u64, i32)> = HashMap::new(); // pod_identifier -> (last_seen_timestamp, share)
                let mut interval_timer = interval(watch_interval);

                info!(
                    device_idx = device_idx,
                    "Starting device watcher task for GPU device"
                );

                loop {
                    tokio::select! {
                        _ = interval_timer.tick() => {
                            // Continue with monitoring logic
                        }
                        _ = token.cancelled() => {
                            info!(device_idx = device_idx, "Device watcher task cancelled");
                            break;
                        }
                    }

                    // Get all pods using this device.
                    let pods_for_device: Vec<(String, PodDeviceUsage)> = {
                        let pods = active_pods.read().unwrap();
                        pods.iter()
                            .filter(|(_, usage)| {
                                usage
                                    .device_configs
                                    .iter()
                                    .any(|config| config.device_idx == device_idx)
                            })
                            .map(|(name, usage)| (name.clone(), usage.clone()))
                            .collect()
                    };

                    if pods_for_device.is_empty() {
                        debug!(device_idx = device_idx, "No pods using this device");
                        continue;
                    }

                    // Monitor each pod.
                    for (pod_identifier, pod_usage) in pods_for_device {
                        let host_pids = pod_usage.get_host_pids();

                        if host_pids.is_empty() {
                            debug!(pod_identifier = %pod_identifier, device_idx = device_idx, "No active processes to monitor");
                            continue;
                        }

                        for device_config in pod_usage.device_configs {
                            let (last_seen_timestamp, current_share) = device_states
                                .entry(pod_identifier.clone())
                                .or_insert((0, device_config.total_cuda_cores as i32));

                            let utilization = match Self::get_pod_utilization_with_retry(
                                device_config.device_idx,
                                &host_pids,
                                *last_seen_timestamp,
                                Duration::from_millis(100),
                            )
                            .await
                            {
                                Ok((utilization, timestamp)) => {
                                    *last_seen_timestamp = timestamp;
                                    Some(utilization)
                                }
                                Err(e) => {
                                    error!(
                                        pod_identifier = %pod_identifier,
                                        device_idx = device_idx,
                                        error = %e,
                                        "Failed to get GPU utilization"
                                    );
                                    None
                                }
                            };

                            // Update pod memory usage in shared memory
                            let memory_used =
                                match Self::get_pod_memory_usage(device_idx, &host_pids).await {
                                    Ok(memory_used) => Some(memory_used),
                                    Err(e) => {
                                        error!(
                                            pod_identifier = %pod_identifier,
                                            device_idx = device_idx,
                                            error = %e,
                                            "Failed to get pod memory usage"
                                        );
                                        None
                                    }
                                };
                            let timestamp = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .expect("Failed to get timestamp")
                                .as_secs();
                            // Calculate the new share value.
                            let new_share = utilization.map(|util| {
                                calculate_delta(&device_config, util.user_current, *current_share)
                            });
                            if let Err(e) = Self::update_shared_memory_state(
                                &pod_identifier,
                                &device_config.device_uuid,
                                memory_used,
                                new_share,
                                timestamp,
                            )
                            .await
                            {
                                tracing::error!(
                                    pod_identifier = %pod_identifier,
                                    device_idx = device_idx,
                                    error = %e,
                                    "Failed to update shared memory state"
                                );
                            }
                        }
                    }
                }

                info!(device_idx = device_idx, "Device watcher task completed");
            });

            // Store the task handle.
            let mut watcher_tasks = self.device_watcher_tasks.write().expect("poisoned");
            watcher_tasks.insert(device_idx, task);
        }

        info!(
            device_count = self.device_count,
            "Started device watcher tasks for all GPU devices"
        );
    }

    /// Registers a device with the coordinator.
    pub fn register_device(
        &self,
        pod_identifier: &str,
        container_name: &str,
        container_pid: u32,
        host_pid: u32,
        configs: Vec<DeviceConfig>,
    ) -> Result<()> {
        // Use the pod_identifier as the shared memory identifier.
        let pod_identifier = pod_identifier.to_string();

        // Create the shared memory.
        self.shared_memory_manager
            .create_or_get_shared_memory(&pod_identifier, &configs)?;

        let device_count = configs.len();
        // Update the pod device usage.
        {
            let mut active_pods = self.active_pods.write().unwrap();
            let pod_usage = active_pods
                .entry(pod_identifier.clone())
                .or_insert_with(|| PodDeviceUsage::new(configs));

            pod_usage.add_container(container_name.to_string(), container_pid, host_pid);
        }

        info!(
            pod_identifier = %pod_identifier,
            container_name = %container_name,
            container_pid = container_pid,
            host_pid = host_pid,
            device_count = device_count,
            "Registered device to coordinator"
        );

        Ok(())
    }

    /// Unregisters a device from the coordinator.
    pub fn unregister_device(
        &self,
        pod_identifier: &str,
        container_name: &str,
        container_pid: u32,
    ) -> Result<()> {
        let pod_identifier = pod_identifier.to_string();

        let should_cleanup = {
            let mut active_pods = self.active_pods.write().unwrap();
            if let Some(pod_usage) = active_pods.get_mut(&pod_identifier) {
                let is_empty = pod_usage.remove_container(container_name);
                if is_empty {
                    active_pods.remove(&pod_identifier);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };

        if should_cleanup {
            // Clean up the shared memory.
            self.shared_memory_manager.cleanup(&pod_identifier)?;

            info!(
                pod_identifier = %pod_identifier,
                "Cleaned up pod shared memory"
            );
        }

        info!(
            pod_identifier = %pod_identifier,
            container_name = %container_name,
            container_pid = container_pid,
            "Unregistered device from coordinator"
        );

        Ok(())
    }

    /// Gets pod-level utilization data with retry logic.
    async fn get_pod_utilization_with_retry(
        device_idx: u32,
        host_pids: &[u32],
        last_seen_timestamp: u64,
        retry_interval: Duration,
    ) -> Result<(Utilization, u64)> {
        const MAX_RETRIES: u32 = 5;
        const BACKOFF_MULTIPLIER: u32 = 2;

        for retry_count in 0..MAX_RETRIES {
            match Self::get_pod_gpu_utilization(device_idx, host_pids, last_seen_timestamp).await {
                Ok(Some((util, timestamp))) => return Ok((util, timestamp)),
                Ok(None) => {
                    return Ok((Utilization::default(), last_seen_timestamp));
                }
                Err(e) => {
                    warn!(
                        device_idx = device_idx,
                        retry_count = retry_count,
                        error = %e,
                        "Failed to get GPU utilization, retrying"
                    );

                    if retry_count < MAX_RETRIES - 1 {
                        sleep(retry_interval * BACKOFF_MULTIPLIER).await;
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "Failed to get GPU utilization after {} retries",
            MAX_RETRIES
        ))
    }

    /// Gets the GPU memory usage of all processes in a pod directly via NVML.
    async fn get_pod_memory_usage(device_idx: u32, host_pids: &[u32]) -> Result<u64> {
        let nvml = nvml_wrapper::Nvml::init().context("Failed to initialize NVML")?;
        tokio::task::spawn_blocking({
            let host_pids = host_pids.to_vec();
            move || -> Result<u64> {
                let device = nvml
                    .device_by_index(device_idx)
                    .context("Failed to get device by index")?;

                // Get all running processes
                let process_info = device
                    .running_compute_processes()
                    .context("Failed to get running compute processes")?;

                // Calculate total memory usage for all processes in this pod
                let mut total_memory = 0u64;
                for pi in process_info {
                    if host_pids.contains(&pi.pid) {
                        match pi.used_gpu_memory {
                            nvml_wrapper::enums::device::UsedGpuMemory::Used(bytes) => {
                                total_memory += bytes;
                            }
                            nvml_wrapper::enums::device::UsedGpuMemory::Unavailable => {
                                // Ignore unavailable memory information
                            }
                        }
                    }
                }

                Ok(total_memory)
            }
        })
        .await
        .context("Blocking task failed")?
    }

    /// Updates the pod memory usage in shared memory.
    async fn update_shared_memory_state(
        pod_identifier: &str,
        device_uuid: &str,
        memory_used: Option<u64>,
        new_share: Option<i32>,
        timestamp: u64,
    ) -> Result<()> {
        tokio::task::spawn_blocking({
            let pod_identifier = pod_identifier.to_string();
            let device_uuid = device_uuid.to_string();

            move || -> Result<()> {
                use utils::shared_memory::SharedMemoryHandle;
                let handle = SharedMemoryHandle::open(&pod_identifier)
                    .context("Failed to open shared memory")?;
                let state = handle.get_state();

                // Update the memory usage in shared memory
                if let Some(device) = state.devices.write().get_mut(&device_uuid) {
                    if let Some(memory_used) = memory_used {
                        device.update_pod_memory_used(memory_used);
                    }
                    if let Some(new_share) = new_share {
                        // Get current state
                        let total_cuda_cores = device.get_total_cores() as i32;
                        // Apply adjustment
                        let adjustment = new_share.max(0).min(total_cuda_cores);
                        device.fetch_add_available_cores(adjustment);
                    }
                } else {
                    error!(
                        pod_identifier = %pod_identifier,
                        device_uuid = %device_uuid,
                        "Device not found in shared memory"
                    );
                }
                state.update_heartbeat(timestamp);
                Ok(())
            }
        })
        .await
        .context("Blocking task failed")?
    }

    /// Gets the GPU utilization of all processes in a pod directly via NVML.
    async fn get_pod_gpu_utilization(
        device_idx: u32,
        host_pids: &[u32],
        last_seen_timestamp: u64,
    ) -> Result<Option<(Utilization, u64)>> {
        // Execute the NVML call in a blocking task.
        tokio::task::spawn_blocking({
            let host_pids = host_pids.to_vec();
            move || -> Result<Option<(Utilization, u64)>> {
                // Initialize NVML.
                let nvml = nvml_wrapper::Nvml::init().context("Failed to initialize NVML")?;

                let mut newest_timestamp_candidate = last_seen_timestamp;
                let dev = nvml
                    .device_by_index(device_idx)
                    .context("Failed to get device by index")?;

                // Get process utilization samples.
                let process_utilization_samples =
                    dev.process_utilization_stats(last_seen_timestamp);

                if let Err(NvmlError::NotFound) = process_utilization_samples {
                    return Ok(None);
                }

                let process_utilization_samples = process_utilization_samples
                    .context("Failed to get process utilization stats")?;
                // Initialize utilization counters.
                let mut current = Utilization::default();
                let mut valid = false;

                // Process each utilization sample.
                for sample in process_utilization_samples {
                    // Skip old samples.
                    if sample.timestamp < last_seen_timestamp {
                        continue;
                    }

                    // Collect the maximum valid timestamp.
                    if sample.timestamp > newest_timestamp_candidate {
                        newest_timestamp_candidate = sample.timestamp;
                    }

                    // Mark that we have valid data.
                    valid = true;

                    // Calculate codec utilization.
                    let codec_util = codec_normalize(sample.enc_util + sample.dec_util);

                    // Add to system-level utilization.
                    current.sys_current += sample.sm_util;
                    current.sys_current += codec_util;

                    // If it's a process in our pod, add to user process utilization.
                    if host_pids.contains(&sample.pid) {
                        current.user_current += sample.sm_util;
                        current.user_current += codec_util;
                    }
                }

                // Return None if no valid data, otherwise return the utilization.
                if !valid {
                    Ok(None)
                } else {
                    Ok(Some((current, newest_timestamp_candidate)))
                }
            }
        })
        .await
        .context("Blocking task failed")?
    }
}

impl Drop for LimiterCoordinator {
    fn drop(&mut self) {
        // Stop all monitoring tasks
        let mut watcher_tasks = self.device_watcher_tasks.write().unwrap();
        for (_, task) in watcher_tasks.drain() {
            task.abort();
        }

        info!("LimiterCoordinator dropped, all watcher tasks stopped");
    }
}

/// Calculate core adjustment value
fn calculate_delta(device: &DeviceConfig, user_current: u32, share: i32) -> i32 {
    let up_limit = device.up_limit as i32;
    let total_cuda_cores = device.total_cuda_cores as i32;

    // Calculate utilization difference
    let utilization_diff = if (up_limit - user_current as i32).abs() < 5 {
        5
    } else {
        (up_limit - user_current as i32).abs()
    };

    // Calculate increment
    let increment = calculate_increment(device, utilization_diff);

    // Determine adjustment direction
    if user_current <= up_limit as u32 {
        // Utilization below limit, increase share
        if share + increment > total_cuda_cores {
            total_cuda_cores
        } else {
            share + increment
        }
    } else {
        // Utilization above limit, decrease share
        if share - increment < 0 {
            0
        } else {
            share - increment
        }
    }
}

/// Calculate the increment value for delta adjustment
fn calculate_increment(device: &DeviceConfig, utilization_diff: i32) -> i32 {
    let up_limit = device.up_limit as i32;
    let mut increment =
        device.sm_count as i32 * device.sm_count as i32 * device.max_thread_per_sm as i32 / 256
            * utilization_diff
            / 10;

    // Apply additional scaling when difference is large
    if utilization_diff > up_limit / 2 {
        increment = increment * utilization_diff * 2 / (up_limit + 1);
    }

    increment
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use utils::shared_memory::DeviceConfig;

    use super::*;

    /// Create test device configuration
    fn create_test_device_config(device_idx: u32) -> DeviceConfig {
        DeviceConfig {
            device_idx,
            device_uuid: "test-device-uuid".to_string(),
            up_limit: 80,
            mem_limit: 8 * 1024 * 1024 * 1024, // 8GB
            total_cuda_cores: 2048,
            sm_count: 10,
            max_thread_per_sm: 1024,
        }
    }

    /// Create test coordinator
    fn create_test_coordinator() -> LimiterCoordinator {
        LimiterCoordinator::new(Duration::from_millis(100), 1) // Assume there is only one GPU device
    }

    /// Create unique test pod identifier
    fn create_unique_test_pod_identifier(test_name: &str) -> String {
        use std::time::SystemTime;
        use std::time::UNIX_EPOCH;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{test_name}-{timestamp}")
    }

    #[tokio::test]
    async fn test_single_pod_single_container() {
        let coordinator = create_test_coordinator();
        let config = create_test_device_config(0);
        let pod_identifier = create_unique_test_pod_identifier("single-pod-single-container");

        // Register device
        coordinator
            .register_device(&pod_identifier, "container-1", 1001, 12345, vec![
                config.clone()
            ])
            .unwrap();

        // Unregister device
        coordinator
            .unregister_device(&pod_identifier, "container-1", 1001)
            .unwrap();
    }

    #[tokio::test]
    async fn test_multiple_containers() {
        let coordinator = create_test_coordinator();
        let config = create_test_device_config(0);
        let pod_identifier = create_unique_test_pod_identifier("multiple-containers");

        // Register multiple containers
        coordinator
            .register_device(&pod_identifier, "container-1", 1001, 12345, vec![
                config.clone()
            ])
            .unwrap();
        coordinator
            .register_device(&pod_identifier, "container-2", 1002, 12346, vec![
                config.clone()
            ])
            .unwrap();

        // Unregister containers
        coordinator
            .unregister_device(&pod_identifier, "container-1", 1001)
            .unwrap();
        coordinator
            .unregister_device(&pod_identifier, "container-2", 1002)
            .unwrap();
    }
}
