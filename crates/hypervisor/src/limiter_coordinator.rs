//! Limiter Coordinator Module

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::error::NvmlError;
use tokio::task::JoinHandle;
use tokio::time::interval;
use tokio_util::sync::CancellationToken;
use tracing::debug;
use tracing::error;
use tracing::info;
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

/// Per-process utilization data
#[derive(Debug, Clone, Copy)]
pub struct ProcessUtilization {
    pub sm_util: u32,
    pub codec_util: u32,
}

/// Pod-level utilization summary
#[derive(Debug, Default, Clone, Copy)]
pub struct PodUtilization {
    pub total_utilization: u32, // Sum of SM + codec utilization for all processes in pod
}

/// Complete snapshot of device state including utilization and memory
#[derive(Debug, Clone)]
pub struct DeviceSnapshot {
    // pid -> utilization
    pub process_utilizations: HashMap<u32, ProcessUtilization>,
    // pid -> memory used
    pub process_memories: HashMap<u32, u64>,
    // timestamp of the snapshot
    pub timestamp: u64,
}

impl DeviceSnapshot {
    /// Calculate pod-level utilization from device snapshot
    pub fn get_pod_utilization(&self, pids: &[u32]) -> PodUtilization {
        let mut total_utilization = 0u32;

        for pid in pids {
            if let Some(process_util) = self.process_utilizations.get(pid) {
                total_utilization += process_util.sm_util + process_util.codec_util;
            }
        }

        PodUtilization { total_utilization }
    }

    /// Calculate pod-level memory usage from device snapshot
    pub fn get_pod_memory(&self, pids: &[u32]) -> u64 {
        pids.iter()
            .filter_map(|pid| self.process_memories.get(pid))
            .sum()
    }
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
                let mut interval_timer = interval(watch_interval);
                let mut last_seen_timestamp = 0u64; // Track timestamp at device level

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

                    // Get all pods using this device
                    let pods_for_device: Vec<(String, PodDeviceUsage)> = {
                        let pods = active_pods.read().expect("poisoned");
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

                    // **Get complete device snapshot ONCE per iteration**
                    let device_snapshot =
                        match Self::get_device_snapshot(device_idx, last_seen_timestamp).await {
                            Ok(Some(snapshot)) => {
                                // Update timestamp for next iteration
                                last_seen_timestamp = snapshot.timestamp;
                                snapshot
                            }
                            Ok(None) => {
                                debug!(device_idx = device_idx, "No device data available");
                                continue;
                            }
                            Err(e) => {
                                error!(
                                    device_idx = device_idx,
                                    error = %e,
                                    "Failed to get device snapshot"
                                );
                                continue;
                            }
                        };

                    // **Process each pod using the snapshot**
                    for (pod_identifier, pod_usage) in pods_for_device {
                        let host_pids = pod_usage.get_host_pids();

                        if host_pids.is_empty() {
                            debug!(pod_identifier = %pod_identifier, device_idx = device_idx, "No active processes to monitor");
                            continue;
                        }
                        let device_config = pod_usage
                            .device_configs
                            .iter()
                            .find(|config| config.device_idx == device_idx)
                            .expect("Device config must exist for filtered pod");
                        // **Stateless operation: Read current share from shared memory**
                        let current_share = match Self::read_current_share_from_shared_memory(
                            &pod_identifier,
                            &device_config.device_uuid,
                        )
                        .await
                        {
                            Ok(share) => share,
                            Err(e) => {
                                error!(
                                    pod_identifier = %pod_identifier,
                                    device_idx = device_idx,
                                    device_uuid = %device_config.device_uuid,
                                    error = %e,
                                    "Failed to read current share from shared memory due to system state inconsistency, skipping pod"
                                );
                                // Skip this pod due to system state inconsistency
                                continue;
                            }
                        };

                        // **Extract pod data from snapshot**
                        let pod_utilization = device_snapshot.get_pod_utilization(&host_pids);
                        let pod_memory = device_snapshot.get_pod_memory(&host_pids);

                        // **Calculate new share based on current utilization**
                        let new_share = calculate_delta(
                            device_config,
                            pod_utilization.total_utilization,
                            current_share,
                        );

                        // **Update shared memory with new values**
                        if let Err(e) = Self::update_shared_memory_state(
                            &pod_identifier,
                            &device_config.device_uuid,
                            Some(pod_memory),
                            Some(new_share),
                            device_snapshot.timestamp,
                        )
                        .await
                        {
                            error!(
                                pod_identifier = %pod_identifier,
                                device_idx = device_idx,
                                error = %e,
                                "Failed to update shared memory state"
                            );
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
            match active_pods.get_mut(&pod_identifier) {
                Some(pod_usage) => {
                    // Pod exists, add container and merge new device configs if needed
                    pod_usage.add_container(container_name.to_string(), container_pid, host_pid);

                    // Add any new device configs that don't already exist
                    for new_config in configs {
                        if !pod_usage
                            .device_configs
                            .iter()
                            .any(|existing| existing.device_uuid == new_config.device_uuid)
                        {
                            pod_usage.device_configs.push(new_config);
                        }
                    }
                }
                None => {
                    // New pod, create with provided configs
                    let mut pod_usage = PodDeviceUsage::new(configs);
                    pod_usage.add_container(container_name.to_string(), container_pid, host_pid);
                    active_pods.insert(pod_identifier.clone(), pod_usage);
                }
            }
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
                        // Get current state and total cores
                        let total_cuda_cores = device.get_total_cores() as i32;
                        let current_cores = device.get_available_cores();

                        // Calculate the delta (difference) to apply
                        let target_cores = new_share.max(0).min(total_cuda_cores);
                        let delta = target_cores - current_cores;

                        // Apply the delta to reach the target value
                        device.fetch_add_available_cores(delta);
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

    /// Gets a complete snapshot of device state (utilization + memory)
    async fn get_device_snapshot(
        device_idx: u32,
        last_seen_timestamp: u64,
    ) -> Result<Option<DeviceSnapshot>> {
        tokio::task::spawn_blocking({
            move || -> Result<Option<DeviceSnapshot>> {
                let nvml = nvml_wrapper::Nvml::init().context("Failed to initialize NVML")?;
                let device = nvml
                    .device_by_index(device_idx)
                    .context("Failed to get device by index")?;

                // Get utilization data from last seen timestamp
                let process_utilization_samples =
                    device.process_utilization_stats(last_seen_timestamp);
                if let Err(NvmlError::NotFound) = process_utilization_samples {
                    return Ok(None);
                }
                let process_utilization_samples = process_utilization_samples
                    .context("Failed to get process utilization stats")?;

                // Get memory data
                let process_info = device
                    .running_compute_processes()
                    .context("Failed to get running compute processes")?;

                let mut process_utilizations = std::collections::HashMap::new();
                let mut process_memories = std::collections::HashMap::new();
                let mut newest_timestamp = last_seen_timestamp;

                // Process utilization data
                for sample in process_utilization_samples {
                    // Skip old samples (defensive programming)
                    if sample.timestamp < last_seen_timestamp {
                        continue;
                    }

                    // Track the newest timestamp
                    if sample.timestamp > newest_timestamp {
                        newest_timestamp = sample.timestamp;
                    }

                    process_utilizations.insert(sample.pid, ProcessUtilization {
                        sm_util: sample.sm_util,
                        codec_util: codec_normalize(sample.enc_util + sample.dec_util),
                    });
                }

                // Process memory data
                for pi in process_info {
                    if let UsedGpuMemory::Used(bytes) = pi.used_gpu_memory {
                        process_memories.insert(pi.pid, bytes);
                    }
                }

                if process_utilizations.is_empty() && process_memories.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(DeviceSnapshot {
                        process_utilizations,
                        process_memories,
                        timestamp: newest_timestamp,
                    }))
                }
            }
        })
        .await
        .context("Blocking task failed")?
    }

    /// Read current share value from shared memory
    async fn read_current_share_from_shared_memory(
        pod_identifier: &str,
        device_uuid: &str,
    ) -> Result<i32> {
        tokio::task::spawn_blocking({
            let pod_identifier = pod_identifier.to_string();
            let device_uuid = device_uuid.to_string();

            move || -> Result<i32> {
                use utils::shared_memory::SharedMemoryHandle;
                let handle = SharedMemoryHandle::open(&pod_identifier)
                    .context("Failed to open shared memory")?;
                let state = handle.get_state();

                let devices_read = state.devices.read();
                if let Some(device) = devices_read.get(&device_uuid) {
                    Ok(device.get_available_cores())
                } else {
                    anyhow::bail!(
                        "Device {} not found in shared memory for pod {}. This indicates a system state inconsistency.",
                        device_uuid, pod_identifier
                    );
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
