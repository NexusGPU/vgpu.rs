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
use utils::shared_memory::handle::SharedMemoryHandle;
use utils::shared_memory::manager::ThreadSafeSharedMemoryManager;
use utils::shared_memory::DeviceConfig;

use super::registry::PodDeviceUsage;
use super::utilization::{codec_normalize, DeviceSnapshot, ProcessUtilization};

/// Limiter coordinator.
pub struct LimiterCoordinator {
    /// Shared memory manager.
    shared_memory_manager: Arc<ThreadSafeSharedMemoryManager>,
    /// Active pod device usage: pod_identifier -> PodDeviceUsage
    active_pods: Arc<RwLock<HashMap<String, PodDeviceUsage>>>,
    /// Monitoring task handles for each device: device_idx -> JoinHandle
    device_watcher_tasks: RwLock<HashMap<u32, JoinHandle<()>>>,
    /// Heartbeat task handle
    heartbeat_task: RwLock<Option<JoinHandle<()>>>,
    /// Monitoring interval.
    watch_interval: Duration,
    /// Number of GPU devices.
    device_count: u32,
    /// glob pattern for shared memory files
    shared_memory_glob_pattern: String,
}

impl LimiterCoordinator {
    pub fn new(
        watch_interval: Duration,
        device_count: u32,
        shared_memory_glob_pattern: String,
    ) -> Self {
        Self {
            shared_memory_manager: Arc::new(ThreadSafeSharedMemoryManager::new()),
            active_pods: Arc::new(RwLock::new(HashMap::new())),
            device_watcher_tasks: RwLock::new(HashMap::new()),
            heartbeat_task: RwLock::new(None),
            watch_interval,
            device_count,
            shared_memory_glob_pattern,
        }
    }

    /// Run the coordinator with cancellation support
    pub async fn run(&self, cancellation_token: CancellationToken) {
        tracing::info!(
            "Starting LimiterCoordinator with {} GPU devices",
            self.device_count
        );

        // Clean up orphaned shared memory files on startup
        if let Err(e) = self.cleanup_orphaned_files_on_startup().await {
            tracing::warn!("Failed to cleanup orphaned files on startup: {}", e);
        }

        // Start monitoring tasks
        self.start_watcher_with_cancellation(cancellation_token.clone())
            .await;

        // Start periodic cleanup task
        let cleanup_task = self.start_periodic_cleanup_task(cancellation_token.clone());

        // Start the heartbeat task
        self.start_heartbeat_task(cancellation_token.clone());

        // Wait for cancellation
        cancellation_token.cancelled().await;

        tracing::info!("LimiterCoordinator received cancellation signal, stopping all tasks");

        // Stop periodic cleanup task
        cleanup_task.abort();

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

        // Stop heartbeat task
        let mut heartbeat_task = self.heartbeat_task.write().unwrap();
        if let Some(task) = heartbeat_task.take() {
            tracing::debug!("Stopping heartbeat task");
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
                        let current_share = match Self::get_available_cores(
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

    /// Ensures a pod is registered with device configurations (idempotent operation)
    pub fn ensure_pod_registered(
        &self,
        pod_identifier: &str,
        configs: Vec<DeviceConfig>,
    ) -> Result<()> {
        let pod_identifier = pod_identifier.to_string();

        // Check if pod is already fully registered
        let already_registered = {
            let active_pods = self.active_pods.read().expect("poisoned");
            active_pods.contains_key(&pod_identifier)
        };

        // First, try to detect if shared memory already exists and restore state if needed
        let mut restored_pids = Vec::new();
        let mut shared_memory_exists = false;

        // Try to open existing shared memory first to check if it already exists
        if let Ok(handle) = SharedMemoryHandle::open(&pod_identifier) {
            // Shared memory already exists - this is a restart scenario
            shared_memory_exists = true;
            debug!(
                pod_identifier = %pod_identifier,
                "Shared memory already exists for pod, ensuring registration consistency"
            );

            // Get all existing PIDs from shared memory
            restored_pids = handle.get_state().get_all_pids();

            if !restored_pids.is_empty() {
                debug!(
                    pod_identifier = %pod_identifier,
                    pids = ?restored_pids,
                    "Found {} existing PIDs in shared memory",
                    restored_pids.len()
                );
            }

            // Directly register the existing handle in the manager to avoid reinitializing
            self.shared_memory_manager
                .register_existing_handle(&pod_identifier, handle)?;
        } else {
            // Shared memory doesn't exist, create new one
            debug!(
                pod_identifier = %pod_identifier,
                "Creating new shared memory for pod"
            );

            self.shared_memory_manager
                .create_or_get_shared_memory(&pod_identifier, &configs)?;
        }

        // Initialize or update pod device usage
        {
            let mut active_pods = self.active_pods.write().expect("poisoned");
            if !active_pods.contains_key(&pod_identifier) {
                let mut pod_usage = PodDeviceUsage::new(configs.clone());

                // Restore PIDs if we found any in existing shared memory
                for pid in &restored_pids {
                    pod_usage.add_process(*pid as u32);
                }

                active_pods.insert(pod_identifier.clone(), pod_usage);

                info!(
                    pod_identifier = %pod_identifier,
                    device_count = configs.len(),
                    restored_processes = restored_pids.len(),
                    shared_memory_existed = shared_memory_exists,
                    "Pod registered successfully"
                );
            } else {
                // Pod already registered, this is an idempotent call
                debug!(
                    pod_identifier = %pod_identifier,
                    device_count = configs.len(),
                    "Pod already registered, ensuring consistency"
                );
            }
        }

        // If we restored PIDs from shared memory, log them
        if !restored_pids.is_empty() && !already_registered {
            info!(
                pod_identifier = %pod_identifier,
                restored_pid_count = restored_pids.len(),
                "Restored {} processes from existing shared memory",
                restored_pids.len()
            );
        }

        Ok(())
    }

    /// Registers a process within a pod (Process-level operation)
    pub fn register_process(
        &self,
        pod_identifier: &str,
        container_name: &str,
        container_pid: u32,
        host_pid: u32,
    ) -> Result<()> {
        let pod_identifier = pod_identifier.to_string();

        {
            let mut active_pods = self.active_pods.write().unwrap();
            let Some(pod_usage) = active_pods.get_mut(&pod_identifier) else {
                return Err(anyhow::anyhow!(
                    "Pod not found: {pod_identifier}. Must register pod first."
                ));
            };

            pod_usage.add_process(host_pid);
            self.shared_memory_manager
                .add_pid(&pod_identifier, host_pid as usize);
        }

        info!(
            pod_identifier = %pod_identifier,
            container_name = %container_name,
            container_pid = container_pid,
            host_pid = host_pid,
            "Registered process to coordinator"
        );

        Ok(())
    }

    /// Unregisters a single process from the coordinator.
    pub fn unregister_process(&self, pod_identifier: &str, host_pid: u32) -> Result<()> {
        let pod_identifier = pod_identifier.to_string();

        let mut active_pods = self.active_pods.write().unwrap();
        if let Some(pod_usage) = active_pods.get_mut(&pod_identifier) {
            if pod_usage.remove_process(host_pid) {
                // process is empty, remove the pod
                active_pods.remove(&pod_identifier);
            }
        }

        self.shared_memory_manager
            .remove_pid(&pod_identifier, host_pid as usize);

        info!(
            pod_identifier = %pod_identifier,
            host_pid = host_pid,
            "Unregistered process from coordinator"
        );

        Ok(())
    }

    /// Unregisters a pod from the coordinator.
    pub fn unregister_pod(&self, pod_identifier: &str) -> Result<()> {
        self.shared_memory_manager.cleanup(pod_identifier)?;
        self.active_pods
            .write()
            .expect("poisoned")
            .remove(&pod_identifier.to_string());
        Ok(())
    }

    /// Updates the shared memory state efficiently without unnecessary blocking
    async fn update_shared_memory_state(
        pod_identifier: &str,
        device_uuid: &str,
        memory_used: Option<u64>,
        new_share: Option<i32>,
        timestamp: u64,
    ) -> Result<()> {
        use utils::shared_memory::handle::SharedMemoryHandle;

        // Simple shared memory operations don't need spawn_blocking
        let handle =
            SharedMemoryHandle::open(pod_identifier).context("Failed to open shared memory")?;
        let state = handle.get_state();

        // Update the memory usage in shared memory
        if state.has_device(device_uuid) {
            if let Some(memory_used) = memory_used {
                state.with_device_by_uuid_mut(device_uuid, |device| {
                    device.set_pod_memory_used(memory_used);
                });
            }
            if let Some(new_share) = new_share {
                state.with_device_by_uuid_mut(device_uuid, |device| {
                    // Get current state and total cores
                    let total_cuda_cores = device.get_total_cores() as i32;
                    let current_cores = device.get_available_cores();

                    // Calculate the delta (difference) to apply
                    let target_cores = new_share.max(0).min(total_cuda_cores);
                    let delta = target_cores - current_cores;

                    // Apply the delta to reach the target value atomically
                    device.fetch_add_available_cores(delta);
                });
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

    /// Gets available cores for a device efficiently
    async fn get_available_cores(pod_identifier: &str, device_uuid: &str) -> Result<i32> {
        use utils::shared_memory::handle::SharedMemoryHandle;

        let handle =
            SharedMemoryHandle::open(pod_identifier).context("Failed to open shared memory")?;
        let state = handle.get_state();

        if let Some(available_cores) =
            state.with_device_by_uuid(device_uuid, |device| device.get_available_cores())
        {
            Ok(available_cores)
        } else {
            anyhow::bail!(
                "Device {} not found in shared memory for pod {}. This indicates a system state inconsistency.",
                device_uuid, pod_identifier
            );
        }
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

                    process_utilizations.insert(
                        sample.pid,
                        ProcessUtilization {
                            sm_util: sample.sm_util,
                            codec_util: codec_normalize(sample.enc_util + sample.dec_util),
                        },
                    );
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

    /// Clean up orphaned shared memory files on startup
    async fn cleanup_orphaned_files_on_startup(&self) -> Result<()> {
        tracing::info!("Cleaning up orphaned shared memory files on startup...");

        let cleaned_files = self
            .shared_memory_manager
            .cleanup_orphaned_files(&self.shared_memory_glob_pattern)
            .context("Failed to cleanup orphaned shared memory files")?;

        if !cleaned_files.is_empty() {
            tracing::info!(
                "Cleaned up {} orphaned shared memory files: {:?}",
                cleaned_files.len(),
                cleaned_files
            );
        } else {
            tracing::info!("No orphaned shared memory files found");
        }

        Ok(())
    }

    /// Start periodic cleanup task for unused shared memory segments
    fn start_periodic_cleanup_task(&self, cancellation_token: CancellationToken) -> JoinHandle<()> {
        let shared_memory_manager = self.shared_memory_manager.clone();

        tokio::spawn(async move {
            // Run cleanup every 5 minutes
            let mut cleanup_interval = interval(Duration::from_secs(300));
            cleanup_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            tracing::info!("Starting periodic shared memory cleanup task (every 5 minutes)");

            loop {
                tokio::select! {
                    _ = cleanup_interval.tick() => {
                        tracing::debug!("Running periodic cleanup of unused shared memory segments");

                        match shared_memory_manager.cleanup_unused() {
                            Ok(cleaned_files) => {
                                if !cleaned_files.is_empty() {
                                    tracing::info!(
                                        "Periodic cleanup: removed {} unused shared memory segments: {:?}",
                                        cleaned_files.len(),
                                        cleaned_files
                                    );
                                } else {
                                    tracing::debug!("Periodic cleanup: no unused shared memory segments found");
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Periodic cleanup failed: {}", e);
                            }
                        }
                    }
                    _ = cancellation_token.cancelled() => {
                        tracing::info!("Periodic cleanup task cancelled");
                        break;
                    }
                }
            }
        })
    }

    /// Starts the heartbeat task that updates heartbeat every 0.5 seconds
    pub fn start_heartbeat_task(&self, cancellation_token: CancellationToken) {
        let active_pods = self.active_pods.clone();
        let heartbeat_interval = Duration::from_millis(500);

        let task = tokio::spawn(async move {
            let mut interval = interval(heartbeat_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let timestamp = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs();

                        // Get all active pods
                        let pods = {
                            let active_pods_guard = active_pods.read().unwrap();
                            active_pods_guard.keys().cloned().collect::<Vec<_>>()
                        };

                        // Update heartbeat for each pod
                        for pod_identifier in pods {
                            if let Err(e) = Self::update_heartbeat_only(&pod_identifier, timestamp).await {
                                debug!("Failed to update heartbeat for pod {}: {}", pod_identifier, e);
                            }
                        }
                    }
                    _ = cancellation_token.cancelled() => {
                        info!("Heartbeat task cancelled");
                        break;
                    }
                }
            }
        });

        // Store the task handle
        {
            let mut heartbeat_task = self.heartbeat_task.write().unwrap();
            *heartbeat_task = Some(task);
        }

        info!("Started heartbeat task");
    }

    /// Updates only the heartbeat timestamp for a pod without modifying other state
    async fn update_heartbeat_only(pod_identifier: &str, timestamp: u64) -> Result<()> {
        use utils::shared_memory::handle::SharedMemoryHandle;

        let handle = SharedMemoryHandle::open(pod_identifier)
            .context("Failed to open shared memory for heartbeat update")?;
        let state = handle.get_state();

        state.update_heartbeat(timestamp);
        Ok(())
    }
}

impl Drop for LimiterCoordinator {
    fn drop(&mut self) {
        // Stop all monitoring tasks
        let mut watcher_tasks = self.device_watcher_tasks.write().unwrap();
        for (_, task) in watcher_tasks.drain() {
            task.abort();
        }

        // Stop heartbeat task
        let mut heartbeat_task = self.heartbeat_task.write().unwrap();
        if let Some(task) = heartbeat_task.take() {
            task.abort();
        }

        info!("LimiterCoordinator dropped, all tasks stopped");
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
