//! Limiter Coordinator Module

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

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

/// State information when detecting/restoring shared memory
struct SharedMemoryState {
    existed: bool,
    restored_pids: Vec<usize>,
}

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
        self.start_heartbeat_task(cancellation_token.clone()).await;

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
        let mut watcher_tasks = self.device_watcher_tasks.write().await;
        for (device_idx, task) in watcher_tasks.drain() {
            tracing::debug!("Stopping watcher task for device {}", device_idx);
            task.abort();
        }

        // Stop heartbeat task
        let mut heartbeat_task = self.heartbeat_task.write().await;
        if let Some(task) = heartbeat_task.take() {
            tracing::debug!("Stopping heartbeat task");
            task.abort();
        }
    }

    /// Starts a monitoring task for each GPU device with cancellation support
    async fn start_watcher_with_cancellation(&self, cancellation_token: CancellationToken) {
        let active_pods = self.active_pods.clone();

        for device_idx in 0..self.device_count {
            let task = self.create_device_watcher_task(
                device_idx,
                self.watch_interval,
                active_pods.clone(),
                cancellation_token.clone(),
            );

            let mut watcher_tasks = self.device_watcher_tasks.write().await;
            watcher_tasks.insert(device_idx, task);
        }

        info!(
            device_count = self.device_count,
            "Started device watcher tasks for all GPU devices"
        );
    }

    /// Create a device watcher task for a specific GPU device
    fn create_device_watcher_task(
        &self,
        device_idx: u32,
        watch_interval: Duration,
        active_pods: Arc<RwLock<HashMap<String, PodDeviceUsage>>>,
        cancellation_token: CancellationToken,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval_timer = interval(watch_interval);
            let mut last_seen_timestamp = 0u64;

            info!(
                device_idx = device_idx,
                "Starting device watcher task for GPU device"
            );

            loop {
                tokio::select! {
                    _ = interval_timer.tick() => {},
                    _ = cancellation_token.cancelled() => {
                        info!(device_idx = device_idx, "Device watcher task cancelled");
                        break;
                    }
                }

                if let Err(e) = Self::run_device_monitoring_cycle(
                    device_idx,
                    &active_pods,
                    &mut last_seen_timestamp,
                )
                .await
                {
                    error!(device_idx = device_idx, error = %e, "Device monitoring cycle failed");
                }
            }

            info!(device_idx = device_idx, "Device watcher task completed");
        })
    }

    /// Run one cycle of device monitoring
    async fn run_device_monitoring_cycle(
        device_idx: u32,
        active_pods: &Arc<RwLock<HashMap<String, PodDeviceUsage>>>,
        last_seen_timestamp: &mut u64,
    ) -> Result<()> {
        let pods_for_device = Self::get_pods_using_device(device_idx, active_pods).await;

        if pods_for_device.is_empty() {
            debug!(device_idx = device_idx, "No pods using this device");
            return Ok(());
        }

        let device_snapshot =
            match Self::get_device_snapshot(device_idx, *last_seen_timestamp).await? {
                Some(snapshot) => {
                    *last_seen_timestamp = snapshot.timestamp;
                    snapshot
                }
                None => {
                    debug!(device_idx = device_idx, "No device data available");
                    return Ok(());
                }
            };

        for (pod_identifier, pod_usage) in pods_for_device {
            if let Err(e) = Self::process_pod_utilization_update(
                &pod_identifier,
                &pod_usage,
                device_idx,
                &device_snapshot,
            )
            .await
            {
                error!(
                    pod_identifier = %pod_identifier,
                    device_idx = device_idx,
                    error = %e,
                    "Failed to process pod utilization update"
                );
            }
        }

        Ok(())
    }

    /// Get pods that are using a specific device
    async fn get_pods_using_device(
        device_idx: u32,
        active_pods: &Arc<RwLock<HashMap<String, PodDeviceUsage>>>,
    ) -> Vec<(String, PodDeviceUsage)> {
        let pods = active_pods.read().await;
        pods.iter()
            .filter(|(_, usage)| {
                usage
                    .device_configs
                    .iter()
                    .any(|config| config.device_idx == device_idx)
            })
            .map(|(name, usage)| (name.clone(), usage.clone()))
            .collect()
    }

    /// Process utilization update for a single pod
    async fn process_pod_utilization_update(
        pod_identifier: &str,
        pod_usage: &PodDeviceUsage,
        device_idx: u32,
        device_snapshot: &DeviceSnapshot,
    ) -> Result<()> {
        let host_pids = pod_usage.get_host_pids();
        if host_pids.is_empty() {
            debug!(pod_identifier = %pod_identifier, device_idx = device_idx, "No active processes to monitor");
            return Ok(());
        }

        let device_config = pod_usage
            .device_configs
            .iter()
            .find(|config| config.device_idx == device_idx)
            .context("Device config must exist for filtered pod")?;

        let current_share = Self::get_available_cores(pod_identifier, &device_config.device_uuid)
            .await
            .context("Failed to read current share from shared memory")?;

        let pod_utilization = device_snapshot.get_pod_utilization(&host_pids);
        let pod_memory = device_snapshot.get_pod_memory(&host_pids);

        let new_share = calculate_delta(
            device_config,
            pod_utilization.total_utilization,
            current_share,
        );

        Self::update_shared_memory_state(
            pod_identifier,
            &device_config.device_uuid,
            Some(pod_memory),
            Some(new_share),
            device_snapshot.timestamp,
        )
        .await
        .context("Failed to update shared memory state")
    }

    /// Ensures a pod is registered with device configurations (idempotent operation)
    pub async fn ensure_pod_registered(
        &self,
        pod_identifier: &str,
        configs: Vec<DeviceConfig>,
    ) -> Result<()> {
        let pod_identifier = pod_identifier.to_string();

        let shared_memory_state = self.detect_existing_shared_memory(&pod_identifier, &configs)?;
        self.finalize_pod_registration(&pod_identifier, configs, shared_memory_state)
            .await
    }

    /// Detect and handle existing shared memory, returning restoration state
    fn detect_existing_shared_memory(
        &self,
        pod_identifier: &str,
        configs: &[DeviceConfig],
    ) -> Result<SharedMemoryState> {
        match SharedMemoryHandle::open(pod_identifier) {
            Ok(handle) => {
                debug!(pod_identifier = %pod_identifier, "Shared memory already exists for pod, ensuring registration consistency");
                self.restore_pod_from_shared_memory(pod_identifier, handle)
            }
            Err(_) => {
                debug!(pod_identifier = %pod_identifier, "Creating new shared memory for pod");
                self.create_new_pod_shared_memory(pod_identifier, configs)
            }
        }
    }

    /// Restore pod state from existing shared memory
    fn restore_pod_from_shared_memory(
        &self,
        pod_identifier: &str,
        handle: SharedMemoryHandle,
    ) -> Result<SharedMemoryState> {
        let restored_pids = handle.get_state().get_all_pids();

        if !restored_pids.is_empty() {
            debug!(
                pod_identifier = %pod_identifier,
                pids = ?restored_pids,
                "Found {} existing PIDs in shared memory",
                restored_pids.len()
            );
        }

        self.shared_memory_manager
            .register_existing_handle(pod_identifier, handle)?;

        Ok(SharedMemoryState {
            existed: true,
            restored_pids,
        })
    }

    /// Create new shared memory for pod
    fn create_new_pod_shared_memory(
        &self,
        pod_identifier: &str,
        configs: &[DeviceConfig],
    ) -> Result<SharedMemoryState> {
        self.shared_memory_manager
            .create_or_get_shared_memory(pod_identifier, configs)?;

        Ok(SharedMemoryState {
            existed: false,
            restored_pids: Vec::new(),
        })
    }

    /// Finalize pod registration with device usage and logging
    async fn finalize_pod_registration(
        &self,
        pod_identifier: &str,
        configs: Vec<DeviceConfig>,
        shared_memory_state: SharedMemoryState,
    ) -> Result<()> {
        let mut pod_usage = PodDeviceUsage::new(configs.clone());

        // Restore PIDs if any were found in existing shared memory
        for pid in &shared_memory_state.restored_pids {
            pod_usage.add_process(*pid as u32);
        }

        {
            let mut active_pods = self.active_pods.write().await;
            active_pods.insert(pod_identifier.to_string(), pod_usage);
        }

        info!(
            pod_identifier = %pod_identifier,
            device_count = configs.len(),
            restored_processes = shared_memory_state.restored_pids.len(),
            shared_memory_existed = shared_memory_state.existed,
            "Pod registered successfully"
        );

        if !shared_memory_state.restored_pids.is_empty() {
            info!(
                pod_identifier = %pod_identifier,
                restored_pid_count = shared_memory_state.restored_pids.len(),
                "Restored {} processes from existing shared memory",
                shared_memory_state.restored_pids.len()
            );
        }

        Ok(())
    }

    /// Registers a process within a pod (Process-level operation)
    pub async fn register_process(
        &self,
        pod_identifier: &str,
        container_name: &str,
        container_pid: u32,
        host_pid: u32,
    ) -> Result<()> {
        let pod_identifier = pod_identifier.to_string();

        {
            let mut active_pods = self.active_pods.write().await;
            let Some(pod_usage) = active_pods.get_mut(&pod_identifier) else {
                return Err(anyhow::anyhow!(
                    "Pod not found: {pod_identifier}. Must register pod first."
                ));
            };

            pod_usage.add_process(host_pid);
            let _ = self
                .shared_memory_manager
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
    pub async fn unregister_process(&self, pod_identifier: &str, host_pid: u32) -> Result<()> {
        let pod_identifier = pod_identifier.to_string();

        let mut active_pods = self.active_pods.write().await;
        if let Some(pod_usage) = active_pods.get_mut(&pod_identifier) {
            if pod_usage.remove_process(host_pid) {
                // process is empty, remove the pod
                active_pods.remove(&pod_identifier);
            }
        }

        let _ = self
            .shared_memory_manager
            .remove_pid(&pod_identifier, host_pid as usize);

        info!(
            pod_identifier = %pod_identifier,
            host_pid = host_pid,
            "Unregistered process from coordinator"
        );

        Ok(())
    }

    /// Unregisters a pod from the coordinator.
    pub async fn unregister_pod(&self, pod_identifier: &str) -> Result<()> {
        self.shared_memory_manager.cleanup(pod_identifier)?;
        self.active_pods
            .write()
            .await
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
        let pod_identifier = pod_identifier.to_string();
        let device_uuid = device_uuid.to_string();
        let pod_id_clone = pod_identifier.clone();
        Self::with_shared_memory_handle(&pod_identifier, move |state| {
            if !state.has_device(&device_uuid) {
                anyhow::bail!(
                    "Device {} not found in shared memory for pod {}",
                    device_uuid,
                    pod_id_clone
                );
            }

            if let Some(memory_used) = memory_used {
                state.with_device_by_uuid_mut(&device_uuid, |device| {
                    device.set_pod_memory_used(memory_used);
                });
            }

            if let Some(new_share) = new_share {
                state.with_device_by_uuid_mut(&device_uuid, |device| {
                    let total_cuda_cores = device.get_total_cores() as i32;
                    let current_cores = device.get_available_cores();
                    let target_cores = new_share.max(0).min(total_cuda_cores);
                    let delta = target_cores - current_cores;
                    device.fetch_add_available_cores(delta);
                });
            }

            state.update_heartbeat(timestamp);
            Ok(())
        })
        .await
    }

    /// Gets available cores for a device efficiently
    async fn get_available_cores(pod_identifier: &str, device_uuid: &str) -> Result<i32> {
        let pod_identifier = pod_identifier.to_string();
        let device_uuid = device_uuid.to_string();
        let pod_id_clone = pod_identifier.clone();
        Self::with_shared_memory_handle(&pod_identifier, move |state| {
            state
                .with_device_by_uuid(&device_uuid, |device| device.get_available_cores())
                .context(format!(
                    "Device {device_uuid} not found in shared memory for pod {pod_id_clone}. This indicates a system state inconsistency."
                ))
        })
        .await
    }

    /// Helper method to safely access shared memory handles with consistent error handling
    async fn with_shared_memory_handle<T, F>(pod_identifier: &str, f: F) -> Result<T>
    where
        F: FnOnce(&utils::shared_memory::SharedDeviceState) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let pod_identifier = pod_identifier.to_string();
        tokio::task::spawn_blocking(move || {
            let handle = SharedMemoryHandle::open(&pod_identifier)
                .context("Failed to open shared memory")?;
            let state = handle.get_state();
            f(state)
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
    pub async fn start_heartbeat_task(&self, cancellation_token: CancellationToken) {
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
                            let active_pods_guard = active_pods.read().await;
                            active_pods_guard.keys().cloned().collect::<Vec<String>>()
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
            let mut heartbeat_task = self.heartbeat_task.write().await;
            *heartbeat_task = Some(task);
        }

        info!("Started heartbeat task");
    }

    /// Updates only the heartbeat timestamp for a pod without modifying other state
    async fn update_heartbeat_only(pod_identifier: &str, timestamp: u64) -> Result<()> {
        let pod_identifier = pod_identifier.to_string();
        Self::with_shared_memory_handle(&pod_identifier, move |state| {
            state.update_heartbeat(timestamp);
            Ok(())
        })
        .await
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
