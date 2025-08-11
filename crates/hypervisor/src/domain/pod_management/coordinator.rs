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
use utils::shared_memory::{
    handle::SharedMemoryHandle, manager::ThreadSafeSharedMemoryManager, DeviceConfig,
    SharedDeviceState,
};

use super::pod_state_store::PodStateStore;
use super::utilization::{codec_normalize, DeviceSnapshot, ProcessUtilization};

/// Limiter coordinator
pub struct LimiterCoordinator {
    /// Shared memory manager.
    shared_memory_manager: Arc<ThreadSafeSharedMemoryManager>,
    /// Reference to pod state store for querying pod information
    pod_state_store: Arc<PodStateStore>,
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
        pod_state_store: Arc<PodStateStore>,
    ) -> Self {
        Self {
            shared_memory_manager: Arc::new(ThreadSafeSharedMemoryManager::new()),
            pod_state_store,
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
        let pod_state_store = self.pod_state_store.clone();

        for device_idx in 0..self.device_count {
            let task = self.create_device_watcher_task(
                device_idx,
                self.watch_interval,
                pod_state_store.clone(),
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
        pod_state_store: Arc<PodStateStore>,
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
                    &pod_state_store,
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
        pod_state_store: &Arc<PodStateStore>,
        last_seen_timestamp: &mut u64,
    ) -> Result<()> {
        let pods_for_device = pod_state_store.get_pods_using_device(device_idx);

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

        for pod_identifier in pods_for_device {
            let Some(host_pids) = pod_state_store.get_host_pids_for_pod(&pod_identifier) else {
                debug!(pod_identifier = %pod_identifier, "Pod not found when fetching host PIDs");
                continue;
            };

            if host_pids.is_empty() {
                debug!(pod_identifier = %pod_identifier, device_idx = device_idx, "No active processes to monitor");
                continue;
            }

            let Some(device_config) =
                pod_state_store.get_device_config_for_pod(&pod_identifier, device_idx)
            else {
                debug!(pod_identifier = %pod_identifier, device_idx = device_idx, "Device config not found for pod");
                continue;
            };

            if let Err(e) = Self::process_pod_utilization_update(
                &pod_identifier,
                &host_pids,
                &device_config,
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

    /// Process utilization update for a single pod
    async fn process_pod_utilization_update(
        pod_identifier: &str,
        host_pids: &[u32],
        device_config: &DeviceConfig,
        device_snapshot: &DeviceSnapshot,
    ) -> Result<()> {
        let current_share =
            Self::get_available_cores(pod_identifier, device_config.device_idx as usize)
                .await
                .context("Failed to read current share from shared memory")?;

        let pod_utilization = device_snapshot.get_pod_utilization(host_pids);
        let pod_memory = device_snapshot.get_pod_memory(host_pids);

        let new_share = calculate_delta(
            device_config,
            pod_utilization.total_utilization,
            current_share,
        );

        Self::update_shared_memory_state(
            pod_identifier,
            device_config.device_idx as usize,
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
        let restored_pids = match self.shared_memory_manager.get_shared_memory(pod_identifier) {
            Ok(ptr) => {
                debug!(pod_identifier = %pod_identifier, "Shared memory already exists for pod, ensuring registration consistency");
                let restored_pids = unsafe { &*ptr }.get_all_pids();

                if !restored_pids.is_empty() {
                    debug!(
                        pod_identifier = %pod_identifier,
                        pids = ?restored_pids,
                        "Found {} existing PIDs in shared memory",
                        restored_pids.len()
                    );
                }
                restored_pids
            }
            Err(_) => {
                debug!(pod_identifier = %pod_identifier, "Creating new shared memory for pod");
                self.shared_memory_manager
                    .create_shared_memory(pod_identifier, &configs)?;
                Vec::new()
            }
        };

        if !restored_pids.is_empty() {
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
    pub async fn register_process(
        &self,
        pod_identifier: &str,
        container_name: &str,
        container_pid: u32,
        host_pid: u32,
        device_configs: Vec<DeviceConfig>,
    ) -> Result<()> {
        self.ensure_pod_registered(pod_identifier, device_configs)
            .await?;

        // Add PID to shared memory
        self.shared_memory_manager
            .add_pid(pod_identifier, host_pid as usize)
            .context("Failed to add PID to shared memory")?;

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
        // Remove PID from shared memory
        self.shared_memory_manager
            .remove_pid(pod_identifier, host_pid as usize)
            .context("Failed to remove PID from shared memory")?;

        info!(
            pod_identifier = %pod_identifier,
            host_pid = host_pid,
            "Unregistered process from coordinator"
        );

        Ok(())
    }

    /// Updates the shared memory state efficiently without unnecessary blocking
    async fn update_shared_memory_state(
        pod_identifier: &str,
        device_index: usize,
        memory_used: Option<u64>,
        new_share: Option<i32>,
        timestamp: u64,
    ) -> Result<()> {
        Self::with_shared_memory_handle(pod_identifier, move |state, pod_identifier: &str| {
            if !state.has_device(device_index) {
                anyhow::bail!(
                    "Device {} not found in shared memory for pod {}",
                    device_index,
                    pod_identifier
                );
            }

            if let Some(memory_used) = memory_used {
                state.with_device(device_index, |device| {
                    device.device_info.set_pod_memory_used(memory_used);
                });
            }

            if let Some(new_share) = new_share {
                state.with_device(device_index, |device| {
                    let total_cuda_cores = device.device_info.get_total_cores() as i32;
                    let current_cores = device.device_info.get_available_cores();
                    let target_cores = new_share.max(0).min(total_cuda_cores);
                    let delta = target_cores - current_cores;
                    tracing::info!(
                        pod_identifier = %pod_identifier,
                        device_index = device_index,
                        current_cores = current_cores,
                        target_cores = target_cores,
                        delta = delta,
                        "Updating shared memory state"
                    );
                    device.device_info.fetch_add_available_cores(delta);
                });
            }

            state.update_heartbeat(timestamp);
            Ok(())
        })
    }

    /// Gets available cores for a device efficiently
    async fn get_available_cores(pod_identifier: &str, device_index: usize) -> Result<i32> {
        Self::with_shared_memory_handle(pod_identifier, move |state, pod_identifier: &str| {
            state
                .with_device(device_index, |device| device.device_info.get_available_cores())
                .context(format!(
                    "Device {device_index} not found in shared memory for pod {pod_identifier}. This indicates a system state inconsistency.",
                ))
        })
    }

    /// Helper method to safely access shared memory handles with consistent error handling
    fn with_shared_memory_handle<T, F>(pod_identifier: &str, f: F) -> Result<T>
    where
        F: FnOnce(&SharedDeviceState, &str) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let handle =
            SharedMemoryHandle::open(pod_identifier).context("Failed to open shared memory")?;
        let state = handle.get_state();
        f(state, pod_identifier)
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
                    match device.process_utilization_stats(last_seen_timestamp) {
                        Ok(process_utilization_samples) => process_utilization_samples,
                        Err(NvmlError::NotFound) => {
                            return Ok(None);
                        }
                        Err(e) => {
                            return Err(e.into());
                        }
                    };

                // Get memory data
                let process_info = device
                    .running_compute_processes()
                    .context("Failed to get running compute processes")?;

                let mut process_utilizations = HashMap::new();
                let mut process_memories = HashMap::new();
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

    /// Start periodic cleanup task for unused shared memory segments
    fn start_periodic_cleanup_task(&self, cancellation_token: CancellationToken) -> JoinHandle<()> {
        let shared_memory_manager = self.shared_memory_manager.clone();

        let shared_memory_glob_pattern = self.shared_memory_glob_pattern.clone();
        let pod_state_store = self.pod_state_store.clone();
        tokio::spawn(async move {
            // Run cleanup every 5 minutes
            let mut cleanup_interval = interval(Duration::from_secs(300));
            cleanup_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            tracing::info!("Starting periodic shared memory cleanup task (every 5 minutes)");

            loop {
                tokio::select! {
                    _ = cleanup_interval.tick() => {
                        tracing::info!("Running periodic cleanup of unused shared memory segments");

                        if let Err(e) = shared_memory_manager.cleanup_orphaned_files(&shared_memory_glob_pattern, |identifier| {
                            !pod_state_store.contains_pod(identifier)
                        }) {
                            tracing::warn!("Failed to cleanup orphaned shared memory files: {}", e);
                        }

                        match shared_memory_manager.cleanup_unused(|identifier| {
                            pod_state_store.contains_pod(identifier)
                        }) {
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
        let pod_state_store = self.pod_state_store.clone();
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

                        // Get all pod identifiers from state store
                        let pod_identifiers = pod_state_store.list_pod_identifiers();

                        // Update heartbeat for each pod
                        for pod_identifier in pod_identifiers {
                            if let Err(e) = Self::with_shared_memory_handle(&pod_identifier, move |state, _| {
                                state.update_heartbeat(timestamp);
                                Ok(())
                            })
                            {
                                tracing::warn!(
                                    pod_identifier = %pod_identifier,
                                    error = %e,
                                    "Failed to update heartbeat for pod"
                                );
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
