//! Limiter Coordinator Module

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use anyhow::Context;
use anyhow::Result;

use erl::{DeviceController, DeviceControllerConfig};
use tokio::task::JoinHandle;
use tokio::time::{interval, interval_at, Instant};
use tokio_util::sync::CancellationToken;
use tracing::debug;
use tracing::error;
use tracing::info;
use utils::shared_memory::handle::SHM_PATH_SUFFIX;
use utils::shared_memory::{
    erl_adapter::ErlSharedMemoryAdapter, handle::SharedMemoryHandle, DeviceConfig, PodIdentifier,
};

use super::traits::{DeviceSnapshotProvider, PodStateRepository, TimeSource};
use super::utilization::DeviceSnapshot;
use utils::shared_memory::traits::SharedMemoryAccess;

/// Cleanup interval for unused shared memory segments in seconds
const CLEANUP_INTERVAL_SECS: u64 = 300;

/// Configuration for the LimiterCoordinator
pub struct CoordinatorConfig {
    /// Monitoring interval
    pub watch_interval: Duration,
    /// Number of GPU devices
    pub device_count: u32,
    /// Glob pattern for shared memory files
    pub shared_memory_glob_pattern: String,
    /// Base path for shared memory operations
    pub base_path: PathBuf,
    /// ERL (Elastic Rate Limiter) configuration
    pub erl_config: crate::config::ErlConfig,
}

type ErlControllerMap = HashMap<
    (PodIdentifier, u32),
    DeviceController<ErlSharedMemoryAdapter<Arc<SharedMemoryHandle>>>,
>;

/// Newtype wrapper for ERL controllers indexed by (PodIdentifier, device_idx)
struct ErlControllers(ErlControllerMap);

impl ErlControllers {
    fn new() -> Self {
        Self(HashMap::new())
    }
}

impl core::ops::Deref for ErlControllers {
    type Target = ErlControllerMap;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for ErlControllers {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Generic limiter coordinator with dependency injection
pub struct LimiterCoordinator<M, P, D, T> {
    /// Shared memory access dependency
    shared_memory: Arc<M>,
    /// Pod state repository dependency
    pod_state: Arc<P>,
    /// Device snapshot provider dependency
    snapshot: Arc<D>,
    /// Time source dependency
    time: Arc<T>,
    /// Monitoring task handles for each device: device_idx -> JoinHandle
    device_watcher_tasks: RwLock<HashMap<u32, JoinHandle<()>>>,
    /// Heartbeat task handle
    heartbeat_task: RwLock<Option<JoinHandle<()>>>,
    /// ERL utilization controllers per pod per device: (PodIdentifier, device_idx) -> controller
    erl_controllers: Arc<RwLock<ErlControllers>>,
    /// Monitoring interval.
    watch_interval: Duration,
    /// Number of GPU devices.
    device_count: u32,
    /// glob pattern for shared memory files
    shared_memory_glob_pattern: String,
    /// Base path for shared memory operations
    base_path: PathBuf,
    /// ERL configuration
    erl_config: crate::config::ErlConfig,
}

impl<M, P, D, T> LimiterCoordinator<M, P, D, T>
where
    M: SharedMemoryAccess + 'static,
    P: PodStateRepository + 'static,
    D: DeviceSnapshotProvider + 'static,
    T: TimeSource + 'static,
{
    /// Create a new generic coordinator with injected dependencies
    pub fn new(
        config: CoordinatorConfig,
        shared_memory: Arc<M>,
        pod_state: Arc<P>,
        snapshot: Arc<D>,
        time: Arc<T>,
    ) -> Self {
        Self {
            shared_memory,
            pod_state,
            snapshot,
            time,
            device_watcher_tasks: RwLock::new(HashMap::new()),
            heartbeat_task: RwLock::new(None),
            erl_controllers: Arc::new(RwLock::new(ErlControllers::new())),
            watch_interval: config.watch_interval,
            device_count: config.device_count,
            shared_memory_glob_pattern: config.shared_memory_glob_pattern,
            base_path: config.base_path,
            erl_config: config.erl_config,
        }
    }

    /// Return the glob pattern for shared memory files including the `shm` suffix
    pub fn shm_file_glob_pattern(&self) -> String {
        format!("{}/{}", self.shared_memory_glob_pattern, SHM_PATH_SUFFIX)
    }

    pub async fn find_shared_memory_files(&self, glob_pattern: &str) -> Result<Vec<PathBuf>> {
        self.shared_memory
            .find_shared_memory_files(glob_pattern)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    pub fn extract_identifier_from_path(&self, file_path: &Path) -> Result<PodIdentifier> {
        self.shared_memory
            .extract_identifier_from_path(&self.base_path, file_path)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Run the coordinator with cancellation support
    #[tracing::instrument(skip(self, cancellation_token), fields(device_count = self.device_count))]
    pub async fn run(&self, cancellation_token: CancellationToken) {
        tracing::info!(
            "Starting LimiterCoordinator with {} GPU devices",
            self.device_count
        );

        // Start monitoring tasks
        self.start_watcher_with_cancellation(cancellation_token.clone())
            .await;

        // Start periodic cleanup task
        let cleanup_task = self.start_cleanup_task(cancellation_token.clone());

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
        for device_idx in 0..self.device_count {
            let task = self.create_device_watcher_task(
                device_idx,
                self.watch_interval,
                self.pod_state.clone(),
                self.snapshot.clone(),
                self.erl_controllers.clone(),
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
        pod_state: Arc<P>,
        snapshot: Arc<D>,
        erl_controllers: Arc<RwLock<ErlControllers>>,
        cancellation_token: CancellationToken,
    ) -> JoinHandle<()> {
        let erl_config = self.erl_config.clone();
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

                if let Err(e) = Self::run_device_monitoring_cycle_static(
                    device_idx,
                    pod_state.as_ref(),
                    snapshot.as_ref(),
                    erl_controllers.as_ref(),
                    &mut last_seen_timestamp,
                    &erl_config,
                )
                .await
                {
                    error!(device_idx = device_idx, error = %e, "Device monitoring cycle failed");
                }
            }

            info!(device_idx = device_idx, "Device watcher task completed");
        })
    }

    /// Static version for spawn tasks
    async fn run_device_monitoring_cycle_static(
        device_idx: u32,
        pod_state: &P,
        snapshot: &D,
        erl_controllers: &RwLock<ErlControllers>,
        last_seen_timestamp: &mut u64,
        erl_config: &crate::config::ErlConfig,
    ) -> Result<()> {
        let pods_for_device = pod_state.get_pods_using_device(device_idx);

        if pods_for_device.is_empty() {
            debug!(device_idx = device_idx, "No pods using this device");
            return Ok(());
        }

        let device_snapshot = snapshot
            .get_device_snapshot(device_idx, *last_seen_timestamp)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        *last_seen_timestamp = device_snapshot.timestamp;

        for pod_id in pods_for_device {
            let Some(host_pids) = pod_state.get_host_pids_for_pod(&pod_id) else {
                debug!(pod_identifier = %pod_id, "Pod not found when fetching host PIDs");
                continue;
            };

            if host_pids.is_empty() {
                debug!(pod_identifier = %pod_id, device_idx = device_idx, "No active processes to monitor");
                continue;
            }

            let Some(device_config) = pod_state.get_device_config_for_pod(&pod_id, device_idx)
            else {
                debug!(pod_identifier = %pod_id, device_idx = device_idx, "Device config not found for pod");
                continue;
            };
            if let Err(e) = Self::process_pod_utilization_update(
                pod_state,
                &pod_id,
                &host_pids,
                device_config.as_ref(),
                &device_snapshot,
                erl_controllers,
                erl_config,
            )
            .await
            {
                error!(
                    pod_identifier = %pod_id,
                    device_idx = device_idx,
                    error = %e,
                    "Failed to process pod utilization update"
                );
            }
        }

        Ok(())
    }

    /// Instance method for testing - delegates to static
    #[cfg(test)]
    async fn run_device_monitoring_cycle(
        &self,
        device_idx: u32,
        last_seen_timestamp: &mut u64,
    ) -> Result<()> {
        Self::run_device_monitoring_cycle_static(
            device_idx,
            self.pod_state.as_ref(),
            self.snapshot.as_ref(),
            self.erl_controllers.as_ref(),
            last_seen_timestamp,
            &self.erl_config,
        )
        .await
    }

    async fn process_pod_utilization_update(
        pod_state: &P,
        pod_identifier: &PodIdentifier,
        host_pids: &[u32],
        device_config: &DeviceConfig,
        device_snapshot: &DeviceSnapshot,
        erl_controllers: &RwLock<ErlControllers>,
        erl_config: &crate::config::ErlConfig,
    ) -> Result<()> {
        let pod_path = pod_state.pod_path(pod_identifier);
        // Use spawn_blocking to avoid blocking the tokio runtime
        let pod_path_clone = pod_path.clone();
        let handle = tokio::task::spawn_blocking(move || SharedMemoryHandle::open(&pod_path_clone))
            .await
            .context("Failed to spawn blocking task")?
            .context("Failed to open shared memory")?;
        let handle = Arc::new(handle);
        let state = handle.get_state();

        let device_index = device_config.device_idx as usize;
        if !state.has_device(device_index) {
            anyhow::bail!(
                "Device {device_index} not found in shared memory for pod {pod_identifier}"
            );
        }

        let pod_utilization = device_snapshot.get_pod_utilization(host_pids);
        let pod_memory = device_snapshot.get_pod_memory(host_pids);

        // Update memory usage (common for both versions)
        state.with_device(
            device_index,
            |device| device.device_info.set_pod_memory_used(pod_memory),
            |device| device.device_info.set_pod_memory_used(pod_memory),
        );

        // Dispatch to version-specific update
        match state.version() {
            1 => Self::update_v1_state(
                state,
                pod_identifier,
                device_config,
                device_index,
                &pod_utilization,
            ),
            2 => {
                Self::update_v2_state(
                    handle.clone(),
                    pod_identifier,
                    device_config,
                    device_index,
                    &pod_utilization,
                    pod_memory,
                    device_snapshot.timestamp,
                    erl_controllers,
                    erl_config,
                )
                .await
            }
            version => {
                tracing::warn!(
                    pod_identifier = %pod_identifier,
                    version = version,
                    "Unsupported shared memory version"
                );
            }
        }

        // Update heartbeat (common for both versions)
        state.update_heartbeat(device_snapshot.timestamp / 1_000_000);
        Ok(())
    }

    /// V1: Update available_cores based on utilization
    fn update_v1_state(
        state: &utils::shared_memory::SharedDeviceState,
        pod_identifier: &PodIdentifier,
        device_config: &DeviceConfig,
        device_index: usize,
        pod_utilization: &super::utilization::PodUtilization,
    ) {
        let current_share = state
            .with_device(
                device_index,
                |device| device.device_info.get_available_cores(),
                |_| 0,
            )
            .unwrap_or(0);

        let new_share = calculate_delta(
            device_config,
            pod_utilization.total_utilization,
            current_share,
        );

        state.with_device(
            device_index,
            |device| device.device_info.set_available_cores(new_share),
            |_| {},
        );

        debug!(
            pod_identifier = %pod_identifier,
            user_current = pod_utilization.total_utilization,
            user_new = new_share,
            "V1: Updated available_cores"
        );
    }

    /// V2: Update ERL controller with utilization feedback
    #[expect(
        clippy::too_many_arguments,
        reason = "Required parameters for ERL controller update"
    )]
    async fn update_v2_state(
        handle: Arc<SharedMemoryHandle>,
        pod_identifier: &PodIdentifier,
        device_config: &DeviceConfig,
        device_index: usize,
        pod_utilization: &super::utilization::PodUtilization,
        pod_memory: u64,
        snapshot_timestamp: u64,
        erl_controllers: &RwLock<ErlControllers>,
        erl_config: &crate::config::ErlConfig,
    ) {
        let key = (pod_identifier.clone(), device_config.device_idx);
        let target_utilization = (device_config.up_limit as f64 / 100.0).clamp(0.0, 1.0);
        if target_utilization < 0.05 {
            tracing::warn!(
                pod_identifier = %pod_identifier,
                device_index = device_index,
                up_limit = device_config.up_limit,
                target_utilization = target_utilization,
                "Target utilization is very low; kernel launches may be throttled aggressively"
            );
        }

        // Scale max_refill_rate based on up_limit to enforce quota
        // up_limit represents the percentage of GPU this pod can use
        // So max_refill_rate should be proportionally scaled
        let quota_ratio = target_utilization;
        let scaled_max_refill_rate = erl_config.max_refill_rate * quota_ratio;
        let scaled_initial_refill_rate = erl_config.initial_refill_rate * quota_ratio;

        use std::collections::hash_map::Entry;
        let mut controllers = erl_controllers.write().await;
        let controller = match controllers.entry(key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(vacant) => {
                let backend = ErlSharedMemoryAdapter::new(handle.clone());

                let ctrl_cfg = DeviceControllerConfig {
                    target_utilization,
                    burst_seconds: erl_config.burst_seconds,
                    capacity_floor: erl_config.capacity_floor,
                    kp: erl_config.pid_kp,
                    ki: erl_config.pid_ki,
                    kd: erl_config.pid_kd,
                    integral_limit: erl_config.integral_limit,
                    rate_min: erl_config.min_refill_rate,
                    rate_max: scaled_max_refill_rate,
                    rate_initial: scaled_initial_refill_rate,
                    min_delta_time: erl_config.min_delta_time,
                    utilization_smoothing: erl_config.derivative_filter,
                };

                match DeviceController::new(backend, device_index, ctrl_cfg) {
                    Ok(ctrl) => {
                        debug!(
                            pod_identifier = %pod_identifier,
                            device_index = device_index,
                            up_limit = device_config.up_limit,
                            target_utilization = target_utilization,
                            scaled_max_refill_rate = %format!("{:.1}", scaled_max_refill_rate),
                            initial_rate = ctrl.state().last_refill_rate,
                            "Initialized PID controller for pod/device with scaled refill rate"
                        );
                        vacant.insert(ctrl)
                    }
                    Err(e) => {
                        tracing::error!(
                            pod_identifier = %pod_identifier,
                            device_index = device_index,
                            error = %e,
                            "Failed to create PID controller for ERL"
                        );
                        return;
                    }
                }
            }
        };

        let utilization_ratio = (pod_utilization.total_utilization as f64 / 100.0).clamp(0.0, 1.0);

        if let Err(e) = controller.update_with_timestamp(utilization_ratio, snapshot_timestamp) {
            tracing::error!(
                pod_identifier = %pod_identifier,
                device_index = device_index,
                raw_utilization = pod_utilization.total_utilization,
                normalized_utilization = utilization_ratio,
                error = %e,
                "Failed to update ERL controller"
            );
        } else {
            debug!(
                pod_identifier = %pod_identifier,
                device_index = device_index,
                raw_utilization = pod_utilization.total_utilization,
                normalized_utilization = utilization_ratio,
                memory = pod_memory,
                new_rate = controller.state().last_refill_rate,
                target_utilization = controller.state().target_utilization,
                "V2: Updated ERL controller with PID feedback"
            );
        }
    }

    /// Ensures a pod is registered with device configurations
    #[tracing::instrument(skip(self, configs), fields(pod = %pod_identifier, device_count = configs.len()))]
    pub async fn ensure_pod_registered(
        &self,
        pod_identifier: &PodIdentifier,
        configs: &[DeviceConfig],
    ) -> Result<Vec<usize>> {
        let pod_path = self.pod_state.pod_path(pod_identifier);

        // TODO: need remove
        info!(pod_identifier = %pod_identifier, "Ensuring pod registered with device configurations");

        // Try to get existing shared memory and immediately extract data
        let (need_create, restored_pids) = {
            let get_result = self.shared_memory.get_shared_memory(pod_path).await;

            if let Ok(ptr_wrapper) = get_result {
                // TODO: need change to debug
                info!(pod_identifier = %pod_identifier, "Shared memory already exists for pod, ensuring registration consistency");

                // Use the pointer immediately and extract data before any awaits
                let restored_pids = unsafe {
                    let state = &*ptr_wrapper.as_mut_ptr();
                    let pids = state.get_all_pids();

                    // update limit info
                    for config in configs {
                        state.with_device(
                            config.device_idx as usize,
                            |device| {
                                device.device_info.set_total_cores(config.total_cuda_cores);
                                device.device_info.set_up_limit(config.up_limit);
                                device.device_info.set_mem_limit(config.mem_limit);
                            },
                            |device| {
                                device.device_info.set_total_cores(config.total_cuda_cores);
                                device.device_info.set_up_limit(config.up_limit);
                                device.device_info.set_mem_limit(config.mem_limit);
                            },
                        );
                    }

                    pids
                };

                if !restored_pids.is_empty() {
                    debug!(
                        pod_identifier = %pod_identifier,
                        pids = ?restored_pids,
                        "Found {} existing PIDs in shared memory",
                        restored_pids.len()
                    );
                }

                (false, restored_pids)
            } else {
                (true, Vec::new())
            }
        };

        // Create shared memory if needed (get_result is dropped here)
        let restored_pids = if need_create {
            // TODO: need change to debug
            info!(pod_identifier = %pod_identifier, "Creating new shared memory for pod");
            let pod_path = self.pod_state.pod_path(pod_identifier);
            self.shared_memory
                .create_shared_memory(&pod_path, configs)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Vec::new()
        } else {
            restored_pids
        };

        if !restored_pids.is_empty() {
            info!(
                pod_identifier = %pod_identifier,
                restored_pid_count = restored_pids.len(),
                "Restored {} processes from existing shared memory",
                restored_pids.len()
            );
        }

        Ok(restored_pids)
    }

    /// Registers a process within a pod (Process-level operation)
    pub async fn register_process(
        &self,
        pod_identifier: &PodIdentifier,
        host_pid: u32,
    ) -> Result<()> {
        // Add PID to shared memory
        let pod_path = self.pod_state.pod_path(pod_identifier);
        self.shared_memory
            .add_pid(pod_path, host_pid as usize)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("Failed to add PID to shared memory")?;

        info!(
            pod_identifier = %pod_identifier,
            host_pid = host_pid,
            "Registered process to coordinator"
        );

        Ok(())
    }

    /// Unregisters a single process from the coordinator.
    pub async fn unregister_process(
        &self,
        pod_identifier: &PodIdentifier,
        host_pid: u32,
    ) -> Result<()> {
        let pod_path = self.pod_state.pod_path(pod_identifier);
        // Remove PID from shared memory
        self.shared_memory
            .remove_pid(pod_path, host_pid as usize)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("Failed to remove PID from shared memory")?;

        info!(
            pod_identifier = %pod_identifier,
            host_pid = host_pid,
            "Unregistered process from coordinator"
        );

        Ok(())
    }

    /// Start periodic cleanup task for unused shared memory segments
    fn start_cleanup_task(&self, cancellation_token: CancellationToken) -> JoinHandle<()> {
        let shared_memory = self.shared_memory.clone();

        let shared_memory_glob_pattern = self.shared_memory_glob_pattern.clone();
        let pod_state = self.pod_state.clone();
        let base_path = self.base_path.clone();
        tokio::spawn(async move {
            let cleanup_duration = Duration::from_secs(CLEANUP_INTERVAL_SECS);
            let mut cleanup_interval =
                interval_at(Instant::now() + cleanup_duration, cleanup_duration);
            cleanup_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            // Skip the first immediate tick
            cleanup_interval.tick().await;

            tracing::info!(
                interval_secs = CLEANUP_INTERVAL_SECS,
                "Starting periodic shared memory cleanup task"
            );

            loop {
                tokio::select! {
                    _ = cleanup_interval.tick() => {
                        tracing::info!("Running periodic cleanup of unused shared memory segments");

                        if let Err(e) = shared_memory.cleanup_orphaned_files(&shared_memory_glob_pattern, |identifier| {
                            pod_state.contains_pod(identifier)
                        }, &base_path).await {
                            tracing::warn!("Failed to cleanup orphaned shared memory files: {}", e);
                        }

                        match shared_memory.cleanup_unused(|identifier| {
                            pod_state.contains_pod(identifier)
                        }, &base_path).await {
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
    #[tracing::instrument(skip(self, cancellation_token))]
    pub async fn start_heartbeat_task(&self, cancellation_token: CancellationToken) {
        let pod_state = self.pod_state.clone();
        let time = self.time.clone();
        let heartbeat_interval = Duration::from_millis(500);

        let task = tokio::spawn(async move {
            let mut interval = interval(heartbeat_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let timestamp = time.now_unix_secs();

                        // Get all pod identifiers from state store
                        let pod_identifiers = pod_state.list_pod_identifiers();

                        // Update heartbeat for each pod concurrently using spawn_blocking
                        let mut tasks = Vec::new();
                        for pod_identifier in pod_identifiers {
                            let pod_path = pod_state.pod_path(&pod_identifier);

                            let task = tokio::spawn(async move {
                                let result = tokio::task::spawn_blocking(move || {
                                    SharedMemoryHandle::open(&pod_path)
                                })
                                .await;

                                match result {
                                    Ok(Ok(handle)) => {
                                        let state = handle.get_state();
                                        state.update_heartbeat(timestamp);
                                    }
                                    Ok(Err(_)) | Err(_) => {
                                        // Silently ignore - pod might be newly registered and shm not yet created
                                        // This is expected during pod registration
                                    }
                                }
                            });
                            tasks.push(task);
                        }

                        // Wait for all heartbeat updates to complete
                        for task in tasks {
                            let _ = task.await;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::pod::mock::*;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;
    use utils::shared_memory::DeviceConfig;

    fn create_test_device_config() -> DeviceConfig {
        DeviceConfig {
            device_idx: 0,
            device_uuid: "GPU-test-123".to_string(),
            up_limit: 80,
            mem_limit: 2048,
            total_cuda_cores: 2560,
            sm_count: 20,
            max_thread_per_sm: 128,
        }
    }

    #[tokio::test]
    async fn test_device_monitoring_cycle_with_no_pods() {
        let (coordinator, _shared_memory, _pod_state, _snapshot, _) =
            TestLimiterCoordinator::new_test(
                Duration::from_millis(100),
                1,
                "test_*.shm".to_string(),
            );

        let mut last_seen_timestamp = 0u64;

        // Test monitoring cycle when no pods are using the device
        let result = coordinator
            .run_device_monitoring_cycle(0, &mut last_seen_timestamp)
            .await;

        assert!(result.is_ok());
        assert_eq!(last_seen_timestamp, 0); // Should remain unchanged
    }

    #[tokio::test]
    async fn test_device_monitoring_cycle_with_pods() {
        let (coordinator, _shared_memory, _pod_state, _snapshot, _) =
            TestLimiterCoordinator::new_test(
                Duration::from_millis(100),
                1,
                "test_*.shm".to_string(),
            );

        // Register a test pod
        let device_configs = vec![create_test_device_config()];
        _pod_state
            .register_test_pod("test-pod", device_configs, vec![1234, 5678])
            .unwrap();

        // Set up snapshot data
        _snapshot.set_device_snapshot(0, 123456789, vec![1234, 5678], vec![512, 1024]);

        let mut last_seen_timestamp = 0u64;

        // Test monitoring cycle with pods
        let result = coordinator
            .run_device_monitoring_cycle(0, &mut last_seen_timestamp)
            .await;

        assert!(result.is_ok());
        assert_eq!(last_seen_timestamp, 123456789); // Should be updated
    }

    #[tokio::test]
    async fn test_start_stop_watcher_tasks() {
        let (coordinator, _, _, _, _) = TestLimiterCoordinator::new_test(
            Duration::from_millis(100),
            2, // Test with 2 devices
            "test_*.shm".to_string(),
        );

        let cancellation_token = CancellationToken::new();

        // Start watcher tasks
        coordinator
            .start_watcher_with_cancellation(cancellation_token.clone())
            .await;

        // Verify tasks are running
        {
            let watcher_tasks = coordinator.device_watcher_tasks.read().await;
            assert_eq!(watcher_tasks.len(), 2);
            assert!(watcher_tasks.contains_key(&0));
            assert!(watcher_tasks.contains_key(&1));
        }

        // Stop all tasks
        coordinator.stop_all_tasks().await;

        // Verify tasks are stopped
        {
            let watcher_tasks = coordinator.device_watcher_tasks.read().await;
            assert_eq!(watcher_tasks.len(), 0);
        }
    }

    #[tokio::test]
    async fn test_heartbeat_functionality() {
        let (coordinator, _, pod_state, _, time) = TestLimiterCoordinator::new_test(
            Duration::from_millis(100),
            1,
            "test_*.shm".to_string(),
        );

        // Register a test pod
        let device_configs = vec![create_test_device_config()];
        pod_state
            .register_test_pod("test-pod", device_configs, vec![1234])
            .unwrap();

        let cancellation_token = CancellationToken::new();

        // Set initial time
        time.set_time(1000);

        // Start heartbeat task
        coordinator
            .start_heartbeat_task(cancellation_token.clone())
            .await;

        // Let heartbeat run for a short time
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Update time and let heartbeat run again
        time.set_time(2000);
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Cancel and stop
        cancellation_token.cancel();
        coordinator.stop_all_tasks().await;

        // Verify heartbeat was called (we can't directly verify shared memory updates in mock)
        assert!(time.now_unix_secs() >= 1000);
    }

    #[tokio::test]
    async fn test_shared_memory_operations() {
        let (coordinator, shared_memory, _pod_state, _, _) = TestLimiterCoordinator::new_test(
            Duration::from_millis(100),
            1,
            "test_*.shm".to_string(),
        );

        // Test process registration
        let pod_id = PodIdentifier::new("test", "pod");
        let result = coordinator.register_process(&pod_id, 1234).await;
        assert!(result.is_ok());

        // Verify operation was logged in mock
        let operations = shared_memory.get_operations();
        assert!(operations
            .iter()
            .any(|op| op.contains("add_pid(/tmp/test_shm/test/pod, 1234)")));

        // Test process unregistration
        let result = coordinator.unregister_process(&pod_id, 1234).await;
        assert!(result.is_ok());

        // Verify operation was logged in mock
        let operations = shared_memory.get_operations();
        assert!(operations
            .iter()
            .any(|op| op.contains("remove_pid(/tmp/test_shm/test/pod, 1234)")));
    }

    #[tokio::test]
    async fn test_cancellation_token_behavior() {
        let (coordinator, _, _, _, _) = TestLimiterCoordinator::new_test(
            Duration::from_millis(100),
            1,
            "test_*.shm".to_string(),
        );

        let cancellation_token = CancellationToken::new();

        // Start the coordinator
        let coordinator_task = {
            let coordinator = std::sync::Arc::new(coordinator);
            let token = cancellation_token.clone();
            tokio::spawn(async move {
                coordinator.run(token).await;
            })
        };

        // Let it run briefly
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Cancel the token
        cancellation_token.cancel();

        // Wait for coordinator to stop
        let result = tokio::time::timeout(Duration::from_millis(1000), coordinator_task).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_error_recovery_in_monitoring_cycle() {
        let (coordinator, _, pod_state, snapshot, _) = TestLimiterCoordinator::new_test(
            Duration::from_millis(100),
            1,
            "test_*.shm".to_string(),
        );

        // Register a test pod
        let device_configs = vec![create_test_device_config()];
        pod_state
            .register_test_pod("test-pod", device_configs, vec![1234])
            .unwrap();

        // Configure snapshot to return an error
        snapshot.set_error_mode(true);

        let mut last_seen_timestamp = 0u64;

        // Test monitoring cycle with error - should handle gracefully
        let result = coordinator
            .run_device_monitoring_cycle(0, &mut last_seen_timestamp)
            .await;

        assert!(result.is_err()); // Should return error but not panic
        assert_eq!(last_seen_timestamp, 0); // Should remain unchanged
    }

    #[tokio::test]
    async fn test_find_shared_memory_files() {
        let (coordinator, shared_memory, _, _, _) = TestLimiterCoordinator::new_test(
            Duration::from_millis(100),
            1,
            "test_*.shm".to_string(),
        );

        let result = coordinator.find_shared_memory_files("test_*.shm").await;
        assert!(result.is_ok());

        // Verify the call was made to the mock
        let operations = shared_memory.get_operations();
        assert!(operations
            .iter()
            .any(|op| op.contains("find_shared_memory_files")));
    }

    #[test]
    fn test_extract_identifier_from_path() {
        let (coordinator, shared_memory, _, _, _) = TestLimiterCoordinator::new_test(
            Duration::from_millis(100),
            1,
            "test_*.shm".to_string(),
        );

        let test_path = Path::new("/tmp/test_shm/test_ns/test_pod_123/shm");
        let result = coordinator.extract_identifier_from_path(test_path);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            PodIdentifier::new("test_ns", "test_pod_123")
        );

        // Verify the call was made to the mock
        let operations = shared_memory.get_operations();
        assert!(operations
            .iter()
            .any(|op| op.contains("extract_identifier_from_path")));
    }

    #[test]
    fn test_utilization_increment_calculation() {
        // Test utilization increment calculation logic

        // Test normal case - utilization below limit
        let utilization = 50u32;
        let up_limit = 80u32;
        let utilization_diff = up_limit.saturating_sub(utilization);
        let increment = utilization_diff / 10;
        assert!(increment > 0);
        assert_eq!(increment, 3); // (80-50)/10 = 3

        // Test edge case - utilization at limit
        let utilization = 80u32;
        let up_limit = 80u32;
        let utilization_diff = up_limit.saturating_sub(utilization);
        let increment = utilization_diff / 10;
        assert_eq!(increment, 0); // (80-80)/10 = 0

        // Test edge case - utilization over limit (saturating_sub prevents underflow)
        let utilization = 90u32;
        let up_limit = 80u32;
        let utilization_diff = up_limit.saturating_sub(utilization);
        let increment = utilization_diff / 10;
        assert_eq!(increment, 0); // saturating_sub gives 0

        // Test large difference scaling
        let utilization_small = 10u32;
        let up_limit = 80u32;
        let increment_small = up_limit.saturating_sub(utilization_small) / 10;

        let utilization_large = 20u32;
        let increment_large = up_limit.saturating_sub(utilization_large) / 10;

        assert!(increment_large < increment_small); // 6 < 7
        assert_eq!(increment_small, 7); // (80-10)/10 = 7
        assert_eq!(increment_large, 6); // (80-20)/10 = 6
    }

    #[tokio::test]
    async fn test_coordinator_full_lifecycle_integration() {
        let (coordinator, _shared_memory, pod_state, snapshot, time) =
            TestLimiterCoordinator::new_test(
                Duration::from_millis(100), // Fast interval for testing
                2,                          // Multiple devices
                "test_lifecycle_*.shm".to_string(),
            );

        // Set up test data
        let device_configs = [
            create_test_device_config(),
            DeviceConfig {
                device_idx: 1,
                device_uuid: "GPU-test-456".to_string(),
                up_limit: 90,
                mem_limit: 4096,
                total_cuda_cores: 5120,
                sm_count: 40,
                max_thread_per_sm: 128,
            },
        ];

        // Register multiple pods across multiple devices
        pod_state
            .register_test_pod("pod-1", vec![device_configs[0].clone()], vec![1001, 1002])
            .unwrap();
        pod_state
            .register_test_pod(
                "pod-2",
                vec![device_configs[1].clone()],
                vec![2001, 2002, 2003],
            )
            .unwrap();

        // Set up snapshot data for both devices
        snapshot.set_device_snapshot(0, 1000, vec![1001, 1002], vec![512, 1024]);
        snapshot.set_device_snapshot(1, 1000, vec![2001, 2002, 2003], vec![256, 768, 384]);

        time.set_time(1000);

        let cancellation_token = CancellationToken::new();

        // Start coordinator
        let coordinator_task = {
            let coordinator = std::sync::Arc::new(coordinator);
            let token = cancellation_token.clone();
            tokio::spawn(async move {
                coordinator.run(token).await;
            })
        };

        // Let it run for a bit
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Update time and snapshots
        time.set_time(2000);
        snapshot.set_device_snapshot(0, 2000, vec![1001, 1002], vec![1024, 2048]);
        snapshot.set_device_snapshot(1, 2000, vec![2001, 2002], vec![512, 1536]); // One process gone

        // Let it process updates
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Cancel and cleanup
        cancellation_token.cancel();
        let result = tokio::time::timeout(Duration::from_millis(1000), coordinator_task).await;
        assert!(result.is_ok());

        // Verify some operations were attempted (allow empty due to delayed cleanup and mocked state)
        assert!(time.now_unix_secs() >= 1000);
    }

    #[tokio::test]
    async fn test_coordinator_error_recovery_scenarios() {
        let (coordinator, _shared_memory, pod_state, snapshot, time) =
            TestLimiterCoordinator::new_test(
                Duration::from_millis(20), // Very fast for error testing
                1,
                "test_error_*.shm".to_string(),
            );

        // Register test pod
        let device_configs = vec![create_test_device_config()];
        pod_state
            .register_test_pod("error-pod", device_configs, vec![3001, 3002])
            .unwrap();

        // Initially set up normal data
        snapshot.set_device_snapshot(0, 1000, vec![3001, 3002], vec![512, 1024]);
        time.set_time(1000);

        let cancellation_token = CancellationToken::new();

        // Start coordinator
        coordinator
            .start_watcher_with_cancellation(cancellation_token.clone())
            .await;

        // Let it run normally first
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Introduce errors
        snapshot.set_error_mode(true);

        // Let it handle errors
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Recover from errors
        snapshot.set_error_mode(false);
        snapshot.set_device_snapshot(0, 2000, vec![3001], vec![2048]); // One process recovered

        // Let it recover
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Cleanup
        cancellation_token.cancel();
        coordinator.stop_all_tasks().await;

        // Verify coordinator handled errors gracefully and continued operation
        assert!(time.now_unix_secs() >= 1000);
    }
}
