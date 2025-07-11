//! Limiter Coordinator Module
//!
//! This module provides functionality for coordinating GPU resource limits across pods and containers.
//! The coordinator manages shared memory segments at the pod level, allowing all containers within
//! a pod to share the same GPU core allocation.
//!
//! # Pod-Level Resource Sharing
//!
//! The coordinator uses pod names as shared memory identifiers, enabling containers within the same
//! pod to share GPU core quotas. This is particularly useful for multi-container pods where
//! containers need to coordinate their GPU usage.
//!
//! # Example Usage
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use std::time::Duration;
//! # use utils::shared_memory::DeviceConfig;
//!
//! // Create a coordinator with device count (automatically starts monitoring tasks)
//! let coordinator = LimiterCoordinator::new(Duration::from_secs(1), 4); // 4 GPUs
//!
//! // Configure device limits
//! let device_config = DeviceConfig {
//!     device_idx: 0,
//!     up_limit: 80,                      // 80% utilization limit
//!     mem_limit: 8 * 1024 * 1024 * 1024, // 8GB memory limit
//!     total_cuda_cores: 2048,
//! };
//!
//! // Register containers from the same pod
//! coordinator.register_device("my-pod", "container-1", 1001, 12345, device_config.clone())?;
//! coordinator.register_device("my-pod", "container-2", 1002, 12346, device_config.clone())?;
//!
//! // Both containers now share the same GPU core allocation through shared memory
//! // identified by the pod name "my-pod"
//! // The coordinator automatically monitors and adjusts GPU core allocation
//!
//! // Query pod usage
//! let pod_usage = coordinator.get_pod_usage("my-pod");
//! let container_count = coordinator.get_pod_container_count("my-pod");
//!
//! // Get shared memory access for a specific container
//! let shared_state = coordinator.get_shared_memory_for_container("my-pod", "container-1")?;
//!
//! // Unregister containers when done
//! coordinator.unregister_device("my-pod", "container-1", 1001)?;
//! coordinator.unregister_device("my-pod", "container-2", 1002)?;
//! // Shared memory is automatically cleaned up when the last container is unregistered
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! # Key Features
//!
//! - **Pod-Level Sharing**: Containers within the same pod share GPU core allocation
//! - **Automatic Cleanup**: Shared memory is cleaned up when the last container is unregistered
//! - **Container Tracking**: Track which containers are using each pod's shared memory
//! - **Dynamic Adjustment**: GPU core allocation is adjusted based on utilization monitoring
//!
//! # Integration with WorkerManager
//!
//! The coordinator integrates with the WorkerManager to get pod and container information:
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # type AddCB = fn(u32, Arc<TensorFusionWorker>);
//! # type RemoveCB = fn(u32);
//! # let worker_manager: Arc<WorkerManager<AddCB, RemoveCB>> = unimplemented!();
//!
//! // Get pod information from host PID
//! let pod_name = worker_manager.get_pod_name_by_host_pid(12345).await;
//! let namespace = worker_manager.get_namespace_by_host_pid(12345).await;
//! let container_name = worker_manager.get_container_name_by_host_pid(12345).await;
//!
//! // Use this information to register with the coordinator
//! if let (Some(pod_name), Some(namespace), Some(container_name)) = (pod_name, namespace, container_name) {
//!     // Register the device using pod name for shared memory
//!     // coordinator.register_device(&pod_name, &container_name, container_pid, host_pid, device_config)?;
//! }
//! ```

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use tokio::task::JoinHandle;
use tokio::time::interval;
use tokio::time::sleep;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;
use utils::shared_memory::DeviceConfig;
use utils::shared_memory::SharedDeviceState;
use utils::shared_memory::ThreadSafeSharedMemoryManager;

/// Tracks device usage at the pod level.
#[derive(Debug, Clone)]
pub struct PodDeviceUsage {
    pub pod_name: String,
    pub device_config: DeviceConfig,
    /// Information about containers in this pod that are using the device: container_name -> (container_pid, host_pid)
    pub active_containers: HashMap<String, (u32, u32)>,
}

impl PodDeviceUsage {
    fn new(pod_name: String, device_config: DeviceConfig) -> Self {
        Self {
            pod_name,
            device_config,
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

    fn has_containers(&self) -> bool {
        !self.active_containers.is_empty()
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
    /// Active pod device usage: pod_name -> PodDeviceUsage
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
        let coordinator = Self {
            shared_memory_manager: Arc::new(ThreadSafeSharedMemoryManager::new()),
            active_pods: Arc::new(RwLock::new(HashMap::new())),
            device_watcher_tasks: RwLock::new(HashMap::new()),
            watch_interval,
            device_count,
        };

        // Start the global monitoring task.
        coordinator.start_watcher();

        coordinator
    }

    /// Starts a monitoring task for each GPU device.
    fn start_watcher(&self) {
        let watch_interval = self.watch_interval;
        let active_pods = self.active_pods.clone();
        let shared_memory_manager = self.shared_memory_manager.clone();

        // Start a monitoring task for each GPU device.
        for device_idx in 0..self.device_count {
            let device_idx = device_idx;
            let watch_interval = watch_interval;
            let active_pods = active_pods.clone();
            let shared_memory_manager = shared_memory_manager.clone();

            let task = tokio::spawn(async move {
                let mut device_states: HashMap<String, (u64, i32)> = HashMap::new(); // pod_name -> (last_seen_timestamp, share)
                let mut interval_timer = interval(watch_interval);

                info!(
                    device_idx = device_idx,
                    "Starting device watcher task for GPU device"
                );

                loop {
                    interval_timer.tick().await;

                    // Get all pods using this device.
                    let pods_for_device: Vec<(String, PodDeviceUsage)> = {
                        let pods = active_pods.read().unwrap();
                        pods.iter()
                            .filter(|(_, usage)| usage.device_config.device_idx == device_idx)
                            .map(|(name, usage)| (name.clone(), usage.clone()))
                            .collect()
                    };

                    if pods_for_device.is_empty() {
                        debug!(device_idx = device_idx, "No pods using this device");
                        continue;
                    }

                    // Monitor each pod.
                    for (pod_name, pod_usage) in pods_for_device {
                        let host_pids = pod_usage.get_host_pids();

                        if host_pids.is_empty() {
                            debug!(pod_name = %pod_name, device_idx = device_idx, "No active processes to monitor");
                            continue;
                        }

                        // Get or initialize device state.
                        let (last_seen_timestamp, current_share) = device_states
                            .entry(pod_name.clone())
                            .or_insert((0, pod_usage.device_config.total_cuda_cores as i32));

                        // Get GPU utilization data.
                        match Self::get_pod_utilization_with_retry(
                            device_idx,
                            &host_pids,
                            *last_seen_timestamp,
                            Duration::from_millis(100),
                        )
                        .await
                        {
                            Ok((utilization, timestamp)) => {
                                *last_seen_timestamp = timestamp;

                                // Calculate the new share value.
                                let new_share = Self::calculate_new_share(
                                    &pod_usage.device_config,
                                    utilization.user_current,
                                    *current_share,
                                );

                                // If the share value has changed, update the shared memory.
                                if new_share != *current_share {
                                    match Self::update_shared_memory_state(
                                        &pod_name,
                                        &pod_usage.device_config,
                                        utilization.user_current,
                                        new_share,
                                    )
                                    .await
                                    {
                                        Ok(()) => {
                                            debug!(
                                                pod_name = %pod_name,
                                                device_idx = device_idx,
                                                old_share = *current_share,
                                                new_share = new_share,
                                                user_utilization = utilization.user_current,
                                                sys_utilization = utilization.sys_current,
                                                "Updated GPU core allocation"
                                            );
                                            *current_share = new_share;
                                        }
                                        Err(e) => {
                                            error!(
                                                pod_name = %pod_name,
                                                device_idx = device_idx,
                                                error = %e,
                                                "Failed to update shared memory state"
                                            );
                                        }
                                    }
                                } else {
                                    debug!(
                                        pod_name = %pod_name,
                                        device_idx = device_idx,
                                        share = *current_share,
                                        user_utilization = utilization.user_current,
                                        sys_utilization = utilization.sys_current,
                                        "GPU core allocation unchanged"
                                    );
                                }
                            }
                            Err(e) => {
                                error!(
                                    pod_name = %pod_name,
                                    device_idx = device_idx,
                                    error = %e,
                                    "Failed to get GPU utilization"
                                );
                            }
                        }
                    }
                }
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
        pod_name: &str,
        container_name: &str,
        container_pid: u32,
        host_pid: u32,
        config: DeviceConfig,
    ) -> Result<()> {
        // Use the pod_name as the shared memory identifier.
        let pod_name_str = pod_name.to_string();

        // Create the shared memory.
        self.shared_memory_manager
            .create_or_get_shared_memory(&pod_name_str, &config)?;

        // Update the pod device usage.
        {
            let mut active_pods = self.active_pods.write().unwrap();
            let pod_usage = active_pods
                .entry(pod_name_str.clone())
                .or_insert_with(|| PodDeviceUsage::new(pod_name_str.clone(), config.clone()));

            pod_usage.add_container(container_name.to_string(), container_pid, host_pid);
        }

        info!(
            pod_name = %pod_name,
            container_name = %container_name,
            container_pid = container_pid,
            host_pid = host_pid,
            device_idx = config.device_idx,
            "Registered device to coordinator"
        );

        Ok(())
    }

    /// Unregisters a device from the coordinator.
    pub fn unregister_device(
        &self,
        pod_name: &str,
        container_name: &str,
        container_pid: u32,
    ) -> Result<()> {
        let pod_name_str = pod_name.to_string();

        let should_cleanup = {
            let mut active_pods = self.active_pods.write().unwrap();
            if let Some(pod_usage) = active_pods.get_mut(&pod_name_str) {
                let is_empty = pod_usage.remove_container(container_name);
                if is_empty {
                    active_pods.remove(&pod_name_str);
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
            self.shared_memory_manager.cleanup(&pod_name_str)?;

            info!(
                pod_name = %pod_name,
                "Cleaned up pod shared memory"
            );
        }

        info!(
            pod_name = %pod_name,
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
                let process_utilization_samples = dev
                    .process_utilization_stats(last_seen_timestamp)
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

    /// 计算核心调整值
    fn calculate_delta_from_ptr(
        shared_state_ptr: *mut SharedDeviceState,
        user_current: u32,
        share: i32,
        up_limit: u32,
    ) -> i32 {
        let up_limit = up_limit as i32;
        let total_cuda_cores =
            unsafe { (*shared_state_ptr).total_cuda_cores.load(Ordering::Acquire) as i32 };

        // 计算利用率差异
        let utilization_diff = if (up_limit - user_current as i32).abs() < 5 {
            5
        } else {
            (up_limit - user_current as i32).abs()
        };

        // 计算增量
        let increment =
            Self::calculate_increment_from_ptr(shared_state_ptr, utilization_diff, up_limit);

        // 确定调整方向
        if user_current <= up_limit as u32 {
            // 利用率低于限制，增加份额
            if share + increment > total_cuda_cores {
                total_cuda_cores
            } else {
                share + increment
            }
        } else {
            // 利用率高于限制，减少份额
            if share - increment < 0 {
                0
            } else {
                share - increment
            }
        }
    }

    /// 计算调整增量
    fn calculate_increment_from_ptr(
        shared_state_ptr: *mut SharedDeviceState,
        utilization_diff: i32,
        up_limit: i32,
    ) -> i32 {
        // 简化的计算逻辑，基于总核心数的比例
        let total_cores =
            unsafe { (*shared_state_ptr).total_cuda_cores.load(Ordering::Acquire) as i32 };
        let mut increment = total_cores / 100 * utilization_diff / 10;

        // 当差异很大时，加大调整幅度
        if utilization_diff > up_limit / 2 {
            increment = increment * utilization_diff * 2 / (up_limit + 1);
        }

        increment.max(1) // 至少调整1个核心
    }

    /// 应用核心调整
    fn apply_core_adjustment_from_ptr(shared_state_ptr: *mut SharedDeviceState, new_share: i32) {
        let total_cores =
            unsafe { (*shared_state_ptr).total_cuda_cores.load(Ordering::Acquire) as i32 };
        let clamped_share = new_share.max(0).min(total_cores);

        unsafe {
            (*shared_state_ptr)
                .available_cuda_cores
                .store(clamped_share, Ordering::Release)
        };
    }

    /// 更新共享内存状态（异步安全版本）
    async fn update_shared_memory_state(
        pod_name: &str,
        device_config: &DeviceConfig,
        user_current: u32,
        current_share: i32,
    ) -> Result<()> {
        tokio::task::spawn_blocking({
            let pod_name = pod_name.to_string();
            let device_config = device_config.clone();
            move || -> Result<()> {
                // 在阻塞任务中打开共享内存
                use utils::shared_memory::SharedMemoryHandle;
                let handle =
                    SharedMemoryHandle::open(&pod_name).context("Failed to open shared memory")?;

                let shared_state_ptr = handle.get_ptr();

                // 获取当前状态
                let up_limit = unsafe { (*shared_state_ptr).up_limit.load(Ordering::Acquire) };
                let total_cuda_cores =
                    unsafe { (*shared_state_ptr).total_cuda_cores.load(Ordering::Acquire) as i32 };

                // 计算新的share值
                let new_share = calculate_delta_blocking(
                    total_cuda_cores,
                    user_current,
                    current_share,
                    up_limit,
                );

                // 应用调整
                let clamped_share = new_share.max(0).min(total_cuda_cores);
                unsafe {
                    (*shared_state_ptr)
                        .available_cuda_cores
                        .store(clamped_share, Ordering::Release)
                };

                Ok(())
            }
        })
        .await
        .context("Blocking task failed")?
    }

    /// 计算新的share值
    fn calculate_new_share(
        device_config: &DeviceConfig,
        user_current: u32,
        current_share: i32,
    ) -> i32 {
        let up_limit = device_config.up_limit as i32;
        let total_cuda_cores = device_config.total_cuda_cores as i32;

        // 计算利用率差异
        let utilization_diff = if (up_limit - user_current as i32).abs() < 5 {
            5
        } else {
            (up_limit - user_current as i32).abs()
        };

        // 计算增量
        let increment = Self::calculate_increment(device_config, utilization_diff, up_limit);

        // 确定调整方向
        if user_current <= up_limit as u32 {
            // 利用率低于限制，增加份额
            if current_share + increment > total_cuda_cores {
                total_cuda_cores
            } else {
                current_share + increment
            }
        } else {
            // 利用率高于限制，减少份额
            if current_share - increment < 0 {
                0
            } else {
                current_share - increment
            }
        }
    }

    /// 计算调整增量
    fn calculate_increment(
        device_config: &DeviceConfig,
        utilization_diff: i32,
        up_limit: i32,
    ) -> i32 {
        // 简化的计算逻辑，基于总核心数的比例
        let total_cores = device_config.total_cuda_cores as i32;
        let mut increment = total_cores / 100 * utilization_diff / 10;

        // 当差异很大时，加大调整幅度
        if utilization_diff > up_limit / 2 {
            increment = increment * utilization_diff * 2 / (up_limit + 1);
        }

        increment.max(1) // 至少调整1个核心
    }

    /// 获取设备状态
    pub fn get_device_state(&self, pod_name: &str) -> Result<*mut SharedDeviceState> {
        let active_pods = self.active_pods.read().unwrap();

        if active_pods.contains_key(pod_name) {
            self.shared_memory_manager.get_shared_memory(pod_name)
        } else {
            Err(anyhow::anyhow!("Pod not found: {}", pod_name))
        }
    }

    /// 获取所有活跃设备的配置
    pub fn get_all_device_configs(&self) -> HashMap<String, DeviceConfig> {
        let active_pods = self.active_pods.read().unwrap();
        let mut all_configs = HashMap::new();
        for (_, pod_usage) in active_pods.iter() {
            all_configs.insert(pod_usage.pod_name.clone(), pod_usage.device_config.clone());
        }
        all_configs
    }

    /// 获取所有活跃的Pod使用情况
    pub fn get_all_pod_usage(&self) -> HashMap<String, PodDeviceUsage> {
        let active_pods = self.active_pods.read().unwrap();
        active_pods.clone()
    }

    /// 获取特定Pod的使用情况
    pub fn get_pod_usage(&self, pod_name: &str) -> Option<PodDeviceUsage> {
        let active_pods = self.active_pods.read().unwrap();
        active_pods.get(pod_name).cloned()
    }

    /// 检查Pod是否存在
    pub fn pod_exists(&self, pod_name: &str) -> bool {
        let active_pods = self.active_pods.read().unwrap();
        active_pods.contains_key(pod_name)
    }

    /// 获取Pod中活跃的容器数量
    pub fn get_pod_container_count(&self, pod_name: &str) -> Option<usize> {
        let active_pods = self.active_pods.read().unwrap();
        active_pods
            .get(pod_name)
            .map(|usage| usage.active_containers.len())
    }

    /// 获取设备的可用核心数
    pub fn get_available_cores(&self, pod_name: &str) -> Result<i32> {
        let shared_state_ptr = self.get_device_state(pod_name)?;
        let cores = unsafe {
            (*shared_state_ptr)
                .available_cuda_cores
                .load(Ordering::Acquire)
        };
        Ok(cores)
    }

    /// 更新设备的利用率限制
    pub fn update_up_limit(&self, pod_name: &str, new_limit: u32) -> Result<()> {
        let shared_state_ptr = self.get_device_state(pod_name)?;
        unsafe {
            (*shared_state_ptr)
                .up_limit
                .store(new_limit, Ordering::Release)
        };
        Ok(())
    }

    /// 更新设备的内存限制
    pub fn update_mem_limit(&self, pod_name: &str, new_limit: u64) -> Result<()> {
        let shared_state_ptr = self.get_device_state(pod_name)?;
        unsafe {
            (*shared_state_ptr)
                .mem_limit
                .store(new_limit, Ordering::Release)
        };
        Ok(())
    }

    /// 为Pod中的特定容器获取共享内存访问
    pub fn get_shared_memory_for_container(
        &self,
        pod_name: &str,
        container_name: &str,
    ) -> Result<*mut SharedDeviceState> {
        let active_pods = self.active_pods.read().unwrap();
        if let Some(pod_usage) = active_pods.get(pod_name) {
            if pod_usage.active_containers.contains_key(container_name) {
                drop(active_pods); // 释放锁
                self.shared_memory_manager.get_shared_memory(pod_name)
            } else {
                Err(anyhow::anyhow!(
                    "Container '{}' not found in pod '{}'",
                    container_name,
                    pod_name
                ))
            }
        } else {
            Err(anyhow::anyhow!("Pod not found: {}", pod_name))
        }
    }
}

impl Drop for LimiterCoordinator {
    fn drop(&mut self) {
        // 停止所有监控任务
        let mut watcher_tasks = self.device_watcher_tasks.write().unwrap();
        for (_, task) in watcher_tasks.drain() {
            task.abort();
        }

        info!("LimiterCoordinator dropped, all watcher tasks stopped");
    }
}

/// 计算核心调整值
fn calculate_delta_blocking(
    total_cuda_cores: i32,
    user_current: u32,
    share: i32,
    up_limit: u32,
) -> i32 {
    let up_limit = up_limit as i32;

    // 计算利用率差异
    let utilization_diff = if (up_limit - user_current as i32).abs() < 5 {
        5
    } else {
        (up_limit - user_current as i32).abs()
    };

    // 计算增量
    let mut increment = total_cuda_cores / 100 * utilization_diff / 10;

    // 当差异很大时，加大调整幅度
    if utilization_diff > up_limit / 2 {
        increment = increment * utilization_diff * 2 / (up_limit + 1);
    }

    let increment = increment.max(1); // 至少调整1个核心

    // 确定调整方向
    if user_current <= up_limit as u32 {
        // 利用率低于限制，增加份额
        if share + increment > total_cuda_cores {
            total_cuda_cores
        } else {
            share + increment
        }
    } else {
        // 利用率高于限制，减少份额
        if share - increment < 0 {
            0
        } else {
            share - increment
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::time::sleep;
    use utils::shared_memory::DeviceConfig;

    use super::*;

    /// 创建测试用的设备配置
    fn create_test_device_config(device_idx: u32) -> DeviceConfig {
        DeviceConfig {
            device_idx,
            up_limit: 80,
            mem_limit: 8 * 1024 * 1024 * 1024, // 8GB
            total_cuda_cores: 2048,
        }
    }

    /// 创建测试协调器
    fn create_test_coordinator() -> LimiterCoordinator {
        LimiterCoordinator::new(Duration::from_millis(100), 1) // 假设只有一个GPU设备
    }

    /// 创建唯一的测试Pod名称
    fn create_unique_test_pod_name(test_name: &str) -> String {
        use std::time::SystemTime;
        use std::time::UNIX_EPOCH;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{}-{}", test_name, timestamp)
    }

    #[tokio::test]
    async fn test_single_pod_single_container() {
        let coordinator = create_test_coordinator();
        let config = create_test_device_config(0);
        let pod_name = create_unique_test_pod_name("single-pod-single-container");

        // 注册设备
        coordinator
            .register_device(&pod_name, "container-1", 1001, 12345, config.clone())
            .unwrap();

        // 验证Pod存在
        assert!(coordinator.pod_exists(&pod_name));
        assert_eq!(coordinator.get_pod_container_count(&pod_name), Some(1));

        // 验证可以获取共享内存
        let shared_state = coordinator
            .get_shared_memory_for_container(&pod_name, "container-1")
            .unwrap();
        assert!(!shared_state.is_null());

        // 验证设备配置
        let pod_usage = coordinator.get_pod_usage(&pod_name).unwrap();
        assert_eq!(pod_usage.device_config.device_idx, 0);
        assert_eq!(pod_usage.device_config.up_limit, 80);
        assert_eq!(pod_usage.active_containers.len(), 1);

        // 注销设备
        coordinator
            .unregister_device(&pod_name, "container-1", 1001)
            .unwrap();

        // 验证Pod已清理
        assert!(!coordinator.pod_exists(&pod_name));
        assert_eq!(coordinator.get_pod_container_count(&pod_name), None);
    }

    #[tokio::test]
    async fn test_single_pod_multiple_containers() {
        let coordinator = create_test_coordinator();
        let config = create_test_device_config(0);
        let pod_name = create_unique_test_pod_name("single-pod-multiple-containers");

        // 注册多个容器到同一个Pod
        coordinator
            .register_device(&pod_name, "container-1", 1001, 12345, config.clone())
            .unwrap();
        coordinator
            .register_device(&pod_name, "container-2", 1002, 12346, config.clone())
            .unwrap();
        coordinator
            .register_device(&pod_name, "container-3", 1003, 12347, config.clone())
            .unwrap();

        // 验证Pod状态
        assert!(coordinator.pod_exists(&pod_name));
        assert_eq!(coordinator.get_pod_container_count(&pod_name), Some(3));

        // 验证所有容器都能访问共享内存
        for container_name in ["container-1", "container-2", "container-3"] {
            let shared_state = coordinator
                .get_shared_memory_for_container(&pod_name, container_name)
                .unwrap();
            assert!(!shared_state.is_null());
        }

        // 验证Pod使用情况
        let pod_usage = coordinator.get_pod_usage(&pod_name).unwrap();
        assert_eq!(pod_usage.active_containers.len(), 3);
        assert!(pod_usage.active_containers.contains_key("container-1"));
        assert!(pod_usage.active_containers.contains_key("container-2"));
        assert!(pod_usage.active_containers.contains_key("container-3"));

        // 逐个注销容器
        coordinator
            .unregister_device(&pod_name, "container-1", 1001)
            .unwrap();
        assert!(coordinator.pod_exists(&pod_name));
        assert_eq!(coordinator.get_pod_container_count(&pod_name), Some(2));

        coordinator
            .unregister_device(&pod_name, "container-2", 1002)
            .unwrap();
        assert!(coordinator.pod_exists(&pod_name));
        assert_eq!(coordinator.get_pod_container_count(&pod_name), Some(1));

        // 最后一个容器注销后，Pod应该被清理
        coordinator
            .unregister_device(&pod_name, "container-3", 1003)
            .unwrap();
        assert!(!coordinator.pod_exists(&pod_name));
        assert_eq!(coordinator.get_pod_container_count(&pod_name), None);
    }

    #[tokio::test]
    async fn test_multiple_pods_multiple_containers() {
        let coordinator = create_test_coordinator();
        let config1 = create_test_device_config(0);
        let config2 = create_test_device_config(1);
        let pod_name1 = create_unique_test_pod_name("multiple-pods-1");
        let pod_name2 = create_unique_test_pod_name("multiple-pods-2");

        // 注册多个Pod，每个Pod有多个容器
        coordinator
            .register_device(&pod_name1, "container-1", 1001, 12345, config1.clone())
            .unwrap();
        coordinator
            .register_device(&pod_name1, "container-2", 1002, 12346, config1.clone())
            .unwrap();
        coordinator
            .register_device(&pod_name2, "container-1", 2001, 22345, config2.clone())
            .unwrap();
        coordinator
            .register_device(&pod_name2, "container-2", 2002, 22346, config2.clone())
            .unwrap();

        // 验证所有Pod都存在
        assert!(coordinator.pod_exists(&pod_name1));
        assert!(coordinator.pod_exists(&pod_name2));
        assert_eq!(coordinator.get_pod_container_count(&pod_name1), Some(2));
        assert_eq!(coordinator.get_pod_container_count(&pod_name2), Some(2));

        // 验证Pod间的独立性
        let pod1_usage = coordinator.get_pod_usage(&pod_name1).unwrap();
        let pod2_usage = coordinator.get_pod_usage(&pod_name2).unwrap();
        assert_eq!(pod1_usage.device_config.device_idx, 0);
        assert_eq!(pod2_usage.device_config.device_idx, 1);

        // 验证共享内存独立性
        let pod1_shared = coordinator
            .get_shared_memory_for_container(&pod_name1, "container-1")
            .unwrap();
        let pod2_shared = coordinator
            .get_shared_memory_for_container(&pod_name2, "container-1")
            .unwrap();
        assert_ne!(pod1_shared, pod2_shared);

        // 获取所有Pod使用情况
        let all_usage = coordinator.get_all_pod_usage();
        assert_eq!(all_usage.len(), 2);
        assert!(all_usage.contains_key(&pod_name1));
        assert!(all_usage.contains_key(&pod_name2));

        // 清理所有容器
        coordinator
            .unregister_device(&pod_name1, "container-1", 1001)
            .unwrap();
        coordinator
            .unregister_device(&pod_name1, "container-2", 1002)
            .unwrap();
        coordinator
            .unregister_device(&pod_name2, "container-1", 2001)
            .unwrap();
        coordinator
            .unregister_device(&pod_name2, "container-2", 2002)
            .unwrap();

        // 验证所有Pod都已清理
        assert!(!coordinator.pod_exists(&pod_name1));
        assert!(!coordinator.pod_exists(&pod_name2));
    }

    #[tokio::test]
    async fn test_sequential_registration_and_unregistration() {
        let coordinator = create_test_coordinator();
        let config = create_test_device_config(0);

        // 顺序注册和注销多个Pod
        for i in 0..10 {
            let pod_name = create_unique_test_pod_name(&format!("sequential-{}", i));
            let container_name = format!("container-{}", i);
            let container_pid = 1000 + i as u32;
            let host_pid = 12000 + i as u32;

            // 注册设备
            coordinator
                .register_device(
                    &pod_name,
                    &container_name,
                    container_pid,
                    host_pid,
                    config.clone(),
                )
                .unwrap();

            // 验证注册成功
            assert!(coordinator.pod_exists(&pod_name));

            // 短暂等待
            sleep(Duration::from_millis(10)).await;

            // 注销设备
            coordinator
                .unregister_device(&pod_name, &container_name, container_pid)
                .unwrap();

            // 验证注销成功
            assert!(!coordinator.pod_exists(&pod_name));
        }
    }

    #[tokio::test]
    async fn test_error_handling_invalid_operations() {
        let coordinator = create_test_coordinator();
        let config = create_test_device_config(0);
        let pod_name = create_unique_test_pod_name("error-handling");

        // 尝试获取不存在的Pod的共享内存
        let result = coordinator.get_shared_memory_for_container("non-existent-pod", "container-1");
        assert!(result.is_err());

        // 尝试注销不存在的设备
        let result = coordinator.unregister_device("non-existent-pod", "container-1", 1001);
        assert!(result.is_ok()); // 应该优雅地处理

        // 注册设备后尝试获取不存在的容器
        coordinator
            .register_device(&pod_name, "container-1", 1001, 12345, config.clone())
            .unwrap();

        let result =
            coordinator.get_shared_memory_for_container(&pod_name, "non-existent-container");
        assert!(result.is_err());

        // 清理
        coordinator
            .unregister_device(&pod_name, "container-1", 1001)
            .unwrap();
    }

    #[tokio::test]
    async fn test_shared_memory_state_updates() {
        let coordinator = create_test_coordinator();
        let config = create_test_device_config(0);
        let pod_name = create_unique_test_pod_name("shared-memory-updates");

        // 注册设备
        coordinator
            .register_device(&pod_name, "container-1", 1001, 12345, config.clone())
            .unwrap();

        // 测试更新利用率限制
        coordinator.update_up_limit(&pod_name, 90).unwrap();

        // 测试更新内存限制
        let new_mem_limit = 16 * 1024 * 1024 * 1024u64; // 16GB
        coordinator
            .update_mem_limit(&pod_name, new_mem_limit)
            .unwrap();

        // 验证更新
        let shared_state = coordinator.get_device_state(&pod_name).unwrap();
        unsafe {
            assert_eq!((*shared_state).up_limit.load(Ordering::Acquire), 90);
            assert_eq!(
                (*shared_state).mem_limit.load(Ordering::Acquire),
                new_mem_limit
            );
        }

        // 清理
        coordinator
            .unregister_device(&pod_name, "container-1", 1001)
            .unwrap();
    }

    #[tokio::test]
    async fn test_core_allocation_boundaries() {
        let coordinator = create_test_coordinator();
        let config = create_test_device_config(0);
        let pod_name = create_unique_test_pod_name("core-allocation");

        // 注册设备
        coordinator
            .register_device(&pod_name, "container-1", 1001, 12345, config.clone())
            .unwrap();

        // 测试获取可用核心数
        let cores = coordinator.get_available_cores(&pod_name).unwrap();
        assert!(cores >= 0);
        assert!(cores <= config.total_cuda_cores as i32);

        // 测试计算新的share值的边界条件
        let new_share_low = LimiterCoordinator::calculate_new_share(&config, 0, 1000);
        let new_share_high = LimiterCoordinator::calculate_new_share(&config, 100, 1000);
        let new_share_normal = LimiterCoordinator::calculate_new_share(&config, 80, 1000);

        assert!(new_share_low >= 0);
        assert!(new_share_high >= 0);
        assert!(new_share_normal >= 0);
        assert!(new_share_low <= config.total_cuda_cores as i32);
        assert!(new_share_high <= config.total_cuda_cores as i32);
        assert!(new_share_normal <= config.total_cuda_cores as i32);

        // 清理
        coordinator
            .unregister_device(&pod_name, "container-1", 1001)
            .unwrap();
    }

    #[tokio::test]
    async fn test_complex_lifecycle_scenario() {
        let coordinator = create_test_coordinator();
        let config = create_test_device_config(0);
        let pod_name = create_unique_test_pod_name("complex-lifecycle");

        // 场景1: 注册多个容器，然后部分注销，再重新注册
        coordinator
            .register_device(&pod_name, "container-1", 1001, 12345, config.clone())
            .unwrap();
        coordinator
            .register_device(&pod_name, "container-2", 1002, 12346, config.clone())
            .unwrap();
        coordinator
            .register_device(&pod_name, "container-3", 1003, 12347, config.clone())
            .unwrap();

        assert_eq!(coordinator.get_pod_container_count(&pod_name), Some(3));

        // 注销中间的容器
        coordinator
            .unregister_device(&pod_name, "container-2", 1002)
            .unwrap();
        assert_eq!(coordinator.get_pod_container_count(&pod_name), Some(2));

        // 重新注册新容器
        coordinator
            .register_device(&pod_name, "container-4", 1004, 12348, config.clone())
            .unwrap();
        assert_eq!(coordinator.get_pod_container_count(&pod_name), Some(3));

        // 验证Pod使用情况
        let pod_usage = coordinator.get_pod_usage(&pod_name).unwrap();
        assert!(!pod_usage.active_containers.contains_key("container-2"));
        assert!(pod_usage.active_containers.contains_key("container-1"));
        assert!(pod_usage.active_containers.contains_key("container-3"));
        assert!(pod_usage.active_containers.contains_key("container-4"));

        // 清理所有容器
        coordinator
            .unregister_device(&pod_name, "container-1", 1001)
            .unwrap();
        coordinator
            .unregister_device(&pod_name, "container-3", 1003)
            .unwrap();
        coordinator
            .unregister_device(&pod_name, "container-4", 1004)
            .unwrap();

        assert!(!coordinator.pod_exists(&pod_name));
    }

    #[tokio::test]
    async fn test_device_configuration_variations() {
        let coordinator = create_test_coordinator();

        // 测试不同的设备配置
        let configs = vec![
            DeviceConfig {
                device_idx: 0,
                up_limit: 50,
                mem_limit: 4 * 1024 * 1024 * 1024,
                total_cuda_cores: 1024,
            },
            DeviceConfig {
                device_idx: 1,
                up_limit: 90,
                mem_limit: 16 * 1024 * 1024 * 1024,
                total_cuda_cores: 4096,
            },
            DeviceConfig {
                device_idx: 2,
                up_limit: 100,
                mem_limit: 32 * 1024 * 1024 * 1024,
                total_cuda_cores: 8192,
            },
        ];

        for (i, config) in configs.iter().enumerate() {
            let pod_name = create_unique_test_pod_name(&format!("config-test-{}", i));
            coordinator
                .register_device(
                    &pod_name,
                    "container-1",
                    1001 + i as u32,
                    12345 + i as u32,
                    config.clone(),
                )
                .unwrap();

            // 验证设备配置
            let pod_usage = coordinator.get_pod_usage(&pod_name).unwrap();
            assert_eq!(pod_usage.device_config.device_idx, config.device_idx);
            assert_eq!(pod_usage.device_config.up_limit, config.up_limit);
            assert_eq!(pod_usage.device_config.mem_limit, config.mem_limit);
            assert_eq!(
                pod_usage.device_config.total_cuda_cores,
                config.total_cuda_cores
            );

            // 清理
            coordinator
                .unregister_device(&pod_name, "container-1", 1001 + i as u32)
                .unwrap();
        }
    }

    #[tokio::test]
    async fn test_stress_large_number_of_pods() {
        let coordinator = create_test_coordinator();
        let config = create_test_device_config(0);
        const NUM_PODS: usize = 50;
        const CONTAINERS_PER_POD: usize = 3;

        // 创建大量Pod和容器
        let mut pod_names = Vec::new();
        for pod_idx in 0..NUM_PODS {
            let pod_name = create_unique_test_pod_name(&format!("stress-{}", pod_idx));
            pod_names.push(pod_name.clone());

            for container_idx in 0..CONTAINERS_PER_POD {
                let container_name = format!("container-{}", container_idx);
                let container_pid = (pod_idx * CONTAINERS_PER_POD + container_idx) as u32 + 1000;
                let host_pid = (pod_idx * CONTAINERS_PER_POD + container_idx) as u32 + 12000;

                coordinator
                    .register_device(
                        &pod_name,
                        &container_name,
                        container_pid,
                        host_pid,
                        config.clone(),
                    )
                    .unwrap();
            }
        }

        // 验证所有Pod都已注册
        for pod_name in &pod_names {
            assert!(coordinator.pod_exists(pod_name));
            assert_eq!(
                coordinator.get_pod_container_count(pod_name),
                Some(CONTAINERS_PER_POD)
            );
        }

        // 验证总体状态
        let all_usage = coordinator.get_all_pod_usage();
        assert_eq!(all_usage.len(), NUM_PODS);

        // 清理所有Pod
        for (pod_idx, pod_name) in pod_names.iter().enumerate() {
            for container_idx in 0..CONTAINERS_PER_POD {
                let container_name = format!("container-{}", container_idx);
                let container_pid = (pod_idx * CONTAINERS_PER_POD + container_idx) as u32 + 1000;

                coordinator
                    .unregister_device(pod_name, &container_name, container_pid)
                    .unwrap();
            }
        }

        // 验证所有Pod都已清理
        for pod_name in &pod_names {
            assert!(!coordinator.pod_exists(pod_name));
        }
    }

    #[tokio::test]
    async fn test_mixed_operations_complex_scenario() {
        let coordinator = create_test_coordinator();
        let config = create_test_device_config(0);
        let mut operation_count = 0;

        // 复杂的混合操作场景 - 顺序执行
        for i in 0..20 {
            let pod_name = create_unique_test_pod_name(&format!("mixed-{}", i));
            let container_name = format!("container-{}", i);
            let container_pid = 1000 + i as u32;
            let host_pid = 12000 + i as u32;

            // 注册设备
            coordinator
                .register_device(
                    &pod_name,
                    &container_name,
                    container_pid,
                    host_pid,
                    config.clone(),
                )
                .unwrap();

            operation_count += 1;

            // 执行一些操作
            sleep(Duration::from_millis(10)).await;

            // 尝试更新配置
            if i % 3 == 0 {
                let _ = coordinator.update_up_limit(&pod_name, 85);
            }

            // 获取状态
            let _ = coordinator.get_pod_usage(&pod_name);
            let _ = coordinator.get_pod_container_count(&pod_name);

            sleep(Duration::from_millis(10)).await;

            // 注销设备
            coordinator
                .unregister_device(&pod_name, &container_name, container_pid)
                .unwrap();
        }

        // 验证所有操作都已完成
        assert_eq!(operation_count, 20);
    }

    #[tokio::test]
    async fn test_limiter_coordinator_integration_with_thread_safe_manager() {
        let coordinator = create_test_coordinator();
        let config = create_test_device_config(0);
        let pod_name = create_unique_test_pod_name("integration-test");

        // Register device
        coordinator
            .register_device(&pod_name, "container-1", 1001, 12345, config.clone())
            .unwrap();

        // Verify Pod exists
        assert!(coordinator.pod_exists(&pod_name));
        assert_eq!(coordinator.get_pod_container_count(&pod_name), Some(1));

        // Verify shared memory is accessible
        let shared_state = coordinator
            .get_shared_memory_for_container(&pod_name, "container-1")
            .unwrap();
        assert!(!shared_state.is_null());

        // Verify device configuration
        let pod_usage = coordinator.get_pod_usage(&pod_name).unwrap();
        assert_eq!(pod_usage.device_config.device_idx, 0);
        assert_eq!(pod_usage.device_config.up_limit, 80);
        assert_eq!(pod_usage.active_containers.len(), 1);

        // Unregister device
        coordinator
            .unregister_device(&pod_name, "container-1", 1001)
            .unwrap();

        // Verify Pod is cleaned up
        assert!(!coordinator.pod_exists(&pod_name));
        assert_eq!(coordinator.get_pod_container_count(&pod_name), None);
    }

    #[tokio::test]
    async fn test_limiter_coordinator_thread_safety() {
        use std::sync::Arc;

        use tokio::task;

        let coordinator = Arc::new(create_test_coordinator());
        let config = create_test_device_config(0);

        // Spawn multiple concurrent tasks
        let mut handles = vec![];

        for i in 0..10 {
            let coordinator_clone = coordinator.clone();
            let config_clone = config.clone();
            let pod_name = create_unique_test_pod_name(&format!("thread-safety-{}", i));

            let handle = task::spawn(async move {
                // Register device
                coordinator_clone
                    .register_device(&pod_name, "container-1", 1001 + i, 12345 + i, config_clone)
                    .unwrap();

                // Verify registration
                assert!(coordinator_clone.pod_exists(&pod_name));

                // Get shared memory
                let shared_state = coordinator_clone
                    .get_shared_memory_for_container(&pod_name, "container-1")
                    .unwrap();
                assert!(!shared_state.is_null());

                // Unregister device
                coordinator_clone
                    .unregister_device(&pod_name, "container-1", 1001 + i)
                    .unwrap();

                // Verify cleanup
                assert!(!coordinator_clone.pod_exists(&pod_name));

                format!("Task {} completed", i)
            });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.contains("completed"));
        }

        // Verify all pods are cleaned up
        let all_pods = coordinator.get_all_pod_usage();
        assert_eq!(all_pods.len(), 0);
    }

    #[tokio::test]
    async fn test_limiter_coordinator_multiple_containers_per_pod() {
        let coordinator = create_test_coordinator();
        let config = create_test_device_config(0);
        let pod_name = create_unique_test_pod_name("multi-container");

        // Register multiple containers for the same pod
        coordinator
            .register_device(&pod_name, "container-1", 1001, 12345, config.clone())
            .unwrap();
        coordinator
            .register_device(&pod_name, "container-2", 1002, 12346, config.clone())
            .unwrap();
        coordinator
            .register_device(&pod_name, "container-3", 1003, 12347, config.clone())
            .unwrap();

        // Verify pod exists with correct container count
        assert!(coordinator.pod_exists(&pod_name));
        assert_eq!(coordinator.get_pod_container_count(&pod_name), Some(3));

        // Verify each container can access shared memory
        for i in 1..=3 {
            let container_name = format!("container-{}", i);
            let shared_state = coordinator
                .get_shared_memory_for_container(&pod_name, &container_name)
                .unwrap();
            assert!(!shared_state.is_null());
        }

        // Verify host PIDs are tracked correctly
        let pod_usage = coordinator.get_pod_usage(&pod_name).unwrap();
        let host_pids = pod_usage.get_host_pids();
        assert_eq!(host_pids.len(), 3);
        assert!(host_pids.contains(&12345));
        assert!(host_pids.contains(&12346));
        assert!(host_pids.contains(&12347));

        // Unregister containers one by one
        coordinator
            .unregister_device(&pod_name, "container-1", 1001)
            .unwrap();
        assert_eq!(coordinator.get_pod_container_count(&pod_name), Some(2));

        coordinator
            .unregister_device(&pod_name, "container-2", 1002)
            .unwrap();
        assert_eq!(coordinator.get_pod_container_count(&pod_name), Some(1));

        coordinator
            .unregister_device(&pod_name, "container-3", 1003)
            .unwrap();

        // Pod should be cleaned up after last container is removed
        assert!(!coordinator.pod_exists(&pod_name));
    }

    #[tokio::test]
    async fn test_limiter_coordinator_error_handling() {
        let coordinator = create_test_coordinator();

        // Test accessing non-existent pod
        assert!(!coordinator.pod_exists("non-existent-pod"));
        assert_eq!(
            coordinator.get_pod_container_count("non-existent-pod"),
            None
        );
        assert!(coordinator.get_pod_usage("non-existent-pod").is_none());

        // Test accessing non-existent container
        let config = create_test_device_config(0);
        let pod_name = create_unique_test_pod_name("test-pod");
        coordinator
            .register_device(&pod_name, "container-1", 1001, 12345, config)
            .unwrap();

        let result =
            coordinator.get_shared_memory_for_container(&pod_name, "non-existent-container");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Container 'non-existent-container' not found"));

        // Test double registration (should not fail)
        let config2 = create_test_device_config(0);
        let result = coordinator.register_device(&pod_name, "container-1", 1001, 12345, config2);
        assert!(result.is_ok());

        // Test unregistering non-existent container (should not fail)
        let result = coordinator.unregister_device(&pod_name, "non-existent-container", 9999);
        assert!(result.is_ok());

        // Clean up
        coordinator
            .unregister_device(&pod_name, "container-1", 1001)
            .unwrap();
    }

    #[tokio::test]
    async fn test_limiter_coordinator_device_config_variations() {
        let coordinator = create_test_coordinator();

        // Test different device configurations
        let configs = vec![
            DeviceConfig {
                device_idx: 0,
                up_limit: 50,
                mem_limit: 4 * 1024 * 1024 * 1024,
                total_cuda_cores: 1024,
            },
            DeviceConfig {
                device_idx: 1,
                up_limit: 90,
                mem_limit: 16 * 1024 * 1024 * 1024,
                total_cuda_cores: 4096,
            },
            DeviceConfig {
                device_idx: 2,
                up_limit: 100,
                mem_limit: 32 * 1024 * 1024 * 1024,
                total_cuda_cores: 8192,
            },
        ];

        for (i, config) in configs.iter().enumerate() {
            let pod_name = create_unique_test_pod_name(&format!("config-test-{}", i));
            coordinator
                .register_device(
                    &pod_name,
                    "container-1",
                    1001 + i as u32,
                    12345 + i as u32,
                    config.clone(),
                )
                .unwrap();

            // Verify device configuration
            let pod_usage = coordinator.get_pod_usage(&pod_name).unwrap();
            assert_eq!(pod_usage.device_config.device_idx, config.device_idx);
            assert_eq!(pod_usage.device_config.up_limit, config.up_limit);
            assert_eq!(pod_usage.device_config.mem_limit, config.mem_limit);
            assert_eq!(
                pod_usage.device_config.total_cuda_cores,
                config.total_cuda_cores
            );

            // Verify shared memory can be accessed
            let shared_state = coordinator
                .get_shared_memory_for_container(&pod_name, "container-1")
                .unwrap();
            assert!(!shared_state.is_null());

            // Clean up
            coordinator
                .unregister_device(&pod_name, "container-1", 1001 + i as u32)
                .unwrap();
        }
    }
}
