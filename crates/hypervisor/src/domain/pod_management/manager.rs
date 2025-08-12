//! Simplified pod manager that handles worker lifecycle with unified state management.

use std::sync::Arc;
use std::time::Duration;

use api_types::{QosLevel, WorkerInfo};
use nvml_wrapper::Nvml;
use tracing::{info, warn};
use utils::shared_memory::handle::SharedMemoryHandle;

use crate::domain::HypervisorType;
use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::{HostPidProbe, PodProcessInfo, SubscriptionRequest};
use crate::infrastructure::k8s::pod_info_cache::PodInfoCache;
use crate::limiter_comm::CommandDispatcher;
use crate::process::worker::TensorFusionWorker;
use crate::process::GpuProcess;
use tokio_util::sync::CancellationToken;

use super::coordinator::LimiterCoordinator;
use super::device_info::create_device_configs_from_worker_info;
use super::pod_state_store::PodStateStore;
use super::types::{PodManagementError, Result};

/// Simplified pod manager with unified state management
pub struct PodManager {
    /// Centralized pod state store
    pod_state_store: Arc<PodStateStore>,
    host_pid_probe: Arc<HostPidProbe>,
    command_dispatcher: Arc<CommandDispatcher>,
    hypervisor: Arc<HypervisorType>,
    limiter_coordinator: Arc<LimiterCoordinator>,
    nvml: Arc<Nvml>,
    pod_info_cache: Arc<PodInfoCache>,
    gpu_observer: Arc<GpuObserver>,
}

impl PodManager {
    /// Create a new pod manager with unified state management.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        host_pid_probe: Arc<HostPidProbe>,
        command_dispatcher: Arc<CommandDispatcher>,
        hypervisor: Arc<HypervisorType>,
        limiter_coordinator: Arc<LimiterCoordinator>,
        nvml: Arc<Nvml>,
        pod_info_cache: Arc<PodInfoCache>,
        pod_state_store: Arc<PodStateStore>,
        gpu_observer: Arc<GpuObserver>,
    ) -> Self {
        Self {
            pod_state_store,
            host_pid_probe,
            command_dispatcher,
            hypervisor,
            limiter_coordinator,
            nvml,
            pod_info_cache,
            gpu_observer,
        }
    }

    pub async fn restore_pod_from_shared_memory(&self, shm_glob_pattern: &str) -> Result<()> {
        let shared_memory_files = self
            .limiter_coordinator
            .find_shared_memory_files(shm_glob_pattern)
            .map_err(|e| PodManagementError::SharedMemoryError {
                message: e.to_string(),
            })?;

        for file in shared_memory_files {
            let identifier = self
                .limiter_coordinator
                .extract_identifier_from_path(&file)
                .map_err(|e| PodManagementError::SharedMemoryError {
                    message: e.to_string(),
                })?;
            let (namespace, pod_name) = self.pod_name_namespace(&identifier);
            if let Err(e) = self.ensure_pod_registered(&namespace, &pod_name).await {
                tracing::error!(
                    "Failed to restore pod from shared memory: {}: {}",
                    identifier,
                    e
                );
            }
        }

        Ok(())
    }

    /// Find a pod by namespace and pod name.
    pub async fn find_pod_by_name(
        &self,
        namespace: &str,
        pod_name: &str,
    ) -> Result<Option<WorkerInfo>> {
        let pod_identifier = self.pod_identifier(namespace, pod_name);

        // Check local state first
        if let Some(pod_state) = self.pod_state_store.get_pod(&pod_identifier) {
            return Ok(Some(pod_state.info));
        }

        // Try to get from pod info cache and register if found
        if let Some(pod_info) = self
            .pod_info_cache
            .get_pod_info(namespace, pod_name)
            .await
            .map_err(|e| PodManagementError::KubernetesError {
                message: e.to_string(),
            })?
        {
            return Ok(Some(pod_info.0));
        }

        Ok(None)
    }

    /// Find a pod by worker PID.
    pub fn find_pod_by_worker_pid(&self, pid: u32) -> Option<String> {
        self.pod_state_store.get_pod_by_pid(pid)
    }

    /// Get the pod state store for API queries.
    pub fn pod_state_store(&self) -> &Arc<PodStateStore> {
        &self.pod_state_store
    }

    /// Initialize a CUDA process: discover PID and register to all components
    pub async fn initialize_process(
        &self,
        pod_name: &str,
        namespace: &str,
        container_name: &str,
        container_pid: u32,
    ) -> Result<u32> {
        // Ensure pod is registered first
        self.ensure_pod_registered(namespace, pod_name).await?;

        // Discover PID
        let process_info = self
            .discover_process_info(pod_name, namespace, container_name, container_pid)
            .await?;

        // Register process to all components
        self.register_process_to_all_components(
            pod_name,
            namespace,
            container_name,
            process_info.host_pid,
        )
        .await?;

        Ok(process_info.host_pid)
    }

    /// Start the resource monitoring task
    pub async fn start_resource_monitor(
        &self,
        interval: Duration,
        cancellation_token: CancellationToken,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        info!("Starting resource monitor with interval: {:?}", interval);

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    info!("Resource monitor shutdown requested");
                    break;
                }
                _ = interval_timer.tick() => {
                    // Continue with monitoring logic
                }
            }

            // Check for dead processes and clean them up
            if let Err(e) = self.check_and_cleanup_dead_processes().await {
                tracing::error!("Failed to check and cleanup dead processes: {}", e);
            }
        }

        info!("Resource monitor stopped");
    }

    // Private helper methods
    fn pod_identifier(&self, namespace: &str, pod_name: &str) -> String {
        format!("tf_shm_{namespace}_{pod_name}")
    }

    fn pod_name_namespace(&self, pod_identifier: &str) -> (String, String) {
        let parts: Vec<&str> = pod_identifier.split('_').collect();
        (parts[1].to_string(), parts[2].to_string())
    }

    /// Ensure pod is registered in all components (lazy loading)
    pub async fn ensure_pod_registered(&self, namespace: &str, pod_name: &str) -> Result<()> {
        let pod_identifier = self.pod_identifier(namespace, pod_name);
        // Check if already registered
        if self.pod_state_store.contains_pod(&pod_identifier) {
            return Ok(());
        }

        // Get pod info from cache
        let pod_info = self
            .pod_info_cache
            .get_pod_info(namespace, pod_name)
            .await
            .map_err(|e| PodManagementError::KubernetesError {
                message: e.to_string(),
            })?
            .ok_or_else(|| PodManagementError::PodNotFound {
                namespace: namespace.to_string(),
                pod_name: pod_name.to_string(),
            })?;

        // Create device configs if GPU resources are specified
        let device_configs = if let Some(gpu_uuids) = &pod_info.0.gpu_uuids {
            if !gpu_uuids.is_empty() {
                let configs = create_device_configs_from_worker_info(&pod_info.0, &self.nvml)
                    .await
                    .map_err(|e| PodManagementError::DeviceError {
                        message: e.to_string(),
                    })?;
                configs
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Register in state store
        self.pod_state_store
            .register_pod(&pod_identifier, pod_info.0, device_configs.clone())?;

        let restored_pids = self
            .limiter_coordinator
            .ensure_pod_registered(&pod_identifier, &device_configs)
            .await
            .map_err(|e| PodManagementError::RegistrationFailed {
                message: e.to_string(),
            })?;

        if !restored_pids.is_empty() {
            for pid in restored_pids {
                self.pod_state_store
                    .register_process(&pod_identifier, pid as u32)?;
                self.limiter_coordinator
                    .register_process(&pod_identifier, pid as u32)
                    .map_err(|e| PodManagementError::RegistrationFailed {
                        message: e.to_string(),
                    })?;
            }
        }

        Ok(())
    }

    /// Discover process info (separated from registration logic)
    async fn discover_process_info(
        &self,
        pod_name: &str,
        namespace: &str,
        container_name: &str,
        container_pid: u32,
    ) -> Result<PodProcessInfo> {
        let subscription_request = SubscriptionRequest {
            pod_name: pod_name.to_string(),
            namespace: namespace.to_string(),
            container_name: container_name.to_string(),
            container_pid,
        };

        info!(
            "Starting PID discovery for container {} in pod {}/{}",
            container_name, namespace, pod_name
        );

        let receiver = self
            .host_pid_probe
            .subscribe(subscription_request, Duration::from_secs(5))
            .await;

        let process_info = receiver.await.map_err(|_| PodManagementError::StateError {
            message: "PID discovery subscription was cancelled".to_string(),
        })?;

        info!(
            "Discovered worker PID: host_pid={}, container_pid={} for container {} in pod {}/{}",
            process_info.host_pid, process_info.container_pid, container_name, namespace, pod_name
        );

        Ok(process_info)
    }

    /// Register process to all components
    async fn register_process_to_all_components(
        &self,
        pod_name: &str,
        namespace: &str,
        container_name: &str,
        host_pid: u32,
    ) -> Result<()> {
        let pod_identifier = self.pod_identifier(namespace, pod_name);

        // Get pod state to extract worker info
        let pod_state = self
            .pod_state_store
            .get_pod(&pod_identifier)
            .ok_or_else(|| PodManagementError::PodIdentifierNotFound {
                pod_identifier: pod_identifier.clone(),
            })?;

        let WorkerInfo {
            namespace: info_namespace,
            pod_name: info_pod_name,
            gpu_uuids,
            qos_level,
            ..
        } = &pod_state.info;

        let gpu_uuids_vec = gpu_uuids.clone().unwrap_or_default();
        let qos = qos_level.unwrap_or(QosLevel::Medium);

        let worker = Arc::new(TensorFusionWorker::new(
            host_pid,
            qos,
            gpu_uuids_vec,
            self.gpu_observer.clone(),
            info_namespace.clone(),
            info_pod_name.clone(),
            self.command_dispatcher.clone(),
        ));

        // 1. Register process in state store
        self.pod_state_store
            .register_process(&pod_identifier, host_pid)?;

        // 2. Register with limiter coordinator
        // Register process with the limiter coordinator.
        self.limiter_coordinator
            .register_process(&pod_identifier, host_pid)
            .map_err(|e| PodManagementError::RegistrationFailed {
                message: e.to_string(),
            })?;

        // 3. Set up shared memory handle
        let shared_memory_handle =
            Arc::new(SharedMemoryHandle::open(&pod_identifier).map_err(|e| {
                PodManagementError::SharedMemoryError {
                    message: format!("Failed to open shared memory for {pod_identifier}: {e}"),
                }
            })?);

        self.pod_state_store
            .set_shared_memory_handle(&pod_identifier, shared_memory_handle)?;

        // 4. Add worker to hypervisor
        if !self.hypervisor.process_exists(host_pid).await {
            info!("Adding new worker to hypervisor: {}", worker.name());
            self.hypervisor.add_process(worker.clone()).await;
        }

        info!(
            "Successfully registered process {host_pid} for pod {pod_identifier} container {container_name}"
        );

        Ok(())
    }

    /// check_and_cleanup_dead_processes for use in monitoring task
    async fn check_and_cleanup_dead_processes(&self) -> Result<Vec<u32>> {
        let mut dead_pids = Vec::new();

        // Get all tracked PIDs from the state store
        let tracked_pids: Vec<u32> = {
            let stats = self.pod_state_store.stats();
            let mut pids = Vec::with_capacity(stats.total_processes);

            for pod_id in self.pod_state_store.list_pod_identifiers() {
                let processes = self.pod_state_store.get_pod_processes(&pod_id);
                for process in processes {
                    pids.push(process);
                }
            }
            pids
        };

        // Check each PID for liveness
        for pid in tracked_pids {
            let is_alive = unsafe { libc::kill(pid as i32, 0) == 0 };
            if !is_alive {
                info!("Detected dead process: {}", pid);
                dead_pids.push(pid);

                // Clean up the dead process
                if let Err(e) = self.handle_process_exited(pid).await {
                    tracing::error!("Failed to cleanup dead process {}: {}", pid, e);
                }
            }
        }

        if !dead_pids.is_empty() {
            info!(
                "Cleaned up {} dead processes: {:?}",
                dead_pids.len(),
                dead_pids
            );
        }

        Ok(dead_pids)
    }

    /// Handle process exit cleanup
    async fn handle_process_exited(&self, host_pid: u32) -> Result<()> {
        info!("Processing process exit: host_pid={}", host_pid);

        // Get pod identifier for this PID
        let pod_identifier = match self.pod_state_store.get_pod_by_pid(host_pid) {
            Some(pod_id) => pod_id,
            None => {
                warn!("Attempted to cleanup non-tracked process: {}", host_pid);
                return Ok(());
            }
        };

        info!(
            "Found tracked process: host_pid={}, pod={}",
            host_pid, pod_identifier
        );

        // 1. Remove from hypervisor
        if self.hypervisor.process_exists(host_pid).await {
            info!("Removing process {} from hypervisor", host_pid);
            self.hypervisor.remove_process(host_pid).await;
        }

        // 2. Unregister from limiter coordinator
        if let Err(e) = self
            .limiter_coordinator
            .unregister_process(&pod_identifier, host_pid)
            .await
        {
            tracing::error!(
                "Failed to unregister process {} from limiter coordinator: {}",
                host_pid,
                e
            );
        } else {
            info!("Unregistered process {} from limiter coordinator", host_pid);
        }

        // 3. Remove from pod state store (atomic cleanup)
        let pod_removed = self
            .pod_state_store
            .unregister_process(&pod_identifier, host_pid)?;

        if pod_removed {
            info!("Pod {} removed (no more processes)", pod_identifier);
        }

        info!(
            "Successfully cleaned up process {} from pod {}",
            host_pid, pod_identifier
        );

        Ok(())
    }
}
