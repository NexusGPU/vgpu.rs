//! Simplified pod manager that handles worker lifecycle with unified state management.

use std::sync::Arc;
use std::time::Duration;

use api_types::{QosLevel, WorkerInfo};
use nvml_wrapper::Nvml;
use tracing::{info, warn};
use utils::shared_memory::handle::SharedMemoryHandle;
use utils::shared_memory::traits::SharedMemoryAccess;

use crate::domain::hypervisor::HypervisorType;
use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::{HostPidProbe, PodProcessInfo, SubscriptionRequest};
use crate::infrastructure::k8s::pod_info_cache::PodInfoCache;
use crate::limiter_comm::CommandDispatcher;
use crate::pod_management::coordinator::LimiterCoordinator;
use crate::pod_management::traits::{DeviceSnapshotProvider, PodStateRepository, TimeSource};
use crate::process::worker::TensorFusionWorker;
use crate::process::GpuProcess;
use tokio_util::sync::CancellationToken;

use super::device_info::create_device_configs_from_worker_info;
use super::pod_state_store::PodStateStore;
use super::types::{PodManagementError, Result};

const IDENTIFIER_PREFIX: &str = "tf_shm_";
/// Simplified pod manager with unified state management
pub struct PodManager<M, P, D, T> {
    /// Centralized pod state store
    pod_state_store: Arc<PodStateStore>,
    host_pid_probe: Arc<HostPidProbe>,
    command_dispatcher: Arc<CommandDispatcher>,
    hypervisor: Arc<HypervisorType>,
    limiter_coordinator: Arc<LimiterCoordinator<M, P, D, T>>,
    nvml: Arc<Nvml>,
    pod_info_cache: Arc<PodInfoCache>,
    gpu_observer: Arc<GpuObserver>,
}

impl<M, P, D, T> PodManager<M, P, D, T> {
    /// Get the pod state store for API queries.
    pub fn pod_state_store(&self) -> &Arc<PodStateStore> {
        &self.pod_state_store
    }
    /// Find a pod by worker PID.
    pub fn find_pod_by_worker_pid(&self, pid: u32) -> Option<String> {
        self.pod_state_store.get_pod_by_pid(pid)
    }
}

impl<M, P, D, T> PodManager<M, P, D, T>
where
    M: SharedMemoryAccess + 'static,
    P: PodStateRepository + 'static,
    D: DeviceSnapshotProvider + 'static,
    T: TimeSource + 'static,
{
    /// Create a new pod manager with unified state management.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        host_pid_probe: Arc<HostPidProbe>,
        command_dispatcher: Arc<CommandDispatcher>,
        hypervisor: Arc<HypervisorType>,
        limiter_coordinator: Arc<LimiterCoordinator<M, P, D, T>>,
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
            let Some((namespace, pod_name)) = self.pod_name_namespace(&identifier) else {
                tracing::warn!(
                    "Skipping shared memory file: {identifier} because it does not match the expected format"
                );
                continue;
            };
            if let Err(e) = self.ensure_pod_registered(namespace, pod_name).await {
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
        format!("{IDENTIFIER_PREFIX}{namespace}_{pod_name}")
    }

    fn pod_name_namespace<'s>(&self, pod_identifier: &'s str) -> Option<(&'s str, &'s str)> {
        let rest = pod_identifier.strip_prefix(IDENTIFIER_PREFIX)?;
        let mut parts = rest.splitn(2, '_');
        let namespace = parts.next()?;
        let pod_name = parts.next().unwrap_or("");
        Some((namespace, pod_name))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::pod_management::pod_state_store::PodStateStore;
    use std::collections::BTreeMap;
    use utils::shared_memory::DeviceConfig;

    // Note: Mock types would be defined here if needed for full PodManager testing
    // For now, we focus on testing the business logic components that can be isolated

    fn create_test_worker_info() -> WorkerInfo {
        WorkerInfo {
            namespace: "test-namespace".to_string(),
            pod_name: "test-pod".to_string(),
            containers: Some(vec!["test-container".to_string()]),
            gpu_uuids: Some(vec!["GPU-test-123".to_string()]),
            qos_level: Some(QosLevel::Medium),
            tflops_request: Some(5.0),
            tflops_limit: Some(10.0),
            vram_request: Some(1024),
            vram_limit: Some(2048),
            node_name: Some("test-node".to_string()),
            host_pid: 12345,
            labels: BTreeMap::new(),
            workload_name: Some("test-workload".to_string()),
        }
    }

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

    #[test]
    fn test_complex_identifier_edge_cases() {
        // Test complex identifier parsing scenarios
        let edge_cases = vec![
            // Normal cases
            ("tf_shm_default_my-pod", Some(("default", "my-pod"))),
            (
                "tf_shm_kube-system_coredns-123",
                Some(("kube-system", "coredns-123")),
            ),
            (
                "tf_shm_ns_very_long_pod_name_with_many_underscores_and_numbers_123",
                Some((
                    "ns",
                    "very_long_pod_name_with_many_underscores_and_numbers_123",
                )),
            ),
            // Invalid cases
            ("tf_shm_no_underscore", Some(("no", "underscore"))),
            ("wrong_prefix_ns_pod", None),
            ("tf_shm_", None),
            ("tf_shm_only-namespace_", Some(("only-namespace", ""))),
            ("", None),
        ];

        for (identifier, expected) in edge_cases {
            let result = identifier
                .strip_prefix(IDENTIFIER_PREFIX)
                .and_then(|s| s.split_once('_'));

            assert_eq!(result, expected, "Failed for identifier: {identifier}");
        }
    }

    #[test]
    fn test_identifier_generation_with_special_characters() {
        // Test identifier generation with various special characters
        let special_cases = vec![
            ("default", "pod-with-dashes"),
            ("namespace-with-dashes", "pod.with.dots"),
            ("ns123", "pod123"),
            ("UPPERCASE", "mixedCASE"),
            ("", "empty-namespace"), // Edge case
            ("namespace", ""),       // Edge case
        ];

        for (namespace, pod_name) in special_cases {
            let identifier = format!("{IDENTIFIER_PREFIX}{namespace}_{pod_name}");

            // Test parsing back
            let parsed = identifier
                .strip_prefix(IDENTIFIER_PREFIX)
                .and_then(|s| s.split_once('_'));

            if namespace.is_empty() {
                // Special handling for empty namespace
                assert_eq!(parsed, Some(("", pod_name)));
            } else if pod_name.is_empty() {
                // Special handling for empty pod name
                assert_eq!(parsed, Some((namespace, "")));
            } else {
                assert_eq!(parsed, Some((namespace, pod_name)));
            }
        }
    }

    #[tokio::test]
    async fn test_pod_state_store_operations() {
        let pod_state_store = Arc::new(PodStateStore::new());
        let worker_info = create_test_worker_info();
        let device_configs = vec![create_test_device_config()];
        let pod_identifier = "tf_shm_test-namespace_test-pod";

        // Test pod registration
        let result =
            pod_state_store.register_pod(pod_identifier, worker_info.clone(), device_configs);
        assert!(result.is_ok());

        // Verify pod is registered
        assert!(pod_state_store.contains_pod(pod_identifier));

        // Test process registration
        let result = pod_state_store.register_process(pod_identifier, 12345);
        assert!(result.is_ok());

        // Verify process is tracked
        assert_eq!(
            pod_state_store.get_pod_by_pid(12345),
            Some(pod_identifier.to_string())
        );

        // Test process unregistration
        let pod_removed = pod_state_store
            .unregister_process(pod_identifier, 12345)
            .unwrap();
        assert!(pod_removed); // Pod should be removed as it has no more processes

        // Verify cleanup
        assert!(!pod_state_store.contains_pod(pod_identifier));
        assert_eq!(pod_state_store.get_pod_by_pid(12345), None);
    }

    #[tokio::test]
    async fn test_multiple_processes_lifecycle() {
        let pod_state_store = Arc::new(PodStateStore::new());
        let worker_info = create_test_worker_info();
        let device_configs = vec![create_test_device_config()];
        let pod_identifier = "tf_shm_test-namespace_test-pod";

        // Register pod
        pod_state_store
            .register_pod(pod_identifier, worker_info, device_configs)
            .unwrap();

        // Register multiple processes
        pod_state_store
            .register_process(pod_identifier, 12345)
            .unwrap();
        pod_state_store
            .register_process(pod_identifier, 12346)
            .unwrap();
        pod_state_store
            .register_process(pod_identifier, 12347)
            .unwrap();

        // Verify all processes are tracked
        assert_eq!(
            pod_state_store.get_pod_by_pid(12345),
            Some(pod_identifier.to_string())
        );
        assert_eq!(
            pod_state_store.get_pod_by_pid(12346),
            Some(pod_identifier.to_string())
        );
        assert_eq!(
            pod_state_store.get_pod_by_pid(12347),
            Some(pod_identifier.to_string())
        );

        // Unregister first process - pod should remain
        let pod_removed = pod_state_store
            .unregister_process(pod_identifier, 12345)
            .unwrap();
        assert!(!pod_removed);
        assert!(pod_state_store.contains_pod(pod_identifier));

        // Unregister second process - pod should remain
        let pod_removed = pod_state_store
            .unregister_process(pod_identifier, 12346)
            .unwrap();
        assert!(!pod_removed);
        assert!(pod_state_store.contains_pod(pod_identifier));

        // Unregister last process - pod should be removed
        let pod_removed = pod_state_store
            .unregister_process(pod_identifier, 12347)
            .unwrap();
        assert!(pod_removed);
        assert!(!pod_state_store.contains_pod(pod_identifier));
    }

    #[test]
    fn test_error_scenarios() {
        let pod_state_store = Arc::new(PodStateStore::new());
        let pod_identifier = "tf_shm_test-namespace_test-pod";

        // Test registering process for non-existent pod
        let result = pod_state_store.register_process(pod_identifier, 12345);
        assert!(result.is_err());

        // Test unregistering process for non-existent pod
        let result = pod_state_store.unregister_process(pod_identifier, 12345);
        assert!(result.is_err());

        // Test getting pod for non-existent identifier
        let result = pod_state_store.get_pod(pod_identifier);
        assert!(result.is_none());

        // Test getting pod by non-existent PID
        let result = pod_state_store.get_pod_by_pid(99999);
        assert!(result.is_none());
    }

    #[test]
    fn test_stats_calculation() {
        let pod_state_store = Arc::new(PodStateStore::new());
        let worker_info = create_test_worker_info();
        let device_configs = vec![create_test_device_config()];

        // Initially empty
        let stats = pod_state_store.stats();
        assert_eq!(stats.total_pods, 0);
        assert_eq!(stats.total_processes, 0);

        // Register first pod with processes
        pod_state_store
            .register_pod("pod1", worker_info.clone(), device_configs.clone())
            .unwrap();
        pod_state_store.register_process("pod1", 1001).unwrap();
        pod_state_store.register_process("pod1", 1002).unwrap();

        let stats = pod_state_store.stats();
        assert_eq!(stats.total_pods, 1);
        assert_eq!(stats.total_processes, 2);

        // Register second pod with processes
        pod_state_store
            .register_pod("pod2", worker_info, device_configs)
            .unwrap();
        pod_state_store.register_process("pod2", 2001).unwrap();

        let stats = pod_state_store.stats();
        assert_eq!(stats.total_pods, 2);
        assert_eq!(stats.total_processes, 3);

        // Remove processes and pods
        pod_state_store.unregister_process("pod1", 1001).unwrap();
        let stats = pod_state_store.stats();
        assert_eq!(stats.total_pods, 2);
        assert_eq!(stats.total_processes, 2);

        // Remove last process from pod1 - should remove pod
        pod_state_store.unregister_process("pod1", 1002).unwrap();
        let stats = pod_state_store.stats();
        assert_eq!(stats.total_pods, 1);
        assert_eq!(stats.total_processes, 1);
    }

    #[test]
    fn test_device_queries() {
        let pod_state_store = Arc::new(PodStateStore::new());
        let worker_info = create_test_worker_info();

        // Create configs for different devices
        let device_configs = [
            DeviceConfig {
                device_idx: 0,
                device_uuid: "GPU-0".to_string(),
                up_limit: 80,
                mem_limit: 2048,
                total_cuda_cores: 2560,
                sm_count: 20,
                max_thread_per_sm: 128,
            },
            DeviceConfig {
                device_idx: 1,
                device_uuid: "GPU-1".to_string(),
                up_limit: 90,
                mem_limit: 4096,
                total_cuda_cores: 5120,
                sm_count: 40,
                max_thread_per_sm: 128,
            },
        ];

        // Register pods using different devices
        pod_state_store
            .register_pod("pod1", worker_info.clone(), vec![device_configs[0].clone()])
            .unwrap();
        pod_state_store
            .register_pod("pod2", worker_info, vec![device_configs[1].clone()])
            .unwrap();

        // Test device queries
        let pods_on_device_0 = pod_state_store.get_pods_using_device(0);
        assert_eq!(pods_on_device_0, vec!["pod1"]);

        let pods_on_device_1 = pod_state_store.get_pods_using_device(1);
        assert_eq!(pods_on_device_1, vec!["pod2"]);

        let pods_on_device_2 = pod_state_store.get_pods_using_device(2);
        assert!(pods_on_device_2.is_empty());

        // Test device config retrieval
        let config = pod_state_store.get_device_config_for_pod("pod1", 0);
        assert!(config.is_some());
        assert_eq!(config.unwrap().device_uuid, "GPU-0");

        let config = pod_state_store.get_device_config_for_pod("pod1", 1);
        assert!(config.is_none());

        let config = pod_state_store.get_device_config_for_pod("pod2", 1);
        assert!(config.is_some());
        assert_eq!(config.unwrap().device_uuid, "GPU-1");
    }

    #[tokio::test]
    async fn test_concurrent_pod_operations() {
        use std::sync::Arc;
        use tokio::task::JoinSet;

        let pod_state_store = Arc::new(PodStateStore::new());

        // Test concurrent registrations and unregistrations
        let mut join_set = JoinSet::new();

        // Start multiple concurrent tasks that register/unregister pods
        for i in 0..50 {
            let store = pod_state_store.clone();
            join_set.spawn(async move {
                let worker_info = create_test_worker_info();
                let device_configs = vec![create_test_device_config()];
                let pod_id = format!("pod-{i}");

                // Register pod
                store
                    .register_pod(&pod_id, worker_info, device_configs)
                    .unwrap();

                // Register processes
                let pid1 = 1000 + i * 2;
                let pid2 = 1000 + i * 2 + 1;
                store.register_process(&pod_id, pid1).unwrap();
                store.register_process(&pod_id, pid2).unwrap();

                // Verify registration
                assert!(store.contains_pod(&pod_id));
                assert_eq!(store.get_pod_by_pid(pid1), Some(pod_id.clone()));
                assert_eq!(store.get_pod_by_pid(pid2), Some(pod_id.clone()));

                // Unregister one process
                let pod_removed = store.unregister_process(&pod_id, pid1).unwrap();
                assert!(!pod_removed); // Pod should still exist

                // Unregister second process
                let pod_removed = store.unregister_process(&pod_id, pid2).unwrap();
                assert!(pod_removed); // Pod should be removed

                // Verify cleanup
                assert!(!store.contains_pod(&pod_id));
                assert_eq!(store.get_pod_by_pid(pid1), None);
                assert_eq!(store.get_pod_by_pid(pid2), None);
            });
        }

        // Wait for all tasks to complete
        while let Some(result) = join_set.join_next().await {
            result.unwrap(); // Ensure no task panicked
        }

        // Final verification - store should be empty
        let stats = pod_state_store.stats();
        assert_eq!(stats.total_pods, 0);
        assert_eq!(stats.total_processes, 0);
    }

    #[tokio::test]
    async fn test_stress_test_with_many_pods_and_processes() {
        let pod_state_store = Arc::new(PodStateStore::new());

        // Create many pods with many processes each
        let num_pods = 100;
        let processes_per_pod = 20;

        // Register all pods and processes
        for pod_idx in 0..num_pods {
            let worker_info = create_test_worker_info();
            let device_configs = vec![create_test_device_config()];
            let pod_id = format!("stress-pod-{pod_idx:03}");

            pod_state_store
                .register_pod(&pod_id, worker_info, device_configs)
                .unwrap();

            for proc_idx in 0..processes_per_pod {
                let pid = (pod_idx * processes_per_pod + proc_idx) as u32 + 10000;
                pod_state_store.register_process(&pod_id, pid).unwrap();
            }
        }

        // Verify all registrations
        let stats = pod_state_store.stats();
        assert_eq!(stats.total_pods, num_pods);
        assert_eq!(stats.total_processes, num_pods * processes_per_pod);

        // Test queries work correctly with large dataset
        for pod_idx in 0..num_pods {
            let pod_id = format!("stress-pod-{pod_idx:03}");
            assert!(pod_state_store.contains_pod(&pod_id));

            let host_pids = pod_state_store.get_host_pids_for_pod(&pod_id).unwrap();
            assert_eq!(host_pids.len(), processes_per_pod);
        }

        // Test device queries
        let pods_using_device_0 = pod_state_store.get_pods_using_device(0);
        assert_eq!(pods_using_device_0.len(), num_pods);

        // Cleanup all pods by removing processes
        for pod_idx in 0..num_pods {
            let pod_id = format!("stress-pod-{pod_idx:03}");

            // Remove all but one process
            for proc_idx in 0..(processes_per_pod - 1) {
                let pid = (pod_idx * processes_per_pod + proc_idx) as u32 + 10000;
                let pod_removed = pod_state_store.unregister_process(&pod_id, pid).unwrap();
                assert!(!pod_removed); // Pod should still exist
            }

            // Remove last process (should remove pod)
            let last_pid = (pod_idx * processes_per_pod + processes_per_pod - 1) as u32 + 10000;
            let pod_removed = pod_state_store
                .unregister_process(&pod_id, last_pid)
                .unwrap();
            assert!(pod_removed); // Pod should be removed
        }

        // Final verification
        let final_stats = pod_state_store.stats();
        assert_eq!(final_stats.total_pods, 0);
        assert_eq!(final_stats.total_processes, 0);
    }
}
