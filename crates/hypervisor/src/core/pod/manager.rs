//! Simplified pod manager that handles worker lifecycle with unified state management.

use std::fs;
use std::sync::Arc;
use std::time::Duration;

use api_types::{QosLevel, WorkerInfo};
use nvml_wrapper::Nvml;
use tracing::{info, warn};
use utils::shared_memory::traits::SharedMemoryAccess;
use utils::shared_memory::PodIdentifier;

use crate::core::hypervisor::HypervisorType;
use crate::core::pod::coordinator::LimiterCoordinator;
use crate::core::pod::traits::{DeviceSnapshotProvider, PodStateRepository, TimeSource};
use crate::core::process::worker::TensorFusionWorker;
use crate::core::process::GpuProcess;
use crate::platform::host_pid_probe::{HostPidProbe, PodProcessInfo, SubscriptionRequest};
use crate::platform::k8s::pod_info_cache::PodInfoCache;
use crate::platform::limiter_comm::CommandDispatcher;
use crate::platform::nvml::gpu_observer::GpuObserver;
use tokio_util::sync::CancellationToken;
use utils::keyed_lock::KeyedAsyncLock;

use super::device_info::create_device_configs_from_worker_info;
use super::pod_state_store::PodStateStore;
use super::types::{PodManagementError, Result};

/// Timeout for PID discovery subscription in seconds
const PID_DISCOVERY_TIMEOUT_SECS: u64 = 5;

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
    /// Per-pod registration locks to prevent concurrent registration of same pod
    registration_locks: KeyedAsyncLock<PodIdentifier>,
}

impl<M, P, D, T> PodManager<M, P, D, T> {
    /// Get the pod state store for API queries.
    pub fn pod_state_store(&self) -> &Arc<PodStateStore> {
        &self.pod_state_store
    }
    /// Find a pod by worker PID.
    pub fn find_pod_by_worker_pid(&self, pid: u32) -> Option<PodIdentifier> {
        self.pod_state_store.get_pod_by_pid(pid)
    }

    /// Parse namespace and pod name from a shared memory path identifier
    /// Format: {base_path}/{namespace}/{pod_name}/shm
    pub fn pod_name_namespace<'a>(
        &self,
        identifier: &'a PodIdentifier,
    ) -> Option<(&'a str, &'a str)> {
        Some((&identifier.namespace, &identifier.name))
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
            registration_locks: KeyedAsyncLock::new(),
        }
    }

    pub async fn restore_pod_from_shared_memory(&self) -> Result<()> {
        // Generate glob pattern from coordinator configuration
        let shm_glob_pattern = self.limiter_coordinator.shm_file_glob_pattern();
        let shared_memory_files = self
            .limiter_coordinator
            .find_shared_memory_files(&shm_glob_pattern)
            .await
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
            match self.ensure_pod_registered(namespace, pod_name).await {
                Err(PodManagementError::PodNotFound {
                    namespace,
                    pod_name,
                }) => {
                    tracing::warn!("Pod {namespace}/{pod_name} not found in cache, skipping");
                    if let Err(e) = fs::remove_file(&file) {
                        tracing::error!(
                            "Failed to remove shared memory file: {}: {e}",
                            file.display()
                        );
                    }
                    continue;
                }
                Err(e) => {
                    tracing::error!("Failed to restore pod from shared memory: {identifier}: {e}");
                }
                _ => {}
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
        let pod_identifier = PodIdentifier::new(namespace, pod_name);

        // Check local state first
        if let Some(pod_info) = self.pod_state_store.get_pod_info(&pod_identifier) {
            return Ok(Some(pod_info));
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
    #[tracing::instrument(skip(self), fields(pod = pod_name, namespace = namespace, container = container_name, container_pid = container_pid))]
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

    /// Start the cleanup loop
    #[tracing::instrument(skip(self, cancellation_token), fields(interval_ms = interval.as_millis()))]
    pub async fn run_cleanup_loop(
        &self,
        interval: Duration,
        cancellation_token: CancellationToken,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        // Skip the first immediate tick
        interval_timer.tick().await;

        info!("Starting cleanup loop with interval: {:?}", interval);
        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    info!("Cleanup loop shutdown requested");
                    break;
                }
                _ = interval_timer.tick() => {
                    // Continue with cleanup logic
                }
            }

            info!("Running periodic cleanup of unused shared memory segments");
            // Check for dead processes and clean them up
            if let Err(e) = self.check_and_cleanup_dead_processes().await {
                tracing::error!("Failed to check and cleanup dead processes: {}", e);
            }
        }

        info!("Cleanup loop stopped");
    }

    /// Ensure pod is registered in all components (lazy loading)
    #[tracing::instrument(skip(self), fields(namespace = namespace, pod = pod_name))]
    pub async fn ensure_pod_registered(&self, namespace: &str, pod_name: &str) -> Result<()> {
        let pod_identifier = PodIdentifier::new(namespace, pod_name);

        // Fast path: check if already registered (lock-free)
        if self.pod_state_store.contains_pod(&pod_identifier) {
            return Ok(());
        }

        // Acquire per-pod lock to serialize registration for the same pod
        let _guard = self.registration_locks.lock(&pod_identifier).await;

        // Double-check after acquiring lock
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
            .subscribe(
                subscription_request,
                Duration::from_secs(PID_DISCOVERY_TIMEOUT_SECS),
            )
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
    #[tracing::instrument(skip(self), fields(pod = pod_name, namespace = namespace, container = container_name, host_pid = host_pid))]
    async fn register_process_to_all_components(
        &self,
        pod_name: &str,
        namespace: &str,
        container_name: &str,
        host_pid: u32,
    ) -> Result<()> {
        let pod_identifier = PodIdentifier::new(namespace, pod_name);

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
            .await
            .map_err(|e| PodManagementError::RegistrationFailed {
                message: e.to_string(),
            })?;

        // 3. Open shared memory for the pod
        self.pod_state_store.open_shared_memory(&pod_identifier)?;

        // 4. Add worker to hypervisor
        if !self.hypervisor.process_exists(host_pid).await {
            info!("Adding new worker to hypervisor: {}", worker.name());
            self.hypervisor.add_process(worker.clone()).await;
        }

        info!(
            "Successfully registered process {host_pid} for pod {} container {container_name}",
            pod_identifier
        );

        Ok(())
    }

    /// check_and_cleanup_dead_processes for use in monitoring task
    #[tracing::instrument(skip(self))]
    async fn check_and_cleanup_dead_processes(&self) -> Result<Vec<u32>> {
        let mut dead_pids = Vec::new();

        // Get all tracked PIDs from the state store
        let tracked_pids: Vec<u32> = {
            let mut pids = vec![];

            let pod_identifiers = self.pod_state_store.list_pod_identifiers();

            let len = pod_identifiers.len();
            for pod_identifier in pod_identifiers {
                let processes = self.pod_state_store.get_pod_processes(&pod_identifier);
                if processes.is_empty() {
                    match self
                        .pod_info_cache
                        .pod_exists(&pod_identifier.namespace, &pod_identifier.name)
                        .await
                    {
                        Ok(true) => {}
                        Ok(false) => {
                            info!(
                                "Pod {} has no processes, and pod not found in api server, unregistering",
                                pod_identifier
                            );

                            if let Err(e) = self.pod_state_store.unregister_pod(&pod_identifier) {
                                tracing::error!(
                                    "Failed to unregister pod {}: {}",
                                    pod_identifier,
                                    e
                                );
                            }
                            continue;
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to check pod existence {}: {}",
                                pod_identifier,
                                e
                            );
                        }
                    }
                }
                pids.extend(processes);
            }

            info!(
                "Checking for dead processes in {} pods, found {} tracked PIDs",
                len,
                pids.len()
            );

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
    #[tracing::instrument(skip(self), fields(host_pid = host_pid))]
    async fn handle_process_exited(&self, host_pid: u32) -> Result<()> {
        info!("Processing process exit: host_pid={}", host_pid);

        // Get pod path for this PID
        let pod_id = match self.pod_state_store.get_pod_by_pid(host_pid) {
            Some(pod_id) => pod_id,
            None => {
                warn!("Attempted to cleanup non-tracked process: {}", host_pid);
                return Ok(());
            }
        };

        info!(
            "Found tracked process: host_pid={}, pod={}",
            host_pid, pod_id
        );

        // 1. Remove from hypervisor
        if self.hypervisor.process_exists(host_pid).await {
            info!("Removing process {} from hypervisor", host_pid);
            self.hypervisor.remove_process(host_pid).await;
        }

        // 2. Unregister from limiter coordinator
        if let Err(e) = self
            .limiter_coordinator
            .unregister_process(&pod_id, host_pid)
            .await
        {
            tracing::error!(
                "Failed to unregister process {} from limiter coordinator: {}",
                host_pid,
                e
            );
        }

        // 3. Remove from pod state store (atomic cleanup)
        self.pod_state_store.unregister_process(&pod_id, host_pid)?;

        info!(
            "Successfully cleaned up process {} from pod {}",
            host_pid, pod_id
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::pod::pod_state_store::PodStateStore;
    use std::collections::BTreeMap;
    use utils::shared_memory::DeviceConfig;

    // Note: Mock types would be defined here if needed for full PodManager testing
    // For now, we focus on testing the business logic components that can be isolated

    fn create_test_pod_identifier(name: &str) -> PodIdentifier {
        PodIdentifier::new("test-namespace", name)
    }

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
                .strip_prefix("tf_shm_")
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
            let identifier = format!("tf_shm_{namespace}_{pod_name}");

            // Test parsing back
            let parsed = identifier
                .strip_prefix("tf_shm_")
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
        let pod_state_store = Arc::new(PodStateStore::new("/tmp/test_shm".into()));
        let worker_info = create_test_worker_info();
        let device_configs = vec![create_test_device_config()];
        let pod_identifier = create_test_pod_identifier("test-pod");

        // Test pod registration
        let result =
            pod_state_store.register_pod(&pod_identifier, worker_info.clone(), device_configs);
        assert!(result.is_ok());

        // Verify pod is registered
        assert!(pod_state_store.contains_pod(&pod_identifier));

        // Test process registration
        let result = pod_state_store.register_process(&pod_identifier, 12345);
        assert!(result.is_ok());

        // Verify process is tracked (get_pod_by_pid returns full path with /shm suffix)
        let pod_id = pod_state_store.get_pod_by_pid(12345);
        assert!(pod_id.is_some());
        let pod_id = pod_id.unwrap();
        assert!(pod_id.namespace.contains("test-namespace"));
        assert!(pod_id.name.contains("test-pod"));

        // Test process unregistration
        pod_state_store
            .unregister_process(&pod_identifier, 12345)
            .unwrap();

        assert!(pod_state_store.get_pod_by_pid(12345).is_none());
    }

    #[tokio::test]
    async fn test_multiple_processes_lifecycle() {
        let pod_state_store = Arc::new(PodStateStore::new("/tmp/test_shm".into()));
        let worker_info = create_test_worker_info();
        let device_configs = vec![create_test_device_config()];
        let pod_identifier = create_test_pod_identifier("test-pod");

        // Register pod
        pod_state_store
            .register_pod(&pod_identifier, worker_info, device_configs)
            .unwrap();

        // Register multiple processes
        pod_state_store
            .register_process(&pod_identifier, 12345)
            .unwrap();
        pod_state_store
            .register_process(&pod_identifier, 12346)
            .unwrap();
        pod_state_store
            .register_process(&pod_identifier, 12347)
            .unwrap();
        {
            // Verify all processes are tracked (get_pod_by_pid returns full path with /shm suffix)
            let pod_12345 = pod_state_store.get_pod_by_pid(12345);
            assert!(pod_12345.is_some());
            let pod_12345 = pod_12345.as_ref().unwrap();
            assert!(pod_12345.namespace.contains("test-namespace"));
            assert!(pod_12345.name.contains("test-pod"));

            let path_12346 = pod_state_store.get_pod_by_pid(12346);
            assert!(path_12346.is_some());
            let pod_12346 = path_12346.as_ref().unwrap();
            assert!(pod_12346.namespace.contains("test-namespace"));
            assert!(pod_12346.name.contains("test-pod"));

            let path_12347 = pod_state_store.get_pod_by_pid(12347);
            assert!(path_12347.is_some());
            let pod_12347 = path_12347.as_ref().unwrap();
            assert!(pod_12347.namespace.contains("test-namespace"));
            assert!(pod_12347.name.contains("test-pod"));
        }
        // Unregister first process - pod should remain
        pod_state_store
            .unregister_process(&pod_identifier, 12345)
            .unwrap();
        assert!(pod_state_store.contains_pod(&pod_identifier));

        // Unregister second process - pod should remain
        pod_state_store
            .unregister_process(&pod_identifier, 12346)
            .unwrap();
        assert!(pod_state_store.contains_pod(&pod_identifier));

        // Unregister last process - pod should be removed
        pod_state_store
            .unregister_process(&pod_identifier, 12347)
            .unwrap();
        assert!(pod_state_store.contains_pod(&pod_identifier));
    }

    #[test]
    fn test_error_scenarios() {
        let pod_state_store = Arc::new(PodStateStore::new("/tmp/test_shm".into()));
        let pod_identifier = create_test_pod_identifier("test-pod");

        // Test registering process for non-existent pod
        let result = pod_state_store.register_process(&pod_identifier, 12345);
        assert!(result.is_err());

        // Test unregistering process for non-existent pod
        let result = pod_state_store.unregister_process(&pod_identifier, 12345);
        assert!(result.is_err());

        // Test getting pod for non-existent identifier
        let result = pod_state_store.get_pod(&pod_identifier);
        assert!(result.is_none());

        // Test getting pod by non-existent PID
        let result = pod_state_store.get_pod_by_pid(99999);
        assert!(result.is_none());
    }

    #[test]
    fn test_device_queries() {
        let pod_state_store = Arc::new(PodStateStore::new("/tmp/test_shm".into()));
        let worker_info = create_test_worker_info();
        let pod1_id = create_test_pod_identifier("pod1");
        let pod2_id = create_test_pod_identifier("pod2");

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
            .register_pod(
                &pod1_id,
                worker_info.clone(),
                vec![device_configs[0].clone()],
            )
            .unwrap();
        pod_state_store
            .register_pod(&pod2_id, worker_info, vec![device_configs[1].clone()])
            .unwrap();

        // Test device queries
        let pods_on_device_0 = pod_state_store.get_pods_using_device(0);
        assert_eq!(pods_on_device_0.len(), 1);
        // Note: paths now include namespace/name/shm structure

        let pods_on_device_1 = pod_state_store.get_pods_using_device(1);
        assert_eq!(pods_on_device_1.len(), 1);

        let pods_on_device_2 = pod_state_store.get_pods_using_device(2);
        assert!(pods_on_device_2.is_empty());

        // Test device config retrieval
        let config = pod_state_store.get_device_config_for_pod(&pod1_id, 0);
        assert!(config.is_some());
        assert_eq!(config.unwrap().device_uuid, "GPU-0");

        let config = pod_state_store.get_device_config_for_pod(&pod1_id, 1);
        assert!(config.is_none());

        let config = pod_state_store.get_device_config_for_pod(&pod2_id, 1);
        assert!(config.is_some());
        assert_eq!(config.unwrap().device_uuid, "GPU-1");
    }

    #[tokio::test]
    async fn test_concurrent_pod_operations() {
        use std::sync::Arc;
        use tokio::task::JoinSet;

        let pod_state_store = Arc::new(PodStateStore::new("/tmp/test_shm".into()));

        // Test concurrent registrations and unregistrations
        let mut join_set = JoinSet::new();

        // Start multiple concurrent tasks that register/unregister pods
        for i in 0..50 {
            let store = pod_state_store.clone();
            join_set.spawn(async move {
                let worker_info = create_test_worker_info();
                let device_configs = vec![create_test_device_config()];
                let pod_id = PodIdentifier::new("test-namespace", format!("pod-{i}"));

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
                // Note: get_pod_by_pid returns the full path, not PodIdentifier
                assert!(store.get_pod_by_pid(pid1).is_some());
                assert!(store.get_pod_by_pid(pid2).is_some());

                // Unregister one process
                store.unregister_process(&pod_id, pid1).unwrap();
                assert!(store.contains_pod(&pod_id));

                // Unregister second process
                store.unregister_process(&pod_id, pid2).unwrap();

                // Verify still registered but empty
                assert!(store.contains_pod(&pod_id));
                assert!(store.get_pod_by_pid(pid1).is_none());
                assert!(store.get_pod_by_pid(pid2).is_none());
            });
        }

        // Wait for all tasks to complete
        while let Some(result) = join_set.join_next().await {
            result.unwrap(); // Ensure no task panicked
        }

        assert!(!pod_state_store.list_pod_identifiers().is_empty());
        assert_eq!(pod_state_store.list_all_processes().len(), 0);
    }

    #[tokio::test]
    async fn test_stress_test_with_many_pods_and_processes() {
        let pod_state_store = Arc::new(PodStateStore::new("/tmp/test_shm".into()));

        // Create many pods with many processes each
        let num_pods = 100;
        let processes_per_pod = 20;

        // Register all pods and processes
        for pod_idx in 0..num_pods {
            let worker_info = create_test_worker_info();
            let device_configs = vec![create_test_device_config()];
            let pod_id = PodIdentifier::new("test-namespace", format!("stress-pod-{pod_idx:03}"));

            pod_state_store
                .register_pod(&pod_id, worker_info, device_configs)
                .unwrap();

            for proc_idx in 0..processes_per_pod {
                let pid = (pod_idx * processes_per_pod + proc_idx) as u32 + 10000;
                pod_state_store.register_process(&pod_id, pid).unwrap();
            }
        }

        // Verify all registrations
        assert_eq!(pod_state_store.list_pod_identifiers().len(), num_pods);
        assert_eq!(
            pod_state_store.list_all_processes().len(),
            num_pods * processes_per_pod
        );

        // Test queries work correctly with large dataset
        for pod_idx in 0..num_pods {
            let pod_id = PodIdentifier::new("test-namespace", format!("stress-pod-{pod_idx:03}"));
            assert!(pod_state_store.contains_pod(&pod_id));

            let host_pids = pod_state_store.get_host_pids_for_pod(&pod_id).unwrap();
            assert_eq!(host_pids.len(), processes_per_pod);
        }

        // Test device queries
        let pods_using_device_0 = pod_state_store.get_pods_using_device(0);
        assert_eq!(pods_using_device_0.len(), num_pods);

        // Cleanup all pods by removing processes
        for pod_idx in 0..num_pods {
            let pod_id = PodIdentifier::new("test-namespace", format!("stress-pod-{pod_idx:03}"));

            // Remove all but one process
            for proc_idx in 0..(processes_per_pod - 1) {
                let pid = (pod_idx * processes_per_pod + proc_idx) as u32 + 10000;
                pod_state_store.unregister_process(&pod_id, pid).unwrap();
                assert!(pod_state_store.contains_pod(&pod_id));
            }

            // Remove last process (should remove pod)
            let last_pid = (pod_idx * processes_per_pod + processes_per_pod - 1) as u32 + 10000;
            pod_state_store
                .unregister_process(&pod_id, last_pid)
                .unwrap();
            assert!(pod_state_store.contains_pod(&pod_id));
        }

        assert_eq!(pod_state_store.list_pod_identifiers().len(), num_pods);
        assert_eq!(pod_state_store.list_all_processes().len(), 0);
    }
}
