//! Unified pod state management store
//!
//! This module provides a centralized, thread-safe store for all pod-related state,
//! eliminating the need for multiple synchronized registries and providing atomic operations.

use std::path::PathBuf;
use std::sync::Arc;

use api_types::PodResourceInfo;
use dashmap::DashMap;
use tracing::debug;
use tracing::info;
use utils::shared_memory::{handle::SharedMemoryHandle, DeviceConfig, PodIdentifier};

use super::types::{PodManagementError, PodStatus, Result};

// Re-export unified types for backward compatibility
use super::types::PodState;

/// Centralized pod state store providing atomic operations
///
/// This store maintains all pod-related state in a thread-safe manner,
/// eliminating the need for multiple synchronized registries.
#[derive(Debug)]
pub struct PodStateStore {
    /// All pod states, keyed by full path
    pods: DashMap<PodIdentifier, PodState>,
    /// Reverse mapping from host PID to pod path
    pid_to_pod: DashMap<u32, PodIdentifier>,
    /// Base path for shared memory operations
    shm_base_path: PathBuf,
}

impl PodStateStore {
    /// Create a new pod state store with the specified base path
    pub fn new(shm_base_path: PathBuf) -> Self {
        Self {
            pods: DashMap::new(),
            pid_to_pod: DashMap::new(),
            shm_base_path,
        }
    }

    /// Generate full pod path - Format: {base_path}/{namespace}/{pod_name}/shm
    /// This matches the path format used by PodManager
    pub fn pod_path(&self, pod_identifier: &PodIdentifier) -> PathBuf {
        pod_identifier.to_path(&self.shm_base_path)
    }

    /// Get the base path for shared memory operations
    pub fn base_path(&self) -> &PathBuf {
        &self.shm_base_path
    }

    /// Register a pod with device configurations
    ///
    /// This operation is idempotent - if the pod already exists, it will be updated.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if an existing pod was overwritten, or `Ok(false)` if a new pod was created.
    pub fn register_pod(
        &self,
        pod_identifier: &PodIdentifier,
        pod_info: PodResourceInfo,
        device_configs: Vec<DeviceConfig>,
    ) -> Result<bool> {
        use dashmap::mapref::entry::Entry;

        let device_count = device_configs.len();
        let device_configs: Vec<Arc<DeviceConfig>> =
            device_configs.into_iter().map(Arc::new).collect();

        // Use entry API to atomically check and update, avoiding nested locks
        let was_overwritten = match self.pods.entry(pod_identifier.clone()) {
            Entry::Occupied(mut entry) => {
                // Preserve existing runtime state when updating pod config
                let (processes, shm_handle) = {
                    let existing = entry.get();
                    (
                        existing.processes.clone(),
                        existing.shared_memory_handle.clone(),
                    )
                };

                let mut pod_state = PodState::new(pod_info, device_configs);
                pod_state.processes = processes;
                pod_state.shared_memory_handle = shm_handle;

                debug!(
                    namespace = %pod_identifier.namespace,
                    pod_name = %pod_identifier.name,
                    existing_processes = pod_state.processes.len(),
                    "Overwriting existing pod registration"
                );

                entry.insert(pod_state);
                true
            }
            Entry::Vacant(entry) => {
                let pod_state = PodState::new(pod_info, device_configs);
                entry.insert(pod_state);
                false
            }
        };

        info!(
            namespace = %pod_identifier.namespace,
            pod_name = %pod_identifier.name,
            device_count = device_count,
            was_overwritten = was_overwritten,
            "Pod registered in state store"
        );

        Ok(was_overwritten)
    }

    /// Unregister a pod from the state store
    ///
    /// return error if there are still processes registered to the pod
    pub fn unregister_pod(&self, pod_identifier: &PodIdentifier) -> Result<()> {
        // Check pod exists and is empty in a separate scope to release the reference
        {
            let pod_ref = self.pods.get(pod_identifier).ok_or_else(|| {
                PodManagementError::PodIdentifierNotFound {
                    pod_identifier: pod_identifier.clone(),
                }
            })?;

            if !pod_ref.is_empty() {
                return Err(PodManagementError::PodNotEmpty {
                    pod_identifier: pod_identifier.clone(),
                });
            }
            // pod_ref is dropped here, releasing the read reference
        }

        self.pods.remove(pod_identifier);
        Ok(())
    }

    /// Register a process within a pod
    ///
    /// If the pod doesn't exist, this will return an error.
    pub fn register_process(&self, pod_identifier: &PodIdentifier, host_pid: u32) -> Result<()> {
        let mut pod_ref = self.pods.get_mut(pod_identifier).ok_or_else(|| {
            PodManagementError::PodIdentifierNotFound {
                pod_identifier: pod_identifier.clone(),
            }
        })?;

        pod_ref.register_process(host_pid);
        self.pid_to_pod.insert(host_pid, pod_identifier.clone());

        info!(
            namespace = %pod_identifier.namespace,
            pod_name = %pod_identifier.name,
            host_pid = host_pid,
            "Process registered in pod state store"
        );

        Ok(())
    }

    /// Unregister a process from a pod
    ///
    /// Returns Ok(()) after removal.
    pub fn unregister_process(&self, pod_identifier: &PodIdentifier, host_pid: u32) -> Result<()> {
        let pod_empty = {
            let mut pod_ref = self.pods.get_mut(pod_identifier).ok_or_else(|| {
                PodManagementError::PodIdentifierNotFound {
                    pod_identifier: pod_identifier.clone(),
                }
            })?;

            pod_ref.remove_process(host_pid);
            // Remove PID mapping
            self.pid_to_pod.remove(&host_pid);
            pod_ref.is_empty()
        };

        if pod_empty {
            info!(
                namespace = %pod_identifier.namespace,
                pod_name = %pod_identifier.name,
                host_pid = host_pid,
                "Process unregistered (pod has no more processes)"
            );
        }

        Ok(())
    }

    /// Get pod state by identifier
    pub fn get_pod(&self, pod_identifier: &PodIdentifier) -> Option<PodState> {
        self.pods.get(pod_identifier).map(|entry| entry.clone())
    }

    /// Access pod state with a closure without cloning
    ///
    /// This is more efficient than `get_pod()` when you only need to read fields.
    pub fn with_pod<F, R>(&self, pod_identifier: &PodIdentifier, f: F) -> Option<R>
    where
        F: FnOnce(&PodState) -> R,
    {
        self.pods.get(pod_identifier).map(|entry| f(&entry))
    }

    /// Get pod info by identifier without cloning entire state
    pub fn get_pod_info(&self, pod_identifier: &PodIdentifier) -> Option<PodResourceInfo> {
        self.pods
            .get(pod_identifier)
            .map(|entry| entry.pod_info.clone())
    }

    /// Get pod path by process PID
    pub fn get_pod_by_pid(&self, host_pid: u32) -> Option<PodIdentifier> {
        self.pid_to_pod
            .get(&host_pid)
            .map(|entry| entry.value().clone())
    }

    /// Check if a pod exists
    pub fn contains_pod(&self, pod_identifier: &PodIdentifier) -> bool {
        self.pods.contains_key(pod_identifier)
    }

    /// Get all pod identifiers using a specific device
    pub fn get_pods_using_device(&self, device_idx: u32) -> Vec<PodIdentifier> {
        self.pods
            .iter()
            .filter(|entry| entry.value().uses_device(device_idx))
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get host PIDs for a specific pod
    pub fn get_host_pids_for_pod(&self, pod_identifier: &PodIdentifier) -> Option<Vec<u32>> {
        self.pods.get(pod_identifier).map(|pod| pod.get_host_pids())
    }

    /// Get a borrowed device config for a pod and device index
    /// Returns a wrapper that holds the DashMap guard and provides access to the DeviceConfig
    pub fn get_device_config_for_pod(
        &self,
        pod_identifier: &PodIdentifier,
        device_idx: u32,
    ) -> Option<Arc<DeviceConfig>> {
        let guard = self.pods.get(pod_identifier)?;

        guard
            .device_configs
            .iter()
            .find(|cfg| cfg.device_idx == device_idx)
            .map(Arc::clone)
    }

    /// Open shared memory for a pod and store the handle
    /// This method creates and opens the SharedMemoryHandle from the pod identifier
    pub async fn open_shared_memory(&self, pod_identifier: &PodIdentifier) -> Result<()> {
        let pod_path = self.pod_path(pod_identifier);

        // Create SharedMemoryHandle by opening the shared memory
        // Use spawn_blocking to avoid blocking the tokio runtime
        let pod_path_clone = pod_path.clone();
        let handle = tokio::task::spawn_blocking(move || SharedMemoryHandle::open(&pod_path_clone))
            .await
            .map_err(|e| PodManagementError::SharedMemoryError {
                message: format!("Task join error: {e}"),
            })?
            .map_err(|e| PodManagementError::SharedMemoryError {
                message: format!("Failed to open shared memory for {pod_path:?}: {e}"),
            })?;

        let handle = Arc::new(handle);

        let mut pod_ref = self.pods.get_mut(pod_identifier).ok_or_else(|| {
            PodManagementError::PodIdentifierNotFound {
                pod_identifier: pod_identifier.clone(),
            }
        })?;

        pod_ref.shared_memory_handle = Some(handle);

        debug!(
            namespace = %pod_identifier.namespace,
            pod_name = %pod_identifier.name,
            "Shared memory opened and handle stored for pod"
        );

        Ok(())
    }

    /// Update pod status
    pub fn update_pod_status(
        &self,
        pod_identifier: &PodIdentifier,
        status: PodStatus,
    ) -> Result<()> {
        let mut pod_ref = self.pods.get_mut(pod_identifier).ok_or_else(|| {
            PodManagementError::PodIdentifierNotFound {
                pod_identifier: pod_identifier.clone(),
            }
        })?;

        pod_ref.status = status;

        debug!(
            namespace = %pod_identifier.namespace,
            pod_name = %pod_identifier.name,
            status = ?pod_ref.status,
            "Pod status updated"
        );

        Ok(())
    }

    /// List all pod identifiers
    pub fn list_pod_identifiers(&self) -> Vec<PodIdentifier> {
        self.pods.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Get all processes for a pod
    pub fn get_pod_processes(&self, pod_identifier: &PodIdentifier) -> Vec<u32> {
        self.pods
            .get(pod_identifier)
            .map(|pod| pod.processes.iter().copied().collect())
            .unwrap_or_default()
    }

    pub fn list_all_processes(&self) -> Vec<u32> {
        self.pid_to_pod.iter().map(|entry| *entry.key()).collect()
    }

    /// Clear all data (useful for testing)
    #[cfg(test)]
    pub fn clear(&self) {
        self.pods.clear();
        self.pid_to_pod.clear();
    }
}

#[cfg(test)]
impl PodStateStore {
    /// Simple pod registration for testing (without PodResourceInfo complexity)
    pub fn register_test_pod(
        &self,
        pod_identifier: &str,
        device_configs: Vec<DeviceConfig>,
        host_pids: Vec<u32>,
    ) -> Result<()> {
        use std::collections::BTreeMap;

        // Create minimal PodResourceInfo for testing
        let test_info = PodResourceInfo {
            namespace: "test".to_string(),
            pod_name: pod_identifier.to_string(),
            containers: Some(vec!["test-container".to_string()]),
            gpu_uuids: device_configs
                .iter()
                .map(|cfg| cfg.device_uuid.clone())
                .collect::<Vec<_>>()
                .into(),
            qos_level: Some(api_types::QosLevel::Medium),
            tflops_request: Some(1.0),
            tflops_limit: Some(2.0),
            vram_request: Some(1024),
            vram_limit: Some(2048),
            node_name: Some("test-node".to_string()),
            host_pid: host_pids.first().copied().unwrap_or(1234),
            labels: BTreeMap::new(),
            workload_name: Some("test-workload".to_string()),
            compute_shard: false,
            isolation: None,
        };

        // Create PodIdentifier from namespace and pod name
        let pod_id = PodIdentifier::new("test", pod_identifier);

        // Register pod
        self.register_pod(&pod_id, test_info, device_configs)?;

        // Register processes
        for pid in host_pids {
            self.register_process(&pod_id, pid)?;
        }

        Ok(())
    }
}

// Implement PodStateRepository trait directly
impl super::traits::PodStateRepository for PodStateStore {
    fn get_pods_using_device(&self, device_idx: u32) -> Vec<PodIdentifier> {
        self.get_pods_using_device(device_idx)
    }

    fn get_host_pids_for_pod(&self, pod_identifier: &PodIdentifier) -> Option<Vec<u32>> {
        self.get_host_pids_for_pod(pod_identifier)
    }

    fn get_device_config_for_pod(
        &self,
        pod_identifier: &PodIdentifier,
        device_idx: u32,
    ) -> Option<Arc<DeviceConfig>> {
        self.get_device_config_for_pod(pod_identifier, device_idx)
    }

    fn contains_pod(&self, pod_identifier: &PodIdentifier) -> bool {
        self.contains_pod(pod_identifier)
    }

    fn list_pod_identifiers(&self) -> Vec<PodIdentifier> {
        self.list_pod_identifiers()
    }

    fn pod_path(&self, pod_identifier: &PodIdentifier) -> PathBuf {
        self.pod_path(pod_identifier)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use api_types::QosLevel;
    use utils::shared_memory::DeviceConfig;

    use super::*;

    fn create_test_pod_identifier(name: &str) -> PodIdentifier {
        PodIdentifier::new("default", name)
    }

    fn create_test_worker_info(pod_name: &str) -> PodResourceInfo {
        PodResourceInfo {
            namespace: "default".to_string(),
            pod_name: pod_name.to_string(),
            containers: Some(vec!["main".to_string()]),
            gpu_uuids: Some(vec!["GPU-12345".to_string()]),
            qos_level: Some(QosLevel::Medium),
            tflops_request: Some(10.0),
            tflops_limit: Some(20.0),
            vram_request: Some(1024),
            vram_limit: Some(2048),
            node_name: Some("test-node".to_string()),
            host_pid: 1234,
            labels: BTreeMap::new(),
            workload_name: Some("test-workload".to_string()),
            compute_shard: false,
            isolation: None,
        }
    }

    fn create_test_device_config() -> DeviceConfig {
        DeviceConfig {
            device_idx: 0,
            device_uuid: "GPU-12345".to_string(),
            up_limit: 80,
            mem_limit: 2048,
            total_cuda_cores: 2560,
            sm_count: 20,
            max_thread_per_sm: 128,
        }
    }

    #[test]
    fn test_pod_state_store_operations() {
        let store = PodStateStore::new("/tmp/test_shm".into());
        let pod_id = create_test_pod_identifier("pod-1");
        let info = create_test_worker_info("test-pod");
        let device_configs = vec![create_test_device_config()];

        // Test pod registration
        let was_overwritten = store
            .register_pod(&pod_id, info.clone(), device_configs)
            .unwrap();
        assert!(!was_overwritten, "First registration should not overwrite");

        assert!(store.contains_pod(&pod_id));
        assert_eq!(
            store.get_pod(&pod_id).unwrap().pod_info.pod_name,
            "test-pod"
        );

        store.register_process(&pod_id, 1234).unwrap();

        // Note: get_pod_by_pid still returns pod path, as it's used for reverse lookup
        assert!(store.get_pod_by_pid(1234).is_some());

        let pod_state = store.get_pod(&pod_id).unwrap();
        assert_eq!(pod_state.processes.len(), 1);
        assert!(pod_state.has_processes(1234));
        // Test process unregistration
        store.unregister_process(&pod_id, 1234).unwrap();
        assert!(store.contains_pod(&pod_id));
        assert!(store.get_pod_by_pid(1234).is_none());
    }

    #[test]
    fn test_simplified_registration_for_testing() {
        let store = PodStateStore::new("/tmp/test_shm".into());
        // Note: register_test_pod creates PodResourceInfo with namespace "test"
        let pod_id = PodIdentifier::new("test", "test-pod");
        let device_configs = vec![create_test_device_config()];

        // Test simplified registration (this uses pod_path internally)
        store
            .register_test_pod("test-pod", device_configs, vec![1234, 5678])
            .unwrap();

        assert!(store.contains_pod(&pod_id));
        assert_eq!(store.get_host_pids_for_pod(&pod_id), Some(vec![1234, 5678]));
        // Note: get_pods_using_device still returns paths since it's for compatibility
        assert!(!store.get_pods_using_device(0).is_empty());

        // Test clear functionality
        store.clear();
        assert!(!store.contains_pod(&pod_id));
        assert_eq!(store.get_host_pids_for_pod(&pod_id), None);
    }

    #[test]
    fn test_device_filtering() {
        let store = PodStateStore::new("/tmp/test_shm".into());
        let pod_id = create_test_pod_identifier("pod-1");
        let info = create_test_worker_info("test-pod");
        let device_configs = vec![create_test_device_config()];

        let _ = store.register_pod(&pod_id, info, device_configs).unwrap();

        let pods_using_device_0 = store.get_pods_using_device(0);
        assert_eq!(pods_using_device_0.len(), 1);
        // The path contains namespace/name/shm, so we just check it contains our data
        assert!(pods_using_device_0[0].namespace == "default");
        assert!(pods_using_device_0[0].name == "pod-1");

        let pods_using_device_1 = store.get_pods_using_device(1);
        assert_eq!(pods_using_device_1.len(), 0);
    }

    #[test]
    fn test_device_config_lookup_returns_arc() {
        let store = PodStateStore::new("/tmp/test_shm".into());
        let pod_id = create_test_pod_identifier("pod-1");
        let info = create_test_worker_info("test-pod");
        let device_configs = vec![create_test_device_config()];

        let _ = store.register_pod(&pod_id, info, device_configs).unwrap();

        let config = store
            .get_device_config_for_pod(&pod_id, 0)
            .expect("should find device config");
        assert_eq!(config.device_idx, 0);
        assert_eq!(config.device_uuid, "GPU-12345");
        assert_eq!(config.up_limit, 80);

        // Ensure the Arc-returning lookup does not hold locks by performing a mutation afterwards
        store
            .register_process(&pod_id, 4242)
            .expect("should register process after lookup");
        assert!(store
            .get_host_pids_for_pod(&pod_id)
            .unwrap()
            .contains(&4242));

        // Test getting non-existent device config
        assert!(store.get_device_config_for_pod(&pod_id, 999).is_none());
        let non_existent_id = create_test_pod_identifier("non-existent");
        assert!(store
            .get_device_config_for_pod(&non_existent_id, 0)
            .is_none());
    }

    #[test]
    fn test_register_pod_overwrite_detection() {
        let store = PodStateStore::new("/tmp/test_shm".into());
        let pod_id = create_test_pod_identifier("overwrite-test");
        let info = create_test_worker_info("test-pod");
        let device_configs = vec![create_test_device_config()];

        // First registration should not overwrite
        let was_overwritten = store
            .register_pod(&pod_id, info.clone(), device_configs.clone())
            .unwrap();
        assert!(!was_overwritten, "First registration should return false");

        // Register a process
        store.register_process(&pod_id, 1234).unwrap();
        assert_eq!(store.get_host_pids_for_pod(&pod_id).unwrap().len(), 1);

        // Second registration should overwrite and preserve processes
        let was_overwritten = store
            .register_pod(&pod_id, info.clone(), device_configs.clone())
            .unwrap();
        assert!(was_overwritten, "Second registration should return true");

        // Verify processes are preserved after overwrite
        assert_eq!(
            store.get_host_pids_for_pod(&pod_id).unwrap().len(),
            1,
            "Processes should be preserved after overwrite"
        );
        assert!(
            store.get_pod_by_pid(1234).is_some(),
            "PID mapping should be preserved after overwrite"
        );
    }
}
