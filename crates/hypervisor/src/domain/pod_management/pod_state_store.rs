//! Unified pod state management store
//!
//! This module provides a centralized, thread-safe store for all pod-related state,
//! eliminating the need for multiple synchronized registries and providing atomic operations.

use std::collections::HashSet;
use std::sync::Arc;

use api_types::WorkerInfo;
use dashmap::DashMap;
use tracing::debug;
use tracing::info;
use utils::shared_memory::{handle::SharedMemoryHandle, DeviceConfig};

use super::types::{PodManagementError, PodStatus, Result, SystemStats};

// Re-export unified types for backward compatibility
use super::types::{PodState, ProcessState};

/// Centralized pod state store providing atomic operations
///
/// This store maintains all pod-related state in a thread-safe manner,
/// eliminating the need for multiple synchronized registries.
#[derive(Debug)]
pub struct PodStateStore {
    /// All pod states, keyed by pod identifier
    pods: DashMap<String, PodState>,
    /// Reverse mapping from host PID to pod identifier
    pid_to_pod: DashMap<u32, String>,
}

impl Default for PodStateStore {
    fn default() -> Self {
        Self::new()
    }
}

impl PodStateStore {
    /// Create a new pod state store
    pub fn new() -> Self {
        Self {
            pods: DashMap::new(),
            pid_to_pod: DashMap::new(),
        }
    }

    /// Register a pod with device configurations
    ///
    /// This operation is idempotent - if the pod already exists, it will be updated.
    pub fn register_pod(
        &self,
        pod_identifier: &str,
        info: WorkerInfo,
        device_configs: Vec<DeviceConfig>,
    ) -> Result<()> {
        let mut pod_state = PodState::new(info.clone(), device_configs);
        let pod_identifier = pod_identifier.to_string();
        // If pod already exists, preserve existing processes and shared memory handle
        if let Some(existing) = self.pods.get(&pod_identifier) {
            pod_state.processes = existing.processes.clone();
            pod_state.shared_memory_handle = existing.shared_memory_handle.clone();
            debug!(
                pod_identifier = %pod_identifier,
                existing_processes = existing.processes.len(),
                "Updated existing pod registration"
            );
        }

        let device_count = pod_state.device_configs.len();
        info!(
            pod_identifier = %pod_identifier,
            device_count = device_count,
            "Pod registered in state store"
        );

        self.pods.insert(pod_identifier.to_string(), pod_state);

        Ok(())
    }

    /// Register a process within a pod
    ///
    /// If the pod doesn't exist, this will return an error.
    pub fn register_process(
        &self,
        pod_identifier: &str,
        host_pid: u32,
        container_pid: u32,
        container_name: String,
    ) -> Result<()> {
        let mut pod_ref = self.pods.get_mut(pod_identifier).ok_or_else(|| {
            PodManagementError::PodIdentifierNotFound {
                pod_identifier: pod_identifier.to_string(),
            }
        })?;

        let process_state = ProcessState::new(host_pid, container_pid, container_name);

        pod_ref.add_process(process_state);
        self.pid_to_pod.insert(host_pid, pod_identifier.to_string());

        info!(
            pod_identifier = %pod_identifier,
            host_pid = host_pid,
            container_pid = container_pid,
            "Process registered in pod state store"
        );

        Ok(())
    }

    /// Unregister a process from a pod
    ///
    /// Returns true if the pod should be removed (no more processes).
    pub fn unregister_process(&self, pod_identifier: &str, host_pid: u32) -> Result<bool> {
        // Remove PID mapping first
        self.pid_to_pod.remove(&host_pid);

        let should_remove_pod = {
            let mut pod_ref = self.pods.get_mut(pod_identifier).ok_or_else(|| {
                PodManagementError::PodIdentifierNotFound {
                    pod_identifier: pod_identifier.to_string(),
                }
            })?;

            pod_ref.remove_process(host_pid);
            !pod_ref.has_processes()
        };

        if should_remove_pod {
            self.pods.remove(pod_identifier);
            info!(
                pod_identifier = %pod_identifier,
                host_pid = host_pid,
                "Process unregistered and pod removed (no more processes)"
            );
        } else {
            info!(
                pod_identifier = %pod_identifier,
                host_pid = host_pid,
                "Process unregistered from pod"
            );
        }

        Ok(should_remove_pod)
    }

    /// Get pod state by identifier
    pub fn get_pod(&self, pod_identifier: &str) -> Option<PodState> {
        self.pods.get(pod_identifier).map(|entry| entry.clone())
    }

    /// Get pod identifier by process PID
    pub fn get_pod_by_pid(&self, host_pid: u32) -> Option<String> {
        self.pid_to_pod.get(&host_pid).map(|entry| entry.clone())
    }

    /// Check if a pod exists
    pub fn contains_pod(&self, pod_identifier: &str) -> bool {
        self.pods.contains_key(pod_identifier)
    }

    /// Get all pod identifiers using a specific device
    pub fn get_pods_using_device(&self, device_idx: u32) -> Vec<String> {
        self.pods
            .iter()
            .filter(|entry| entry.value().uses_device(device_idx))
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get host PIDs for a specific pod
    pub fn get_host_pids_for_pod(&self, pod_identifier: &str) -> Option<Vec<u32>> {
        self.pods.get(pod_identifier).map(|pod| pod.get_host_pids())
    }

    /// Get a device config for a pod and device index
    // TODO: avoid cloning the device config
    pub fn get_device_config_for_pod(
        &self,
        pod_identifier: &str,
        device_idx: u32,
    ) -> Option<DeviceConfig> {
        self.pods.get(pod_identifier).and_then(|pod| {
            pod.device_configs
                .iter()
                .find(|cfg| cfg.device_idx == device_idx)
                .cloned()
        })
    }

    /// Set shared memory handle for a pod
    pub fn set_shared_memory_handle(
        &self,
        pod_identifier: &str,
        handle: Arc<SharedMemoryHandle>,
    ) -> Result<()> {
        let mut pod_ref = self.pods.get_mut(pod_identifier).ok_or_else(|| {
            PodManagementError::PodIdentifierNotFound {
                pod_identifier: pod_identifier.to_string(),
            }
        })?;

        pod_ref.shared_memory_handle = Some(handle);

        debug!(
            pod_identifier = %pod_identifier,
            "Shared memory handle set for pod"
        );

        Ok(())
    }

    /// Update pod status
    pub fn update_pod_status(&self, pod_identifier: &str, status: PodStatus) -> Result<()> {
        let mut pod_ref = self.pods.get_mut(pod_identifier).ok_or_else(|| {
            PodManagementError::PodIdentifierNotFound {
                pod_identifier: pod_identifier.to_string(),
            }
        })?;

        pod_ref.status = status;

        debug!(
            pod_identifier = %pod_identifier,
            status = ?pod_ref.status,
            "Pod status updated"
        );

        Ok(())
    }

    /// Get store statistics
    pub fn stats(&self) -> SystemStats {
        let total_pods = self.pods.len();
        let total_processes = self.pid_to_pod.len();
        let gpu_pods = self
            .pods
            .iter()
            .filter(|entry| !entry.device_configs.is_empty())
            .count();

        let active_devices = self
            .pods
            .iter()
            .flat_map(|entry| {
                entry
                    .device_configs
                    .iter()
                    .map(|config| config.device_idx)
                    .collect::<Vec<_>>()
            })
            .collect::<HashSet<_>>()
            .len();

        SystemStats {
            total_pods,
            total_processes,
            gpu_pods,
            active_devices,
        }
    }

    /// List all pod identifiers
    pub fn list_pod_identifiers(&self) -> Vec<String> {
        self.pods.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Get all processes for a pod
    pub fn get_pod_processes(&self, pod_identifier: &str) -> Vec<ProcessState> {
        self.pods
            .get(pod_identifier)
            .map(|pod| pod.processes.values().cloned().collect())
            .unwrap_or_default()
    }
}

// Re-export system stats for backward compatibility
pub use super::types::SystemStats as StoreStats;

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use api_types::QosLevel;
    use utils::shared_memory::DeviceConfig;

    use super::*;

    fn create_test_worker_info(pod_name: &str) -> WorkerInfo {
        WorkerInfo {
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
        let store = PodStateStore::new();
        let info = create_test_worker_info("test-pod");
        let device_configs = vec![create_test_device_config()];

        // Test pod registration
        store
            .register_pod("pod-1", info.clone(), device_configs)
            .unwrap();

        assert!(store.contains_pod("pod-1"));
        assert_eq!(store.get_pod("pod-1").unwrap().info.pod_name, "test-pod");

        store
            .register_process("pod-1", 1234, 5678, "main".to_string())
            .unwrap();

        assert_eq!(store.get_pod_by_pid(1234), Some("pod-1".to_string()));

        let pod_state = store.get_pod("pod-1").unwrap();
        assert_eq!(pod_state.processes.len(), 1);
        assert!(pod_state.get_process(1234).is_some());

        // Test process unregistration
        let should_remove = store.unregister_process("pod-1", 1234).unwrap();
        assert!(should_remove); // Pod should be removed as it has no more processes

        assert!(!store.contains_pod("pod-1"));
        assert_eq!(store.get_pod_by_pid(1234), None);
    }

    #[test]
    fn test_device_filtering() {
        let store = PodStateStore::new();
        let info = create_test_worker_info("test-pod");
        let device_configs = vec![create_test_device_config()];

        store.register_pod("pod-1", info, device_configs).unwrap();

        let pods_using_device_0 = store.get_pods_using_device(0);
        assert_eq!(pods_using_device_0.len(), 1);
        assert_eq!(pods_using_device_0[0], "pod-1");

        let pods_using_device_1 = store.get_pods_using_device(1);
        assert_eq!(pods_using_device_1.len(), 0);
    }

    #[test]
    fn test_store_stats() {
        let store = PodStateStore::new();
        let info = create_test_worker_info("test-pod");
        let device_configs = vec![create_test_device_config()];

        // Initially empty
        let stats = store.stats();
        assert_eq!(stats.total_pods, 0);
        assert_eq!(stats.total_processes, 0);

        // Add pod
        store.register_pod("pod-1", info, device_configs).unwrap();

        let stats = store.stats();
        assert_eq!(stats.total_pods, 1);
        assert_eq!(stats.total_processes, 0);

        store
            .register_process("pod-1", 1234, 5678, "main".to_string())
            .unwrap();

        let stats = store.stats();
        assert_eq!(stats.total_pods, 1);
        assert_eq!(stats.total_processes, 1);
    }
}
