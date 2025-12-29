//! Unified type definitions for pod management
//!
//! This module contains all the core data structures used across the pod management system,
//! eliminating duplication and providing a single source of truth for type definitions.

use std::collections::HashSet;
use std::sync::Arc;

use api_types::PodResourceInfo;
use utils::shared_memory::{handle::SharedMemoryHandle, DeviceConfig, PodIdentifier};

/// Unified error type for pod management operations
#[derive(Debug, thiserror::Error)]
pub enum PodManagementError {
    #[error("Pod not found: {namespace}/{pod_name}")]
    PodNotFound { namespace: String, pod_name: String },

    #[error("Pod identifier not found: {pod_identifier}")]
    PodIdentifierNotFound { pod_identifier: PodIdentifier },

    #[error("Pod not empty: {pod_identifier}")]
    PodNotEmpty { pod_identifier: PodIdentifier },

    #[error("Process not found: {pid}")]
    ProcessNotFound { pid: u32 },

    #[error("Device configuration error: {message}")]
    DeviceError { message: String },

    #[error("State inconsistency: {message}")]
    StateError { message: String },

    #[error("Registration failed: {message}")]
    RegistrationFailed { message: String },

    #[error("Shared memory error: {message}")]
    SharedMemoryError { message: String },

    #[error("Kubernetes API error: {message}")]
    KubernetesError { message: String },
}

/// Result type for pod management operations
pub type Result<T> = std::result::Result<T, PodManagementError>;

/// Pod status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PodStatus {
    /// Pod is running normally
    #[default]
    Running,
    /// Pod is being terminated
    Terminating,
    /// Pod has been terminated
    Terminated,
}

/// Unified state for a single pod containing all related information
#[derive(Clone)]
pub struct PodState {
    /// Basic pod information from Kubernetes
    pub pod_info: PodResourceInfo,
    /// Device configurations for GPU resources
    pub device_configs: Vec<Arc<DeviceConfig>>,
    /// All processes in this pod, keyed by host PID
    pub processes: HashSet<u32>,
    /// Shared memory handle for GPU coordination
    pub shared_memory_handle: Option<Arc<SharedMemoryHandle>>,
    /// Current pod status
    pub status: PodStatus,
}

impl PodState {
    /// Create a new pod state
    pub fn new(pod_info: PodResourceInfo, device_configs: Vec<Arc<DeviceConfig>>) -> Self {
        Self {
            pod_info,
            device_configs,
            processes: HashSet::new(),
            shared_memory_handle: None,
            status: PodStatus::Running,
        }
    }

    /// Register a process with this pod
    pub fn register_process(&mut self, host_pid: u32) {
        self.processes.insert(host_pid);
    }

    /// Remove a process from this pod
    pub fn remove_process(&mut self, host_pid: u32) -> bool {
        self.processes.remove(&host_pid)
    }

    /// Check if pod has any active processes
    pub fn has_processes(&self, host_pid: u32) -> bool {
        self.processes.contains(&host_pid)
    }

    pub fn is_empty(&self) -> bool {
        self.processes.is_empty()
    }

    /// Get all host PIDs in this pod
    pub fn get_host_pids(&self) -> Vec<u32> {
        let mut pids: Vec<u32> = self.processes.iter().copied().collect();
        pids.sort_unstable();
        pids
    }

    /// Check if pod uses a specific device
    pub fn uses_device(&self, device_idx: u32) -> bool {
        self.device_configs
            .iter()
            .any(|config| config.device_idx == device_idx)
    }

    /// Get processes using a specific device
    pub fn get_processes_for_device(&self, device_idx: u32) -> Vec<u32> {
        if self.uses_device(device_idx) {
            self.get_host_pids()
        } else {
            Vec::new()
        }
    }
}

impl std::fmt::Debug for PodState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedPodState")
            .field("pod_info", &self.pod_info)
            .field("device_configs", &self.device_configs)
            .field("processes", &self.processes)
            .field("shared_memory_handle", &self.shared_memory_handle.is_some())
            .field("status", &self.status)
            .finish()
    }
}

/// Helper trait for converting errors to our unified error type
pub trait IntoResult<T> {
    fn into_pod_management_result(self) -> Result<T>;
}

impl<T, E> IntoResult<T> for std::result::Result<T, E>
where
    E: std::fmt::Display,
{
    fn into_pod_management_result(self) -> Result<T> {
        self.map_err(|e| PodManagementError::StateError {
            message: e.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use api_types::QosLevel;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use utils::shared_memory::DeviceConfig;

    use super::*;

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
    fn test_device_filtering() {
        let info = create_test_worker_info("test-pod");
        let device_configs = vec![Arc::new(create_test_device_config())];
        let pod_state = PodState::new(info, device_configs);

        assert!(pod_state.uses_device(0));
        assert!(!pod_state.uses_device(1));
    }
}
