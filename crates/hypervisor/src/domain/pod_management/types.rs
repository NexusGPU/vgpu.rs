//! Unified type definitions for pod management
//!
//! This module contains all the core data structures used across the pod management system,
//! eliminating duplication and providing a single source of truth for type definitions.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use api_types::WorkerInfo;
use utils::shared_memory::handle::SharedMemoryHandle;
use utils::shared_memory::DeviceConfig;

use crate::process::worker::TensorFusionWorker;
use thiserror::Error;

/// Unified error type for pod management operations
#[derive(Debug, Error)]
pub enum PodManagementError {
    #[error("Pod not found: {namespace}/{pod_name}")]
    PodNotFound { namespace: String, pod_name: String },

    #[error("Pod identifier not found: {pod_identifier}")]
    PodIdentifierNotFound { pod_identifier: String },

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PodStatus {
    /// Pod is running normally
    Running,
    /// Pod is being terminated
    Terminating,
    /// Pod has been terminated
    Terminated,
}

impl Default for PodStatus {
    fn default() -> Self {
        Self::Running
    }
}

/// Unified state for a single pod containing all related information
#[derive(Clone)]
pub struct PodState {
    /// Basic pod information from Kubernetes
    pub info: WorkerInfo,
    /// Device configurations for GPU resources
    pub device_configs: Vec<DeviceConfig>,
    /// All processes in this pod, keyed by host PID
    pub processes: HashMap<u32, ProcessState>,
    /// Shared memory handle for GPU coordination
    pub shared_memory_handle: Option<Arc<SharedMemoryHandle>>,
    /// Current pod status
    pub status: PodStatus,
}

/// Unified state information for a single process within a pod
#[derive(Debug, Clone)]
pub struct ProcessState {
    /// Host process ID
    pub host_pid: u32,
    /// Container process ID
    pub container_pid: u32,
    /// Container name this process belongs to
    pub container_name: String,
    /// Worker instance for this process
    pub worker: Arc<TensorFusionWorker>,
}

/// Device usage information for compatibility with existing components
#[derive(Debug, Clone)]
pub struct DeviceUsage {
    pub device_configs: Vec<DeviceConfig>,
    /// Set of active host PIDs
    pub active_processes: HashSet<u32>,
}

impl PodState {
    /// Create a new pod state
    pub fn new(info: WorkerInfo, device_configs: Vec<DeviceConfig>) -> Self {
        Self {
            info,
            device_configs,
            processes: HashMap::new(),
            shared_memory_handle: None,
            status: PodStatus::Running,
        }
    }

    /// Add a process to this pod
    pub fn add_process(&mut self, process_state: ProcessState) {
        self.processes.insert(process_state.host_pid, process_state);
    }

    /// Remove a process from this pod
    pub fn remove_process(&mut self, host_pid: u32) -> Option<ProcessState> {
        self.processes.remove(&host_pid)
    }

    /// Check if pod has any active processes
    pub fn has_processes(&self) -> bool {
        !self.processes.is_empty()
    }

    /// Get all host PIDs in this pod
    pub fn get_host_pids(&self) -> Vec<u32> {
        self.processes.keys().copied().collect()
    }

    /// Get process state by host PID
    pub fn get_process(&self, host_pid: u32) -> Option<&ProcessState> {
        self.processes.get(&host_pid)
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

    /// Convert to device usage format (for backward compatibility)
    pub fn to_device_usage(&self) -> DeviceUsage {
        DeviceUsage {
            device_configs: self.device_configs.clone(),
            active_processes: self.processes.keys().copied().collect(),
        }
    }

    /// Get processes grouped by container
    pub fn get_processes_by_container(&self) -> HashMap<String, Vec<&ProcessState>> {
        let mut container_processes: HashMap<String, Vec<&ProcessState>> = HashMap::new();

        for process in self.processes.values() {
            container_processes
                .entry(process.container_name.clone())
                .or_default()
                .push(process);
        }

        container_processes
    }

    /// Get all container names in this pod
    pub fn get_container_names(&self) -> Vec<String> {
        self.processes
            .values()
            .map(|p| p.container_name.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }
}

impl std::fmt::Debug for PodState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedPodState")
            .field("info", &self.info)
            .field("device_configs", &self.device_configs)
            .field("processes", &self.processes)
            .field("shared_memory_handle", &self.shared_memory_handle.is_some())
            .field("status", &self.status)
            .finish()
    }
}

impl ProcessState {
    /// Create a new process state
    pub fn new(
        host_pid: u32,
        container_pid: u32,
        container_name: String,
        worker: Arc<TensorFusionWorker>,
    ) -> Self {
        Self {
            host_pid,
            container_pid,
            container_name,
            worker,
        }
    }
}

impl DeviceUsage {
    pub fn new(device_configs: Vec<DeviceConfig>) -> Self {
        Self {
            device_configs,
            active_processes: HashSet::new(),
        }
    }

    pub fn add_process(&mut self, host_pid: u32) {
        self.active_processes.insert(host_pid);
    }

    pub fn remove_process(&mut self, host_pid: u32) -> bool {
        self.active_processes.remove(&host_pid);
        self.active_processes.is_empty()
    }

    /// Gets all host_pids in the usage.
    pub fn get_host_pids(&self) -> Vec<u32> {
        self.active_processes.iter().copied().collect()
    }
}

/// Statistics about the pod management system
#[derive(Debug, Clone)]
pub struct SystemStats {
    /// Total number of pods
    pub total_pods: usize,
    /// Total number of processes across all pods
    pub total_processes: usize,
    /// Number of pods with GPU resources
    pub gpu_pods: usize,
    /// Number of active devices
    pub active_devices: usize,
}

// Conversion implementations for backward compatibility
impl From<PodState> for DeviceUsage {
    fn from(pod_state: PodState) -> Self {
        pod_state.to_device_usage()
    }
}

impl From<&PodState> for DeviceUsage {
    fn from(pod_state: &PodState) -> Self {
        pod_state.to_device_usage()
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
    use nvml_wrapper::Nvml;
    use std::collections::BTreeMap;

    use utils::shared_memory::DeviceConfig;

    use super::*;
    use crate::{gpu_observer::GpuObserver, limiter_comm::CommandDispatcher};

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

    fn create_test_worker() -> Arc<TensorFusionWorker> {
        Arc::new(TensorFusionWorker::new(
            1234,
            QosLevel::Medium,
            vec!["GPU-12345".to_string()],
            GpuObserver::create(Arc::new(Nvml::init().unwrap())),
            "default".to_string(),
            "test-pod".to_string(),
            Arc::new(CommandDispatcher::new()),
        ))
    }

    #[test]
    fn test_unified_pod_state_creation() {
        let info = create_test_worker_info("test-pod");
        let device_configs = vec![create_test_device_config()];

        let pod_state = PodState::new(info.clone(), device_configs.clone());

        assert_eq!(pod_state.info.pod_name, "test-pod");
        assert_eq!(pod_state.device_configs.len(), 1);
        assert!(pod_state.processes.is_empty());
        assert_eq!(pod_state.status, PodStatus::Running);
    }

    #[test]
    fn test_unified_process_state_creation() {
        let worker = create_test_worker();
        let process_state = ProcessState::new(1234, 5678, "main".to_string(), worker);

        assert_eq!(process_state.host_pid, 1234);
        assert_eq!(process_state.container_pid, 5678);
        assert_eq!(process_state.container_name, "main");
    }

    #[test]
    fn test_pod_state_process_management() {
        let info = create_test_worker_info("test-pod");
        let device_configs = vec![create_test_device_config()];
        let mut pod_state = PodState::new(info, device_configs);

        // Add process
        let worker = create_test_worker();
        let process_state = ProcessState::new(1234, 5678, "main".to_string(), worker);
        pod_state.add_process(process_state);

        assert!(pod_state.has_processes());
        assert_eq!(pod_state.get_host_pids(), vec![1234]);
        assert!(pod_state.get_process(1234).is_some());

        // Remove process
        let removed = pod_state.remove_process(1234);
        assert!(removed.is_some());
        assert!(!pod_state.has_processes());
    }

    #[test]
    fn test_device_usage_conversion() {
        let info = create_test_worker_info("test-pod");
        let device_configs = vec![create_test_device_config()];
        let mut pod_state = PodState::new(info, device_configs);

        let worker = create_test_worker();
        let process_state = ProcessState::new(1234, 5678, "main".to_string(), worker);
        pod_state.add_process(process_state);

        let device_usage = pod_state.to_device_usage();
        assert_eq!(device_usage.device_configs.len(), 1);
        assert_eq!(device_usage.active_processes.len(), 1);
        assert!(device_usage.active_processes.contains(&1234));
    }

    #[test]
    fn test_device_filtering() {
        let info = create_test_worker_info("test-pod");
        let device_configs = vec![create_test_device_config()];
        let pod_state = PodState::new(info, device_configs);

        assert!(pod_state.uses_device(0));
        assert!(!pod_state.uses_device(1));
    }

    #[test]
    fn test_container_grouping() {
        let info = create_test_worker_info("test-pod");
        let device_configs = vec![create_test_device_config()];
        let mut pod_state = PodState::new(info, device_configs);

        // Add processes to different containers
        let worker1 = create_test_worker();
        let worker2 = create_test_worker();

        pod_state.add_process(ProcessState::new(1234, 5678, "main".to_string(), worker1));
        pod_state.add_process(ProcessState::new(
            1235,
            5679,
            "sidecar".to_string(),
            worker2,
        ));

        let containers = pod_state.get_processes_by_container();
        assert_eq!(containers.len(), 2);
        assert!(containers.contains_key("main"));
        assert!(containers.contains_key("sidecar"));

        let container_names = pod_state.get_container_names();
        assert_eq!(container_names.len(), 2);
        assert!(container_names.contains(&"main".to_string()));
        assert!(container_names.contains(&"sidecar".to_string()));
    }
}
