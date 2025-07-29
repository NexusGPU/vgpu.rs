//! Registry structures for managing pods, containers, and device usage

use crate::process::worker::TensorFusionWorker;
use api_types::WorkerInfo;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use utils::shared_memory::DeviceConfig;

/// Tracks device usage at the pod level.
#[derive(Debug, Clone)]
pub struct PodDeviceUsage {
    pub device_configs: Vec<DeviceConfig>,
    /// Set of active host PIDs in this pod
    pub active_processes: HashSet<u32>,
}

impl PodDeviceUsage {
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

    /// Gets all host_pids in the pod.
    pub fn get_host_pids(&self) -> Vec<u32> {
        self.active_processes.iter().copied().collect()
    }
}

/// Container information for tracking container-specific details.
#[derive(Debug, Clone)]
pub struct ContainerInfo {
    /// Mapping from container PID to host PID for this container
    pub container_pid_to_host_pid: HashMap<u32, u32>,
    /// Workers (processes) in this container, keyed by host PID
    pub workers: HashMap<u32, Arc<TensorFusionWorker>>,
}

impl ContainerInfo {
    pub fn new() -> Self {
        Self {
            container_pid_to_host_pid: HashMap::new(),
            workers: HashMap::new(),
        }
    }

    /// Add a worker to this container
    pub fn add_worker(
        &mut self,
        host_pid: u32,
        container_pid: u32,
        worker: Arc<TensorFusionWorker>,
    ) {
        self.container_pid_to_host_pid
            .insert(container_pid, host_pid);
        self.workers.insert(host_pid, worker);
    }

    /// Check if container has any workers
    pub fn has_workers(&self) -> bool {
        !self.workers.is_empty()
    }
}

/// Pod entry combining pod info with container tracking.
/// Each pod can contain multiple containers, each with multiple worker processes.
#[derive(Debug, Clone)]
pub struct PodEntry {
    pub info: WorkerInfo,
    /// Container information keyed by container name
    pub containers: HashMap<String, ContainerInfo>,
}

impl PodEntry {
    pub fn new(info: WorkerInfo) -> Self {
        let mut containers = HashMap::new();

        // Initialize containers from WorkerInfo
        if let Some(container_names) = &info.containers {
            for container_name in container_names {
                containers.insert(container_name.clone(), ContainerInfo::new());
            }
        }
        Self { info, containers }
    }

    /// Get container info by container name
    pub fn get_container(&self, container_name: &str) -> Option<&ContainerInfo> {
        self.containers.get(container_name)
    }
}

/// Pod registry for storing and managing pod information.
pub type PodRegistry = Arc<RwLock<HashMap<String, PodEntry>>>;

/// PID registry for mapping PIDs to pod entries.
pub type PidToPodRegistry = Arc<RwLock<HashMap<u32, PodEntry>>>;
