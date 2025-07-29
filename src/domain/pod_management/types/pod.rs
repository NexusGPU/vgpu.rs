//! Pod and container related types

use std::collections::HashMap;
use std::fmt;
use api_types::WorkerInfo;
use super::worker::Worker;
use super::device::DeviceAllocation;

/// Unique identifier for a pod
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PodId(String);

impl PodId {
    /// Create a new pod ID from namespace and pod name
    pub fn new(namespace: &str, pod_name: &str) -> Self {
        Self(format!("tf_shm_{namespace}_{pod_name}"))
    }

    /// Create from worker info
    pub fn from_worker_info(worker_info: &WorkerInfo) -> Self {
        Self::new(&worker_info.namespace, &worker_info.pod_name)
    }

    /// Get the underlying string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PodId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for PodId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Unique identifier for a container within a pod
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContainerId {
    pod_id: PodId,
    container_name: String,
}

impl ContainerId {
    pub fn new(pod_id: PodId, container_name: String) -> Self {
        Self { pod_id, container_name }
    }

    pub fn pod_id(&self) -> &PodId {
        &self.pod_id
    }

    pub fn container_name(&self) -> &str {
        &self.container_name
    }
}

/// Pod status indicating its current state
#[derive(Debug, Clone, PartialEq)]
pub enum PodStatus {
    Pending,
    Running,
    Terminating,
    Terminated,
    Failed,
}

/// Container information within a pod
#[derive(Debug, Clone)]
pub struct Container {
    pub id: ContainerId,
    pub workers: HashMap<u32, Worker>, // host_pid -> Worker
    pub status: PodStatus,
}

impl Container {
    pub fn new(id: ContainerId) -> Self {
        Self {
            id,
            workers: HashMap::new(),
            status: PodStatus::Pending,
        }
    }

    /// Add a worker to this container
    pub fn add_worker(&mut self, host_pid: u32, worker: Worker) {
        self.workers.insert(host_pid, worker);
        if self.status == PodStatus::Pending {
            self.status = PodStatus::Running;
        }
    }

    /// Remove a worker from this container
    pub fn remove_worker(&mut self, host_pid: u32) -> Option<Worker> {
        let worker = self.workers.remove(&host_pid);
        if self.workers.is_empty() && self.status == PodStatus::Running {
            self.status = PodStatus::Terminated;
        }
        worker
    }

    /// Check if container is empty
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Get all host PIDs
    pub fn host_pids(&self) -> Vec<u32> {
        self.workers.keys().copied().collect()
    }
}

/// Pod representation combining metadata and runtime state
#[derive(Debug, Clone)]
pub struct Pod {
    pub id: PodId,
    pub info: WorkerInfo,
    pub containers: HashMap<String, Container>,
    pub device_allocation: DeviceAllocation,
    pub status: PodStatus,
}

impl Pod {
    /// Create a new pod from worker info
    pub fn new(worker_info: WorkerInfo, device_allocation: DeviceAllocation) -> Self {
        let id = PodId::from_worker_info(&worker_info);
        let mut containers = HashMap::new();

        // Initialize containers from WorkerInfo
        if let Some(container_names) = &worker_info.containers {
            for container_name in container_names {
                let container_id = ContainerId::new(id.clone(), container_name.clone());
                containers.insert(container_name.clone(), Container::new(container_id));
            }
        }

        Self {
            id,
            info: worker_info,
            containers,
            device_allocation,
            status: PodStatus::Pending,
        }
    }

    /// Get container by name
    pub fn get_container(&self, container_name: &str) -> Option<&Container> {
        self.containers.get(container_name)
    }

    /// Get mutable container by name
    pub fn get_container_mut(&mut self, container_name: &str) -> Option<&mut Container> {
        self.containers.get_mut(container_name)
    }

    /// Add worker to a container
    pub fn add_worker(&mut self, container_name: &str, host_pid: u32, worker: Worker) -> bool {
        if let Some(container) = self.containers.get_mut(container_name) {
            container.add_worker(host_pid, worker);
            if self.status == PodStatus::Pending {
                self.status = PodStatus::Running;
            }
            true
        } else {
            false
        }
    }

    /// Remove worker from a container
    pub fn remove_worker(&mut self, container_name: &str, host_pid: u32) -> Option<Worker> {
        if let Some(container) = self.containers.get_mut(container_name) {
            let worker = container.remove_worker(host_pid);
            
            // Update pod status if all containers are empty
            if self.containers.values().all(|c| c.is_empty()) {
                self.status = PodStatus::Terminated;
            }
            
            worker
        } else {
            None
        }
    }

    /// Check if pod is empty (no workers)
    pub fn is_empty(&self) -> bool {
        self.containers.values().all(|c| c.is_empty())
    }

    /// Get all host PIDs in this pod
    pub fn all_host_pids(&self) -> Vec<u32> {
        self.containers
            .values()
            .flat_map(|c| c.host_pids())
            .collect()
    }

    /// Mark pod as terminating
    pub fn mark_terminating(&mut self) {
        self.status = PodStatus::Terminating;
        for container in self.containers.values_mut() {
            if container.status == PodStatus::Running {
                container.status = PodStatus::Terminating;
            }
        }
    }
}