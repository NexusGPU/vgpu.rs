//! Worker and process related types

use std::sync::Arc;
use api_types::QosLevel;
use crate::domain::process::worker::TensorFusionWorker;

/// Unique identifier for a worker
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkerId {
    pub host_pid: u32,
    pub container_pid: u32,
}

impl WorkerId {
    pub fn new(host_pid: u32, container_pid: u32) -> Self {
        Self { host_pid, container_pid }
    }
}

/// Worker status indicating its current state
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerStatus {
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed(String),
}

/// Process information for a worker
#[derive(Debug, Clone)]
pub struct ProcessInfo {
    pub host_pid: u32,
    pub container_pid: u32,
    pub container_name: String,
    pub qos_level: QosLevel,
    pub start_time: std::time::SystemTime,
}

impl ProcessInfo {
    pub fn new(
        host_pid: u32,
        container_pid: u32,
        container_name: String,
        qos_level: QosLevel,
    ) -> Self {
        Self {
            host_pid,
            container_pid,
            container_name,
            qos_level,
            start_time: std::time::SystemTime::now(),
        }
    }
}

/// Worker representation combining process info and worker instance
#[derive(Debug, Clone)]
pub struct Worker {
    pub id: WorkerId,
    pub process_info: ProcessInfo,
    pub worker_instance: Arc<TensorFusionWorker>,
    pub status: WorkerStatus,
}

impl Worker {
    pub fn new(
        host_pid: u32,
        container_pid: u32,
        container_name: String,
        qos_level: QosLevel,
        worker_instance: Arc<TensorFusionWorker>,
    ) -> Self {
        let id = WorkerId::new(host_pid, container_pid);
        let process_info = ProcessInfo::new(host_pid, container_pid, container_name, qos_level);
        
        Self {
            id,
            process_info,
            worker_instance,
            status: WorkerStatus::Starting,
        }
    }

    /// Mark worker as running
    pub fn start(&mut self) {
        self.status = WorkerStatus::Running;
    }

    /// Mark worker as stopping
    pub fn stop(&mut self) {
        self.status = WorkerStatus::Stopping;
    }

    /// Mark worker as stopped
    pub fn terminate(&mut self) {
        self.status = WorkerStatus::Stopped;
    }

    /// Mark worker as failed
    pub fn fail(&mut self, reason: String) {
        self.status = WorkerStatus::Failed(reason);
    }

    /// Check if worker is active (running or starting)
    pub fn is_active(&self) -> bool {
        matches!(self.status, WorkerStatus::Starting | WorkerStatus::Running)
    }

    /// Get worker name from the underlying instance
    pub fn name(&self) -> String {
        self.worker_instance.name()
    }

    /// Get worker QoS level
    pub fn qos_level(&self) -> QosLevel {
        self.process_info.qos_level
    }
}