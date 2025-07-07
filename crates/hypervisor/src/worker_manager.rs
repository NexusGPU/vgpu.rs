//! Worker management module.
#![allow(dead_code)]
//! This module provides functionality for managing workers based on Kubernetes pod events.
//! It replaces the old socket-based worker watcher with a pod-centric approach where
//! pods are treated as workers.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Ok;
use anyhow::Result;
use api_types::QosLevel;
use api_types::WorkerInfo;
use tokio::sync::RwLock;
use tracing::info;
use tracing::warn;

use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::HostPidProbe;
use crate::host_pid_probe::PodProcessInfo;
use crate::host_pid_probe::SubscriptionRequest;
use crate::k8s::TensorFusionPodInfo;
use crate::process::worker::TensorFusionWorker;

/// Container-level information within a pod
#[derive(Clone)]
pub struct ContainerInfo {
    pub container_name: String,
    /// Mapping from container PID to host PID for this container
    pub container_pid_to_host_pid: HashMap<u32, u32>,
    /// Optional worker instance for this container
    pub worker: Option<Arc<TensorFusionWorker>>,
}

impl ContainerInfo {
    fn new(container_name: String) -> Self {
        Self {
            container_name,
            container_pid_to_host_pid: HashMap::new(),
            worker: None,
        }
    }
}

/// Entry that combines Kubernetes annotation info and container-level information
#[derive(Clone)]
pub struct WorkerEntry {
    pub info: WorkerInfo,
    /// Container information keyed by container name
    pub containers: HashMap<String, ContainerInfo>,
}

impl WorkerEntry {
    fn new(info: WorkerInfo) -> Self {
        let mut containers = HashMap::new();

        // Initialize containers from WorkerInfo
        if let Some(container_names) = &info.containers {
            for container_name in container_names {
                containers.insert(
                    container_name.clone(),
                    ContainerInfo::new(container_name.clone()),
                );
            }
        }

        Self { info, containers }
    }

    /// Get container info by container name
    pub fn get_container(&self, container_name: &str) -> Option<&ContainerInfo> {
        self.containers.get(container_name)
    }

    /// Get mutable container info by container name
    pub fn get_container_mut(&mut self, container_name: &str) -> Option<&mut ContainerInfo> {
        self.containers.get_mut(container_name)
    }
}

impl Default for WorkerEntry {
    fn default() -> Self {
        Self {
            info: WorkerInfo::default(),
            containers: HashMap::new(),
        }
    }
}

/// Worker registry for storing and managing worker information.
pub type WorkerRegistry = Arc<RwLock<HashMap<String, WorkerEntry>>>;

/// PID registry for mapping PIDs to worker entries.
pub type PidRegistry = Arc<RwLock<HashMap<u32, WorkerEntry>>>;

/// Worker manager that handles worker lifecycle based on pod events.
pub struct WorkerManager<AddCB, RemoveCB> {
    registry: WorkerRegistry,
    pid_registry: PidRegistry,
    add_callback: AddCB,
    remove_callback: RemoveCB,
    host_pid_probe: Arc<HostPidProbe>,
}

impl<AddCB, RemoveCB> WorkerManager<AddCB, RemoveCB> {
    /// Find a worker by its PID.
    pub async fn find_worker_by_pid(&self, pid: u32) -> Option<WorkerEntry> {
        let pid_registry = self.pid_registry.read().await;
        pid_registry.get(&pid).cloned()
    }

    /// Get the worker registry for API queries.
    pub fn registry(&self) -> &WorkerRegistry {
        &self.registry
    }
}

impl<AddCB, RemoveCB> WorkerManager<AddCB, RemoveCB>
where
    AddCB: Fn(u32, Arc<TensorFusionWorker>) + Send + Sync + 'static,
    RemoveCB: Fn(u32) + Send + Sync + 'static,
{
    /// Create a new worker manager with host PID probe.
    pub fn new(
        host_pid_probe: Arc<HostPidProbe>,
        add_callback: AddCB,
        remove_callback: RemoveCB,
    ) -> Self {
        Self {
            registry: Arc::new(RwLock::new(HashMap::new())),
            pid_registry: Arc::new(RwLock::new(HashMap::new())),
            add_callback,
            remove_callback,
            host_pid_probe,
        }
    }

    /// Handle a pod creation event.
    pub async fn handle_pod_created(
        &self,
        pod_name: String,
        namespace: String,
        pod_info: TensorFusionPodInfo,
        gpu_observer: Arc<GpuObserver>,
    ) -> Result<()> {
        let worker_key = format!("{namespace}/{pod_name}");
        info!("Processing pod creation: {worker_key}");

        // Store worker info in registry
        {
            let mut registry = self.registry.write().await;
            registry.insert(worker_key.clone(), WorkerEntry::new(pod_info.0.clone()));
            info!("Added worker to registry: {worker_key}");
        }
        Ok(())
    }

    /// Discover worker PID using HostPidProbe and automatically associate it.
    pub async fn discover_worker_pid(
        &self,
        pod_name: String,
        namespace: String,
        container_name: String,
        container_pid: u32,
        gpu_observer: Arc<GpuObserver>,
    ) -> Result<PodProcessInfo> {
        let subscription_request = SubscriptionRequest {
            pod_name: pod_name.clone(),
            namespace: namespace.clone(),
            container_name: container_name.clone(),
            container_pid: container_pid,
        };

        info!(
            pod_name = %pod_name,
            namespace = %namespace,
            container_name = %container_name,
            "Starting PID discovery for worker"
        );

        let receiver = self.host_pid_probe.subscribe(subscription_request).await;

        // Handle PID discovery result without spawning a task (sequential processing)
        let process_info = receiver
            .await
            .map_err(|_| anyhow::anyhow!("PID discovery subscription was cancelled"))?;

        info!(
            pod_name = %pod_name,
            namespace = %namespace,
            container_name = %container_name,
            host_pid = process_info.host_pid,
            container_pid = process_info.container_pid,
            "Discovered worker PID"
        );

        // Associate the worker with the discovered PID
        self.associate_discovered_worker(
            pod_name.clone(),
            namespace.clone(),
            container_name.clone(),
            process_info.host_pid,
            process_info.container_pid,
            gpu_observer.clone(),
        )
        .await?;

        Ok(process_info)
    }

    /// Associate a worker with a discovered PID.
    async fn associate_discovered_worker(
        &self,
        pod_name: String,
        namespace: String,
        container_name: String,
        host_pid: u32,
        container_pid: u32,
        gpu_observer: Arc<GpuObserver>,
    ) -> Result<()> {
        let worker_key = format!("{namespace}/{pod_name}");

        let mut registry = self.registry.write().await;
        if let Some(entry) = registry.get_mut(&worker_key) {
            let WorkerInfo {
                namespace: info_namespace,
                pod_name: info_pod_name,
                gpu_uuids,
                qos_level,
                ..
            } = &entry.info;

            let gpu_uuids_vec = gpu_uuids.clone().unwrap_or_default();
            let qos = qos_level.unwrap_or(QosLevel::Medium);

            let worker = Arc::new(TensorFusionWorker::new(
                host_pid,
                qos,
                gpu_uuids_vec,
                gpu_observer,
                info_namespace.clone(),
                info_pod_name.clone(),
            ));

            // Find or create container entry
            let container_info = entry
                .containers
                .entry(container_name.clone())
                .or_insert_with(|| ContainerInfo::new(container_name.clone()));

            // Update container info
            container_info.worker = Some(worker.clone());
            container_info
                .container_pid_to_host_pid
                .insert(container_pid, host_pid);

            (self.add_callback)(host_pid, worker);

            // Add to PID registry
            {
                let mut pid_registry = self.pid_registry.write().await;
                pid_registry.insert(host_pid, entry.clone());
            }

            info!(
                "Associated worker {worker_key} container {container_name} with host PID {host_pid} and container PID {container_pid}"
            );
        } else {
            warn!("Attempted to associate PID with non-existent worker: {worker_key}");
            return Err(anyhow::anyhow!("Worker not found in registry"));
        }

        Ok(())
    }

    /// Handle a pod update event.
    pub async fn handle_pod_updated(
        &self,
        pod_name: String,
        namespace: String,
        pod_info: TensorFusionPodInfo,
        node_name: Option<String>,
    ) -> Result<()> {
        let worker_key = format!("{namespace}/{pod_name}");
        info!("Processing pod update: {worker_key}");

        // For now, treat update the same as creation
        // In the future, we might want to handle updates differently
        {
            let mut registry = self.registry.write().await;
            if let Some(entry) = registry.get_mut(&worker_key) {
                entry.info.tflops_request = pod_info.0.tflops_request;
                entry.info.tflops_limit = pod_info.0.tflops_limit;
                entry.info.vram_request = pod_info.0.vram_request;
                entry.info.vram_limit = pod_info.0.vram_limit;
                entry.info.node_name = node_name;
                info!("Updated worker in registry: {worker_key}");
            } else {
                warn!("Attempted to update non-existent worker: {worker_key}");
            }
        }

        Ok(())
    }

    /// Handle a pod deletion event.
    pub async fn handle_pod_deleted(&self, pod_name: String, namespace: String) -> Result<()> {
        let worker_key = format!("{namespace}/{pod_name}");
        info!("Processing pod deletion: {worker_key}");

        {
            let mut registry = self.registry.write().await;
            if let Some(entry) = registry.remove(&worker_key) {
                info!("Removed worker from registry: {worker_key}");

                // Call remove callback for all containers with workers
                for (container_name, container_info) in &entry.containers {
                    if let Some(worker) = &container_info.worker {
                        use crate::process::GpuProcess;
                        info!("Removing worker for container: {container_name}");
                        (self.remove_callback)(worker.pid());
                    }
                }

                {
                    let mut pid_registry = self.pid_registry.write().await;
                    pid_registry.remove(&entry.info.host_pid);
                }
            } else {
                warn!("Attempted to remove non-existent worker: {worker_key}");
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::AtomicU32;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::time::Duration;

    use api_types::QosLevel;

    use super::*;

    #[test]
    fn worker_entry_new_with_containers() {
        let worker_info = WorkerInfo {
            pod_name: "test-pod".to_string(),
            namespace: "test-namespace".to_string(),
            node_name: Some("test-node".to_string()),
            containers: Some(vec!["container1".to_string(), "container2".to_string()]),
            tflops_request: Some(1.0),
            vram_request: Some(1024),
            tflops_limit: Some(2.0),
            vram_limit: Some(2048),
            gpu_uuids: Some(vec!["gpu1".to_string()]),
            qos_level: Some(QosLevel::High),
            host_pid: 0,
            labels: BTreeMap::new(),
            workload_name: Some("unknown".to_string()),
        };

        let entry = WorkerEntry::new(worker_info);

        assert_eq!(entry.containers.len(), 2);
        assert!(entry.containers.contains_key("container1"));
        assert!(entry.containers.contains_key("container2"));
    }

    #[test]
    fn worker_entry_new_without_containers() {
        let worker_info = WorkerInfo {
            pod_name: "test-pod".to_string(),
            namespace: "test-namespace".to_string(),
            node_name: Some("test-node".to_string()),
            containers: None,
            tflops_request: Some(1.0),
            vram_request: Some(1024),
            tflops_limit: Some(2.0),
            vram_limit: Some(2048),
            gpu_uuids: Some(vec!["gpu1".to_string()]),
            qos_level: Some(QosLevel::High),
            host_pid: 0,
            labels: BTreeMap::new(),
            workload_name: Some("unknown".to_string()),
        };

        let entry = WorkerEntry::new(worker_info);

        assert_eq!(entry.containers.len(), 0);
    }

    #[test]
    fn worker_entry_get_container() {
        let worker_info = WorkerInfo {
            pod_name: "test-pod".to_string(),
            namespace: "test-namespace".to_string(),
            node_name: Some("test-node".to_string()),
            containers: Some(vec!["container1".to_string()]),
            tflops_request: Some(1.0),
            vram_request: Some(1024),
            tflops_limit: Some(2.0),
            vram_limit: Some(2048),
            gpu_uuids: Some(vec!["gpu1".to_string()]),
            qos_level: Some(QosLevel::High),
            host_pid: 0,
            labels: BTreeMap::new(),
            workload_name: Some("unknown".to_string()),
        };

        let entry = WorkerEntry::new(worker_info);

        let container = entry.get_container("container1");
        assert!(container.is_some());
        assert_eq!(container.unwrap().container_name, "container1");

        let non_existent = entry.get_container("non-existent");
        assert!(non_existent.is_none());
    }

    #[test]
    fn container_info_new() {
        let container_info = ContainerInfo::new("test-container".to_string());

        assert_eq!(container_info.container_name, "test-container");
        assert!(container_info.container_pid_to_host_pid.is_empty());
        assert!(container_info.worker.is_none());
    }

    #[test]
    fn container_info_pid_mapping() {
        let mut container_info = ContainerInfo::new("test-container".to_string());

        container_info.container_pid_to_host_pid.insert(100, 1234);
        container_info.container_pid_to_host_pid.insert(101, 1235);

        assert_eq!(
            container_info.container_pid_to_host_pid.get(&100),
            Some(&1234)
        );
        assert_eq!(
            container_info.container_pid_to_host_pid.get(&101),
            Some(&1235)
        );
        assert_eq!(container_info.container_pid_to_host_pid.get(&102), None);
    }

    #[tokio::test]
    async fn worker_manager_new() {
        let host_pid_probe = Arc::new(HostPidProbe::new(Duration::from_millis(100)));

        let add_count = Arc::new(AtomicU32::new(0));
        let remove_count = Arc::new(AtomicU32::new(0));

        let add_count_clone = add_count.clone();
        let remove_count_clone = remove_count.clone();

        let add_callback = move |_pid: u32, _worker: Arc<TensorFusionWorker>| {
            add_count_clone.fetch_add(1, Ordering::SeqCst);
        };

        let remove_callback = move |_pid: u32| {
            remove_count_clone.fetch_add(1, Ordering::SeqCst);
        };

        let worker_manager = WorkerManager::new(host_pid_probe, add_callback, remove_callback);

        assert!(worker_manager.registry().read().await.is_empty());
    }
}
