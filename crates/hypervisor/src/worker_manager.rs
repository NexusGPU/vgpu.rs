//! This module provides functionality for managing workers based on Kubernetes pod events.
//! It replaces the old socket-based worker watcher with a pod-centric approach where
//! pods are treated as workers.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use api_types::QosLevel;
use api_types::WorkerInfo;
use nvml_wrapper::Nvml;
use tokio::sync::RwLock;
use tracing::info;
use tracing::warn;

use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::HostPidProbe;
use crate::host_pid_probe::PodProcessInfo;
use crate::host_pid_probe::SubscriptionRequest;
use crate::hypervisor::Hypervisor;
use crate::k8s::TensorFusionPodInfo;
use crate::limiter_comm::CommandDispatcher;
use crate::limiter_coordinator::LimiterCoordinator;
use crate::process::worker::TensorFusionWorker;
use crate::process::GpuProcess;
use crate::scheduler::weighted::WeightedScheduler;
use crate::worker_registration::register_worker_to_limiter_coordinator;
use crate::worker_registration::unregister_worker_from_limiter_coordinator;

/// Container information for tracking container-specific details.
#[derive(Debug, Clone)]
pub struct ContainerInfo {
    /// Mapping from container PID to host PID for this container
    pub container_pid_to_host_pid: HashMap<u32, u32>,
    /// Optional worker instance for this container
    pub worker: Option<Arc<TensorFusionWorker>>,
}

impl ContainerInfo {
    fn new() -> Self {
        Self {
            container_pid_to_host_pid: HashMap::new(),
            worker: None,
        }
    }
}

/// Worker entry combining worker info with container tracking.
#[derive(Debug, Clone)]
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

/// Worker registry for storing and managing worker information.
pub type WorkerRegistry = Arc<RwLock<HashMap<String, WorkerEntry>>>;

/// PID registry for mapping PIDs to worker entries.
pub type PidRegistry = Arc<RwLock<HashMap<u32, WorkerEntry>>>;

/// Worker manager that handles worker lifecycle based on pod events.
pub struct WorkerManager {
    registry: WorkerRegistry,
    pid_registry: PidRegistry,
    host_pid_probe: Arc<HostPidProbe>,
    command_dispatcher: Arc<CommandDispatcher>,
    hypervisor: Arc<Hypervisor<TensorFusionWorker, WeightedScheduler<TensorFusionWorker>>>,
    limiter_coordinator: Arc<LimiterCoordinator>,
    nvml: Arc<Nvml>,
}

impl WorkerManager {
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

impl WorkerManager {
    /// Create a new worker manager with host PID probe.
    pub fn new(
        host_pid_probe: Arc<HostPidProbe>,
        command_dispatcher: Arc<CommandDispatcher>,
        hypervisor: Arc<Hypervisor<TensorFusionWorker, WeightedScheduler<TensorFusionWorker>>>,
        limiter_coordinator: Arc<LimiterCoordinator>,
        nvml: Arc<Nvml>,
    ) -> Self {
        Self {
            registry: Arc::new(RwLock::new(HashMap::new())),
            pid_registry: Arc::new(RwLock::new(HashMap::new())),
            host_pid_probe,
            command_dispatcher,
            hypervisor,
            limiter_coordinator,
            nvml,
        }
    }

    /// Handle a pod creation event.
    pub async fn handle_pod_created(&self, pod_info: TensorFusionPodInfo) -> Result<()> {
        let worker_key = format!("{}_{}", pod_info.0.namespace, pod_info.0.pod_name);
        info!("Processing pod creation: {worker_key}");

        // Store worker info in registry
        {
            let mut registry = self.registry.write().await;
            registry.insert(worker_key.clone(), WorkerEntry::new(pod_info.0));
            info!("Added worker to registry: {worker_key}");
        }
        Ok(())
    }

    /// Discover the host PID for a container
    pub async fn discover_worker_pid(
        &self,
        pod_name: &str,
        namespace: &str,
        container_name: &str,
        container_pid: u32,
        gpu_observer: Arc<GpuObserver>,
    ) -> Result<PodProcessInfo> {
        self.discover_worker_pid_with_registration(
            pod_name,
            namespace,
            container_name,
            container_pid,
            gpu_observer,
        )
        .await
    }

    /// Discover worker PID and register with limiter coordinator
    pub async fn discover_worker_pid_with_registration(
        &self,
        pod_name: &str,
        namespace: &str,
        container_name: &str,
        container_pid: u32,
        gpu_observer: Arc<GpuObserver>,
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

        let process_info = receiver
            .await
            .map_err(|_| anyhow::anyhow!("PID discovery subscription was cancelled"))?;

        info!(
            "Discovered worker PID: host_pid={}, container_pid={} for container {} in pod {}/{}",
            process_info.host_pid, process_info.container_pid, container_name, namespace, pod_name
        );

        // Associate the discovered worker
        self.associate_discovered_worker(
            pod_name,
            namespace,
            container_name,
            process_info.host_pid,
            process_info.container_pid,
            gpu_observer,
        )
        .await?;

        Ok(process_info)
    }

    /// Associate a discovered worker with the registry and register it
    async fn associate_discovered_worker(
        &self,
        pod_name: &str,
        namespace: &str,
        container_name: &str,
        host_pid: u32,
        container_pid: u32,
        gpu_observer: Arc<GpuObserver>,
    ) -> Result<()> {
        let worker_key = format!("{namespace}_{pod_name}");

        info!("associate_discovered_worker started for {worker_key} host_pid={host_pid}");

        // Create worker and update registry in a limited scope to reduce lock hold time
        info!("About to acquire registry write lock for {worker_key}");
        let (worker, entry_for_pid_registry) = {
            let mut registry = self.registry.write().await;
            info!("Registry write lock acquired for {worker_key}");
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
                    self.command_dispatcher.clone(),
                ));

                // Find or create container entry
                let container_info = entry
                    .containers
                    .entry(container_name.to_string())
                    .or_insert_with(ContainerInfo::new);

                // Update container info
                container_info.worker = Some(worker.clone());
                container_info
                    .container_pid_to_host_pid
                    .insert(container_pid, host_pid);

                // Clone entry for PID registry update (done after releasing main registry lock)
                // TODO: remove this clone
                let entry_clone = entry.clone();

                (worker, entry_clone)
            } else {
                warn!("Attempted to associate PID with non-existent worker: {worker_key}");
                return Err(anyhow::anyhow!("Worker not found in registry"));
            }
        };

        // Add worker to hypervisor
        if !self.hypervisor.process_exists(host_pid) {
            tracing::info!("Adding new worker to hypervisor: {}", worker.name());
            self.hypervisor.add_process(worker.as_ref().clone());
        }

        // Register worker to limiter coordinator
        if let Err(e) = register_worker_to_limiter_coordinator(
            &self.limiter_coordinator,
            &worker,
            container_name,
            container_pid,
            host_pid,
            &self.nvml,
        )
        .await
        {
            tracing::error!("Failed to register worker to limiter coordinator: {}", e);
        } else {
            tracing::info!("Successfully registered worker to limiter coordinator");
        }

        {
            let mut pid_registry = self.pid_registry.write().await;
            pid_registry.insert(host_pid, entry_for_pid_registry);
        }

        info!(
            "Associated worker {worker_key} container {container_name} with host PID {host_pid} and container PID {container_pid}"
        );

        Ok(())
    }

    /// Handle a pod update event.
    pub async fn handle_pod_updated(
        &self,
        pod_name: &str,
        namespace: &str,
        pod_info: TensorFusionPodInfo,
        node_name: Option<String>,
    ) -> Result<()> {
        let worker_key = format!("{namespace}_{pod_name}");
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
    pub async fn handle_pod_deleted(&self, pod_name: &str, namespace: &str) -> Result<()> {
        let worker_key = format!("{namespace}_{pod_name}");
        info!("Processing pod deletion: {worker_key}");

        {
            let mut registry = self.registry.write().await;
            if let Some(entry) = registry.remove(&worker_key) {
                info!("Removed worker from registry: {worker_key}");

                // Call remove callback for all containers with workers
                for (container_name, container_info) in &entry.containers {
                    if let Some(worker) = &container_info.worker {
                        info!("Removing worker for container: {container_name}");

                        // Find the container_pid for this worker's host_pid
                        let host_pid = worker.pid();
                        if let Some((container_pid, _)) = container_info
                            .container_pid_to_host_pid
                            .iter()
                            .find(|(_, &hpid)| hpid == host_pid)
                        {
                            // Remove worker from hypervisor
                            self.hypervisor.remove_process(host_pid);

                            // Unregister worker from limiter coordinator
                            if let Err(e) = unregister_worker_from_limiter_coordinator(
                                &self.limiter_coordinator,
                                pod_name,
                                namespace,
                                container_name,
                                *container_pid,
                            )
                            .await
                            {
                                tracing::error!(
                                    "Failed to unregister worker from limiter coordinator: {}",
                                    e
                                );
                            }
                        } else {
                            warn!(
                                "Could not find container_pid for host_pid {} in container {}",
                                host_pid, container_name
                            );
                        }
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

        let non_existent = entry.get_container("non-existent");
        assert!(non_existent.is_none());
    }

    #[test]
    fn container_info_new() {
        let container_info = ContainerInfo::new();

        assert!(container_info.container_pid_to_host_pid.is_empty());
        assert!(container_info.worker.is_none());
    }

    #[test]
    fn container_info_pid_mapping() {
        let mut container_info = ContainerInfo::new();

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
}
