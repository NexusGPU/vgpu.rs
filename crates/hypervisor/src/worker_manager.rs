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

/// Container information for tracking container-specific details.
#[derive(Debug, Clone)]
pub struct ContainerInfo {
    /// Mapping from container PID to host PID for this container
    pub container_pid_to_host_pid: HashMap<u32, u32>,
    /// Workers (processes) in this container, keyed by host PID
    pub workers: HashMap<u32, Arc<TensorFusionWorker>>,
}

impl ContainerInfo {
    fn new() -> Self {
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
            registry.insert(worker_key.clone(), WorkerEntry::new(pod_info.0.clone()));
            info!("Added worker to registry: {worker_key}");
        }

        // Register pod with limiter coordinator (Pod-level operation)
        if let Some(gpu_uuids) = &pod_info.0.gpu_uuids {
            if !gpu_uuids.is_empty() {
                use crate::worker_registration::create_device_configs_from_worker_info;
                let device_configs =
                    create_device_configs_from_worker_info(&pod_info.0, &self.nvml).await?;
                self.limiter_coordinator
                    .register_pod(&worker_key, device_configs)?;
                info!("Registered pod {} with limiter coordinator", worker_key);
            }
        }

        Ok(())
    }

    /// Initialize a CUDA process: discover PID and register to all components
    pub async fn initialize_process(
        &self,
        pod_name: &str,
        namespace: &str,
        container_name: &str,
        container_pid: u32,
        gpu_observer: Arc<GpuObserver>,
    ) -> Result<u32> {
        // Discover and register the process
        let process_info = self
            .discover_worker_pid_with_registration(
                pod_name,
                namespace,
                container_name,
                container_pid,
                gpu_observer,
            )
            .await?;

        Ok(process_info.host_pid)
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
        let (worker, worker_entry) = {
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
                container_info.add_worker(host_pid, container_pid, worker.clone());

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
            &worker_entry,
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
            pid_registry.insert(host_pid, worker_entry);
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

        // Step 1: Remove from worker registry and collect all host PIDs
        let all_host_pids = {
            let mut registry = self.registry.write().await;
            if let Some(entry) = registry.remove(&worker_key) {
                info!("Removed pod from registry: {worker_key}");

                let mut host_pids = Vec::new();

                // Collect all host PIDs from all containers
                for (container_name, container_info) in &entry.containers {
                    if container_info.has_workers() {
                        info!("Processing container: {container_name}");

                        for (host_pid, _worker) in &container_info.workers {
                            host_pids.push(*host_pid);
                            info!("Collected host PID {} for cleanup", host_pid);
                        }
                    }
                }

                host_pids
            } else {
                warn!("Attempted to remove non-existent pod: {worker_key}");
                return Ok(());
            }
        };

        // Step 2: Remove all workers from hypervisor
        for host_pid in &all_host_pids {
            info!("Removing process {} from hypervisor", host_pid);
            self.hypervisor.remove_process(*host_pid);
        }

        // Step 3: Unregister all processes from limiter coordinator
        for host_pid in &all_host_pids {
            if let Err(e) = self
                .limiter_coordinator
                .unregister_process(&worker_key, *host_pid)
            {
                tracing::error!(
                    "Failed to unregister process {} from limiter coordinator: {}",
                    host_pid,
                    e
                );
            } else {
                info!("Unregistered process {} from limiter coordinator", host_pid);
            }
        }

        // Step 4: Clean up PID registry
        {
            let mut pid_registry = self.pid_registry.write().await;
            for host_pid in &all_host_pids {
                pid_registry.remove(host_pid);
                info!("Removed PID {} from PID registry", host_pid);
            }
        }

        info!(
            "Successfully deleted pod {} with {} processes",
            worker_key,
            all_host_pids.len()
        );

        Ok(())
    }
}
