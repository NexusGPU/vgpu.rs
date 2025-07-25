//! Pod manager that handles pod and worker lifecycle based on pod events.

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
use utils::shared_memory::handle::SharedMemoryHandle;

use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::HostPidProbe;
use crate::host_pid_probe::PodProcessInfo;
use crate::host_pid_probe::SubscriptionRequest;
use crate::hypervisor::Hypervisor;
use crate::k8s::TensorFusionPodInfo;
use crate::limiter_comm::CommandDispatcher;
use crate::process::worker::TensorFusionWorker;
use crate::process::GpuProcess;
use crate::scheduler::weighted::WeightedScheduler;
use tokio_util::sync::CancellationToken;

use super::coordinator::LimiterCoordinator;
use super::registration::{
    create_device_configs_from_worker_info, register_worker_to_limiter_coordinator,
};
use super::registry::{ContainerInfo, PidToPodRegistry, PodEntry, PodRegistry};
use super::resource_tracker::ProcessResourceTracker;

/// Pod manager that handles pod and worker lifecycle based on pod events.
pub struct PodManager {
    registry: PodRegistry,
    pid_registry: PidToPodRegistry,
    /// Process resource tracking: PID -> ProcessResourceTracker
    process_resources: Arc<RwLock<HashMap<u32, ProcessResourceTracker>>>,
    host_pid_probe: Arc<HostPidProbe>,
    command_dispatcher: Arc<CommandDispatcher>,
    hypervisor: Arc<Hypervisor<TensorFusionWorker, WeightedScheduler<TensorFusionWorker>>>,
    limiter_coordinator: Arc<LimiterCoordinator>,
    nvml: Arc<Nvml>,
}

impl PodManager {
    fn pod_identifier(&self, namespace: &str, pod_name: &str) -> String {
        format!("tf_shm_{namespace}_{pod_name}")
    }

    /// Generate pod identifier for the given pod info.
    /// This method encapsulates the pod identifier generation logic.
    pub fn generate_pod_identifier_for_info(&self, pod_info: &WorkerInfo) -> String {
        self.pod_identifier(&pod_info.namespace, &pod_info.pod_name)
    }

    /// Find a pod entry by namespace and pod name.
    /// This allows other modules to access pod information without needing to call pod_identifier directly.
    pub async fn find_pod_by_name(&self, namespace: &str, pod_name: &str) -> Option<PodEntry> {
        let pod_identifier = self.pod_identifier(namespace, pod_name);
        let registry = self.registry.read().await;
        registry.get(&pod_identifier).cloned()
    }

    /// Find a pod by worker PID.
    pub async fn find_pod_by_worker_pid(&self, pid: u32) -> Option<PodEntry> {
        let pid_registry = self.pid_registry.read().await;
        pid_registry.get(&pid).cloned()
    }

    /// Get the pod registry for API queries.
    pub fn registry(&self) -> &PodRegistry {
        &self.registry
    }
}

impl PodManager {
    /// Create a new pod manager with host PID probe.
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
            process_resources: Arc::new(RwLock::new(HashMap::new())),
            host_pid_probe,
            command_dispatcher,
            hypervisor,
            limiter_coordinator,
            nvml,
        }
    }

    /// Handle a pod creation event.
    pub async fn handle_pod_created(&self, pod_info: TensorFusionPodInfo) -> Result<()> {
        let pod_identifier = self.pod_identifier(&pod_info.0.namespace, &pod_info.0.pod_name);
        info!("Processing pod creation: {pod_identifier}");

        // Store pod info in registry
        {
            let mut registry = self.registry.write().await;
            registry.insert(pod_identifier.clone(), PodEntry::new(pod_info.0.clone()));
            info!("Added pod to registry: {pod_identifier}");
        }

        // Register pod with limiter coordinator (Pod-level operation)
        if let Some(gpu_uuids) = &pod_info.0.gpu_uuids {
            if !gpu_uuids.is_empty() {
                let device_configs =
                    create_device_configs_from_worker_info(&pod_info.0, &self.nvml).await?;
                self.limiter_coordinator
                    .register_pod(&pod_identifier, device_configs)?;
                info!("Registered pod {} with limiter coordinator", pod_identifier);
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
        let pod_identifier = self.pod_identifier(namespace, pod_name);

        info!("associate_discovered_worker started for {pod_identifier} host_pid={host_pid}");

        // Create worker and update registry in a limited scope to reduce lock hold time
        info!("About to acquire registry write lock for {pod_identifier}");
        let (worker, pod_entry) = {
            let mut registry = self.registry.write().await;
            info!("Registry write lock acquired for {pod_identifier}");
            if let Some(entry) = registry.get_mut(&pod_identifier) {
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
                warn!("Attempted to associate PID with non-existent worker: {pod_identifier}");
                return Err(anyhow::anyhow!("Worker not found in registry"));
            }
        };

        // get shared memory handle for this process
        let shared_memory_handle =
            Arc::new(SharedMemoryHandle::open(&pod_identifier).map_err(|e| {
                anyhow::anyhow!("Failed to open shared memory for {}: {}", pod_identifier, e)
            })?);

        // Track process resources
        {
            let mut process_resources = self.process_resources.write().await;
            process_resources.insert(
                host_pid,
                ProcessResourceTracker::new(
                    pod_identifier.clone(),
                    container_name.to_string(),
                    shared_memory_handle,
                ),
            );
            info!("Added process resource tracking for host_pid={}", host_pid);
        }

        // Add worker to hypervisor
        if !self.hypervisor.process_exists(host_pid) {
            tracing::info!("Adding new worker to hypervisor: {}", worker.name());
            self.hypervisor.add_process(worker.as_ref().clone());
        }

        // Register worker to limiter coordinator
        if let Err(e) = register_worker_to_limiter_coordinator(
            &pod_identifier,
            &self.limiter_coordinator,
            &worker,
            container_name,
            container_pid,
            host_pid,
        )
        .await
        {
            tracing::error!("Failed to register worker to limiter coordinator: {}", e);
        } else {
            tracing::info!("Successfully registered worker to limiter coordinator");
        }

        {
            let mut pid_registry = self.pid_registry.write().await;
            pid_registry.insert(host_pid, pod_entry);
        }

        info!(
            "Associated worker {pod_identifier} container {container_name} with host PID {host_pid} and container PID {container_pid}"
        );

        Ok(())
    }

    /// Start the resource monitoring task
    pub fn start_resource_monitor(
        &self,
        interval: Duration,
        cancellation_token: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        let process_resources = self.process_resources.clone();
        let hypervisor = self.hypervisor.clone();
        let limiter_coordinator = self.limiter_coordinator.clone();
        let registry = self.registry.clone();
        let pid_registry = self.pid_registry.clone();

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            info!("Starting resource monitor with interval: {:?}", interval);

            loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        info!("Resource monitor shutdown requested");
                        break;
                    }
                    _ = interval_timer.tick() => {
                        // Continue with monitoring logic
                    }
                }

                // Check for dead processes and clean them up
                if let Err(e) = Self::check_and_cleanup_dead_processes_static(
                    &process_resources,
                    &hypervisor,
                    &limiter_coordinator,
                    &registry,
                    &pid_registry,
                )
                .await
                {
                    tracing::error!("Failed to check and cleanup dead processes: {}", e);
                }

                // Verify reference count consistency
                if let Err(e) =
                    Self::verify_reference_count_consistency_static(&process_resources).await
                {
                    tracing::error!("Failed to verify reference count consistency: {}", e);
                }
            }

            info!("Resource monitor stopped");
        })
    }

    /// Static version of check_and_cleanup_dead_processes for use in monitoring task
    async fn check_and_cleanup_dead_processes_static(
        process_resources: &Arc<RwLock<HashMap<u32, ProcessResourceTracker>>>,
        hypervisor: &Arc<Hypervisor<TensorFusionWorker, WeightedScheduler<TensorFusionWorker>>>,
        limiter_coordinator: &Arc<LimiterCoordinator>,
        registry: &PodRegistry,
        pid_registry: &PidToPodRegistry,
    ) -> Result<Vec<u32>> {
        let mut dead_pids = Vec::new();

        // Get all tracked PIDs
        let tracked_pids: Vec<u32> = {
            let process_resources = process_resources.read().await;
            process_resources.keys().copied().collect()
        };

        // Check each PID for liveness
        for pid in tracked_pids {
            if !std::path::Path::new(&format!("/proc/{pid}")).exists() {
                info!("Detected dead process: {}", pid);
                dead_pids.push(pid);

                // Clean up the dead process
                if let Err(e) = Self::handle_process_exited_static(
                    pid,
                    process_resources,
                    hypervisor,
                    limiter_coordinator,
                    registry,
                    pid_registry,
                )
                .await
                {
                    tracing::error!("Failed to cleanup dead process {}: {}", pid, e);
                }
            }
        }

        if !dead_pids.is_empty() {
            info!(
                "Cleaned up {} dead processes: {:?}",
                dead_pids.len(),
                dead_pids
            );
        }

        Ok(dead_pids)
    }

    /// Static version of handle_process_exited for use in monitoring task  
    async fn handle_process_exited_static(
        host_pid: u32,
        process_resources: &Arc<RwLock<HashMap<u32, ProcessResourceTracker>>>,
        hypervisor: &Arc<Hypervisor<TensorFusionWorker, WeightedScheduler<TensorFusionWorker>>>,
        limiter_coordinator: &Arc<LimiterCoordinator>,
        registry: &PodRegistry,
        pid_registry: &PidToPodRegistry,
    ) -> Result<()> {
        info!("Processing process exit: host_pid={}", host_pid);

        // Step 1: Remove from process resources registry and get resource info
        let process_tracker = {
            let mut process_resources = process_resources.write().await;
            process_resources.remove(&host_pid)
        };

        let Some(process_tracker) = process_tracker else {
            warn!("Attempted to cleanup non-tracked process: {}", host_pid);
            return Ok(());
        };

        info!(
            "Found tracked process: host_pid={}, pod={}, container={}",
            host_pid, process_tracker.pod_identifier, process_tracker.container_name
        );

        // Step 2: Remove from hypervisor
        if hypervisor.process_exists(host_pid) {
            info!("Removing process {} from hypervisor", host_pid);
            hypervisor.remove_process(host_pid);
        }

        // Step 3: Unregister from limiter coordinator
        if let Err(e) =
            limiter_coordinator.unregister_process(&process_tracker.pod_identifier, host_pid)
        {
            tracing::error!(
                "Failed to unregister process {} from limiter coordinator: {}",
                host_pid,
                e
            );
        } else {
            info!("Unregistered process {} from limiter coordinator", host_pid);
        }

        // Step 4: Remove from worker registry (update container info)
        {
            let mut registry = registry.write().await;
            if let Some(pod_entry) = registry.get_mut(&process_tracker.pod_identifier) {
                if let Some(container_info) = pod_entry
                    .containers
                    .get_mut(&process_tracker.container_name)
                {
                    container_info.workers.remove(&host_pid);
                    // Also remove from container_pid_to_host_pid mapping
                    container_info
                        .container_pid_to_host_pid
                        .retain(|_, &mut v| v != host_pid);
                    info!(
                        "Removed process {} from worker entry container {}",
                        host_pid, process_tracker.container_name
                    );
                }
            }
        }

        // Step 5: Remove from PID registry
        {
            let mut pid_registry = pid_registry.write().await;
            pid_registry.remove(&host_pid);
            info!("Removed PID {} from PID registry", host_pid);
        }

        // Step 6: Drop shared memory handle (reference count will automatically decrease)
        drop(process_tracker.shared_memory_handle);
        info!(
            "Dropped shared memory handle for process {}, reference count automatically decreased",
            host_pid
        );

        info!(
            "Successfully cleaned up process {} from pod {} container {}",
            host_pid, process_tracker.pod_identifier, process_tracker.container_name
        );

        Ok(())
    }

    /// Static version of verify_reference_count_consistency for use in monitoring task
    async fn verify_reference_count_consistency_static(
        process_resources: &Arc<RwLock<HashMap<u32, ProcessResourceTracker>>>,
    ) -> Result<()> {
        let process_resources = process_resources.read().await;

        // Group processes by pod_identifier
        let mut pod_process_counts: HashMap<String, u32> = HashMap::new();
        for tracker in process_resources.values() {
            *pod_process_counts
                .entry(tracker.pod_identifier.clone())
                .or_insert(0) += 1;
        }

        // Check each pod's shared memory reference count
        for (pod_identifier, expected_count) in pod_process_counts {
            // Get one shared memory handle for this pod to check reference count
            if let Some(tracker) = process_resources
                .values()
                .find(|t| t.pod_identifier == pod_identifier)
            {
                let actual_count = tracker.shared_memory_handle.get_state().get_ref_count();

                if actual_count != expected_count {
                    warn!(
                        "Reference count mismatch for pod {}: expected={}, actual={}",
                        pod_identifier, expected_count, actual_count
                    );
                    // Could implement corrective action here if needed
                }
            }
        }

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
        let pod_identifier = self.pod_identifier(namespace, pod_name);
        info!("Processing pod update: {pod_identifier}");

        // For now, treat update the same as creation
        // In the future, we might want to handle updates differently
        {
            let mut registry = self.registry.write().await;
            if let Some(entry) = registry.get_mut(&pod_identifier) {
                entry.info.tflops_request = pod_info.0.tflops_request;
                entry.info.tflops_limit = pod_info.0.tflops_limit;
                entry.info.vram_request = pod_info.0.vram_request;
                entry.info.vram_limit = pod_info.0.vram_limit;
                entry.info.node_name = node_name;
                info!("Updated worker in registry: {pod_identifier}");
            } else {
                warn!("Attempted to update non-existent worker: {pod_identifier}");
            }
        }

        Ok(())
    }

    /// Handle a pod deletion event.
    pub async fn handle_pod_deleted(&self, pod_name: &str, namespace: &str) -> Result<()> {
        let pod_identifier = self.pod_identifier(namespace, pod_name);
        info!("Processing pod deletion: {pod_identifier}");

        // Step 1: Remove from worker registry and collect all host PIDs
        let all_host_pids = {
            let mut registry = self.registry.write().await;
            if let Some(entry) = registry.remove(&pod_identifier) {
                info!("Removed pod from registry: {pod_identifier}");

                let mut host_pids = Vec::new();

                // Collect all host PIDs from all containers
                for (container_name, container_info) in &entry.containers {
                    if container_info.has_workers() {
                        info!("Processing container: {container_name}");

                        for host_pid in container_info.workers.keys() {
                            host_pids.push(*host_pid);
                            info!("Collected host PID {} for cleanup", host_pid);
                        }
                    }
                }

                host_pids
            } else {
                warn!("Attempted to remove non-existent pod: {pod_identifier}");
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
                .unregister_process(&pod_identifier, *host_pid)
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

        // Step 4: Unregister pod from limiter coordinator
        self.limiter_coordinator.unregister_pod(&pod_identifier)?;

        // Step 6: Clean up PID registry
        {
            let mut pid_registry = self.pid_registry.write().await;
            for host_pid in &all_host_pids {
                pid_registry.remove(host_pid);
                info!("Removed PID {} from PID registry", host_pid);
            }
        }

        // Step 7: Clean up process resource tracking (this will automatically decrease reference counts)
        {
            let mut process_resources = self.process_resources.write().await;
            for host_pid in &all_host_pids {
                if let Some(tracker) = process_resources.remove(host_pid) {
                    info!("Removed process resource tracking for PID {}, shared memory reference count will decrease", host_pid);
                    // When tracker is dropped, the SharedMemoryHandle's reference count is automatically decreased
                    drop(tracker);
                }
            }
        }

        info!(
            "Successfully deleted pod {} with {} processes",
            pod_identifier,
            all_host_pids.len()
        );

        Ok(())
    }
}
