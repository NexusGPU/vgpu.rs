//! Simplified pod manager with clear responsibilities

use std::sync::Arc;
use api_types::{QosLevel, WorkerInfo};
use nvml_wrapper::Nvml;
use tokio_util::sync::CancellationToken;

use crate::domain::{
    hypervisor::Hypervisor,
    process::worker::TensorFusionWorker,
    scheduler::weighted::WeightedScheduler,
};
use crate::infrastructure::{
    host_pid_probe::{HostPidProbe, PodProcessInfo},
    limiter_comm::CommandDispatcher,
    gpu_observer::GpuObserver,
};

use super::{
    registry::PodRegistry,
    error::{PodManagementError, Result},
};
use crate::domain::pod_management::{
    types::{Pod, PodId, Worker, DeviceAllocation},
    services::{DeviceService, WorkerService, ResourceMonitor},
};

/// Simplified pod manager with clear separation of concerns
pub struct PodManager {
    // Core registry
    registry: PodRegistry,
    
    // Services
    device_service: Arc<DeviceService>,
    worker_service: Arc<WorkerService>,
    resource_monitor: Arc<ResourceMonitor>,
    
    // External dependencies
    host_pid_probe: Arc<HostPidProbe>,
    command_dispatcher: Arc<CommandDispatcher>,
    hypervisor: Arc<Hypervisor<TensorFusionWorker, WeightedScheduler<TensorFusionWorker>>>,
    nvml: Arc<Nvml>,
}

impl PodManager {
    /// Create a new pod manager
    pub fn new(
        host_pid_probe: Arc<HostPidProbe>,
        command_dispatcher: Arc<CommandDispatcher>,
        hypervisor: Arc<Hypervisor<TensorFusionWorker, WeightedScheduler<TensorFusionWorker>>>,
        nvml: Arc<Nvml>,
        device_service: Arc<DeviceService>,
        worker_service: Arc<WorkerService>,
        resource_monitor: Arc<ResourceMonitor>,
    ) -> Self {
        Self {
            registry: PodRegistry::new(),
            device_service,
            worker_service,
            resource_monitor,
            host_pid_probe,
            command_dispatcher,
            hypervisor,
            nvml,
        }
    }

    /// Handle pod creation event
    pub async fn handle_pod_created(
        &self,
        pod_name: &str,
        namespace: &str,
        worker_info: WorkerInfo,
        node_name: Option<String>,
    ) -> Result<()> {
        tracing::info!(
            pod_name = pod_name,
            namespace = namespace,
            node_name = ?node_name,
            "Handling pod creation"
        );

        // Create device allocation
        let device_allocation = self.device_service.create_allocation(&worker_info).await?;
        
        // Create pod
        let pod = Pod::new(worker_info, device_allocation.clone());
        let pod_id = pod.id.clone();

        // Register pod
        self.registry.register_pod(pod).await?;
        
        // Register device allocation
        self.device_service.register_pod_allocation(&pod_id, &device_allocation)?;

        tracing::info!(pod_id = %pod_id, "Pod created successfully");
        Ok(())
    }

    /// Handle pod update event
    pub async fn handle_pod_updated(
        &self,
        pod_name: &str,
        namespace: &str,
        worker_info: WorkerInfo,
        node_name: Option<String>,
    ) -> Result<()> {
        let pod_id = PodId::new(namespace, pod_name);
        
        tracing::info!(
            pod_id = %pod_id,
            node_name = ?node_name,
            "Handling pod update"
        );

        // For now, treat updates as recreation
        // In the future, we could implement more sophisticated update logic
        if let Some(_existing_pod) = self.registry.get_pod(&pod_id).await {
            self.handle_pod_deleted(pod_name, namespace).await?;
        }
        
        self.handle_pod_created(pod_name, namespace, worker_info, node_name).await
    }

    /// Handle pod deletion event
    pub async fn handle_pod_deleted(&self, pod_name: &str, namespace: &str) -> Result<()> {
        let pod_id = PodId::new(namespace, pod_name);
        
        tracing::info!(pod_id = %pod_id, "Handling pod deletion");

        // Get the pod before deletion
        let pod = self.registry.get_pod(&pod_id).await
            .ok_or_else(|| PodManagementError::PodNotFound(pod_id.to_string()))?;

        // Mark pod as terminating
        self.registry.update_pod_status(&pod_id, |p| p.mark_terminating()).await?;

        // Stop all workers
        for container in pod.containers.values() {
            for (host_pid, worker) in &container.workers {
                if let Err(e) = self.worker_service.stop_worker(&pod_id, *host_pid).await {
                    tracing::warn!(
                        pod_id = %pod_id,
                        host_pid = host_pid,
                        error = %e,
                        "Failed to stop worker during pod deletion"
                    );
                }
            }
        }

        // Unregister device allocation
        if let Err(e) = self.device_service.unregister_pod_allocation(&pod_id) {
            tracing::warn!(
                pod_id = %pod_id,
                error = %e,
                "Failed to unregister device allocation"
            );
        }

        // Remove pod from registry
        self.registry.unregister_pod(&pod_id).await?;

        tracing::info!(pod_id = %pod_id, "Pod deleted successfully");
        Ok(())
    }

    /// Initialize a worker process
    pub async fn initialize_process(
        &self,
        pod_name: &str,
        namespace: &str,
        container_name: &str,
        gpu_observer: Arc<GpuObserver>,
    ) -> Result<u32> {
        let pod_id = PodId::new(namespace, pod_name);
        
        // Get pod
        let pod = self.registry.get_pod(&pod_id).await
            .ok_or_else(|| PodManagementError::PodNotFound(pod_id.to_string()))?;

        // Create worker
        let (host_pid, worker) = self.worker_service.create_worker(
            &pod_id,
            container_name,
            &pod.info,
            gpu_observer,
        ).await?;

        // Add worker to registry
        self.registry.add_worker(&pod_id, container_name, host_pid, worker).await?;

        tracing::info!(
            pod_id = %pod_id,
            container_name = container_name,
            host_pid = host_pid,
            "Process initialized successfully"
        );

        Ok(host_pid)
    }

    /// Discover and register a worker PID
    pub async fn discover_worker_pid_with_registration(
        &self,
        pod_name: &str,
        namespace: &str,
        container_name: &str,
        gpu_observer: Arc<GpuObserver>,
    ) -> Result<PodProcessInfo> {
        let pod_id = PodId::new(namespace, pod_name);
        
        // Get pod
        let pod = self.registry.get_pod(&pod_id).await
            .ok_or_else(|| PodManagementError::PodNotFound(pod_id.to_string()))?;

        // Discover worker
        let (host_pid, worker, process_info) = self.worker_service.discover_worker(
            &pod_id,
            container_name,
            &pod.info,
            gpu_observer,
        ).await?;

        // Add worker to registry
        self.registry.add_worker(&pod_id, container_name, host_pid, worker).await?;

        tracing::info!(
            pod_id = %pod_id,
            container_name = container_name,
            host_pid = host_pid,
            "Worker discovered and registered successfully"
        );

        Ok(process_info)
    }

    /// Get pod by name
    pub async fn find_pod_by_name(&self, namespace: &str, pod_name: &str) -> Option<Pod> {
        let pod_id = PodId::new(namespace, pod_name);
        self.registry.get_pod(&pod_id).await
    }

    /// Get pod by worker PID
    pub async fn find_pod_by_worker_pid(&self, host_pid: u32) -> Option<Pod> {
        self.registry.get_pod_by_pid(host_pid).await
    }

    /// Start resource monitoring for all pods
    pub async fn start_resource_monitor(&self, cancellation_token: CancellationToken) {
        self.resource_monitor.start_monitoring(
            self.registry.clone(),
            cancellation_token,
        ).await;
    }

    /// Get registry statistics
    pub async fn get_stats(&self) -> super::registry::RegistryStats {
        self.registry.stats().await
    }

    /// Get the pod registry (for API access)
    pub fn registry(&self) -> &PodRegistry {
        &self.registry
    }
}