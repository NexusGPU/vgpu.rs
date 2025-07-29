//! Worker service for managing worker processes

use std::sync::Arc;
use std::time::Duration;
use api_types::{QosLevel, WorkerInfo};
use tokio::time::timeout;

use crate::domain::{
    pod_management::{
        core::error::{PodManagementError, Result},
        types::{Worker, PodId, WorkerStatus},
    },
    hypervisor::Hypervisor,
    process::{worker::TensorFusionWorker, GpuProcess},
    scheduler::weighted::WeightedScheduler,
};
use crate::infrastructure::{
    host_pid_probe::{HostPidProbe, PodProcessInfo, SubscriptionRequest},
    limiter_comm::CommandDispatcher,
    gpu_observer::GpuObserver,
};

/// Service for managing worker processes with comprehensive lifecycle management
#[derive(Debug)]
pub struct WorkerService {
    host_pid_probe: Arc<HostPidProbe>,
    command_dispatcher: Arc<CommandDispatcher>,
    hypervisor: Arc<Hypervisor<TensorFusionWorker, WeightedScheduler<TensorFusionWorker>>>,
    worker_timeout: Duration,
}

impl WorkerService {
    /// Create a new worker service
    pub fn new(
        host_pid_probe: Arc<HostPidProbe>,
        command_dispatcher: Arc<CommandDispatcher>,
        hypervisor: Arc<Hypervisor<TensorFusionWorker, WeightedScheduler<TensorFusionWorker>>>,
    ) -> Self {
        Self {
            host_pid_probe,
            command_dispatcher,
            hypervisor,
            worker_timeout: Duration::from_secs(30), // Default timeout
        }
    }

    /// Create a new worker process
    pub async fn create_worker(
        &self,
        pod_id: &PodId,
        container_name: &str,
        worker_info: &WorkerInfo,
        gpu_observer: Arc<GpuObserver>,
    ) -> Result<(u32, Worker)> {
        tracing::info!(
            pod_id = %pod_id,
            container_name = container_name,
            "Creating new worker"
        );

        // Create tensor fusion worker instance
        let worker_instance = Arc::new(TensorFusionWorker::new(
            format!("{}_{}", pod_id, container_name),
            worker_info.qos_level.unwrap_or(QosLevel::Medium),
            0, // Will be set after process creation
        ));

        // Add worker to hypervisor scheduler
        self.hypervisor.add_process(worker_instance.clone());

        // Subscribe to process events for this container
        let subscription = SubscriptionRequest {
            pod_name: extract_pod_name_from_id(pod_id)?,
            namespace: extract_namespace_from_id(pod_id)?,
            container_name: container_name.to_string(),
        };

        // Wait for process to start (with timeout)
        let process_info = timeout(
            self.worker_timeout,
            self.host_pid_probe.subscribe_and_wait_for_process(subscription),
        )
        .await
        .map_err(|_| PodManagementError::Other("Worker creation timeout".to_string()))?
        .map_err(|e| PodManagementError::Other(format!("Failed to create worker: {}", e)))?;

        // Update worker with actual PID
        let host_pid = process_info.host_pid;
        worker_instance.set_pid(host_pid);

        // Register with GPU observer for monitoring
        gpu_observer
            .register_process(host_pid, worker_instance.clone())
            .await
            .map_err(|e| PodManagementError::GpuError(e.to_string()))?;

        // Create worker wrapper
        let worker = Worker::new(
            host_pid,
            process_info.container_pid,
            container_name.to_string(),
            worker_info.qos_level.unwrap_or(QosLevel::Medium),
            worker_instance,
        );

        tracing::info!(
            pod_id = %pod_id,
            container_name = container_name,
            host_pid = host_pid,
            container_pid = process_info.container_pid,
            "Worker created successfully"
        );

        Ok((host_pid, worker))
    }

    /// Discover an existing worker process
    pub async fn discover_worker(
        &self,
        pod_id: &PodId,
        container_name: &str,
        worker_info: &WorkerInfo,
        gpu_observer: Arc<GpuObserver>,
    ) -> Result<(u32, Worker, PodProcessInfo)> {
        tracing::info!(
            pod_id = %pod_id,
            container_name = container_name,
            "Discovering existing worker"
        );

        // Subscribe to process discovery
        let subscription = SubscriptionRequest {
            pod_name: extract_pod_name_from_id(pod_id)?,
            namespace: extract_namespace_from_id(pod_id)?,
            container_name: container_name.to_string(),
        };

        // Discover existing process
        let process_info = timeout(
            self.worker_timeout,
            self.host_pid_probe.subscribe_and_wait_for_process(subscription),
        )
        .await
        .map_err(|_| PodManagementError::Other("Worker discovery timeout".to_string()))?
        .map_err(|e| PodManagementError::Other(format!("Failed to discover worker: {}", e)))?;

        let host_pid = process_info.host_pid;

        // Create worker instance for discovered process
        let worker_instance = Arc::new(TensorFusionWorker::new(
            format!("{}_{}", pod_id, container_name),
            worker_info.qos_level.unwrap_or(QosLevel::Medium),
            host_pid,
        ));

        // Add to hypervisor scheduler
        self.hypervisor.add_process(worker_instance.clone());

        // Register with GPU observer
        gpu_observer
            .register_process(host_pid, worker_instance.clone())
            .await
            .map_err(|e| PodManagementError::GpuError(e.to_string()))?;

        // Create worker wrapper
        let mut worker = Worker::new(
            host_pid,
            process_info.container_pid,
            container_name.to_string(),
            worker_info.qos_level.unwrap_or(QosLevel::Medium),
            worker_instance,
        );

        // Mark as running since it was discovered
        worker.start();

        tracing::info!(
            pod_id = %pod_id,
            container_name = container_name,
            host_pid = host_pid,
            container_pid = process_info.container_pid,
            "Worker discovered successfully"
        );

        Ok((host_pid, worker, process_info))
    }

    /// Stop a worker process gracefully
    pub async fn stop_worker(&self, pod_id: &PodId, host_pid: u32) -> Result<()> {
        tracing::info!(
            pod_id = %pod_id,
            host_pid = host_pid,
            "Stopping worker"
        );

        // Remove from hypervisor scheduler
        self.hypervisor.remove_process(host_pid);

        // Send termination signal via command dispatcher
        if let Err(e) = self.command_dispatcher.send_stop_signal(host_pid).await {
            tracing::warn!(
                pod_id = %pod_id,
                host_pid = host_pid,
                error = %e,
                "Failed to send stop signal, process may have already terminated"
            );
        }

        // Wait for graceful shutdown with timeout
        let shutdown_timeout = Duration::from_secs(10);
        if let Err(_) = timeout(shutdown_timeout, self.wait_for_process_exit(host_pid)).await {
            tracing::warn!(
                pod_id = %pod_id,
                host_pid = host_pid,
                "Worker did not exit gracefully, may need force termination"
            );
        }

        tracing::info!(
            pod_id = %pod_id,
            host_pid = host_pid,
            "Worker stopped"
        );

        Ok(())
    }

    /// Update worker status based on runtime conditions
    pub async fn update_worker_status(
        &self,
        worker: &mut Worker,
        new_status: WorkerStatus,
    ) -> Result<()> {
        let old_status = worker.status.clone();
        
        match new_status {
            WorkerStatus::Running => worker.start(),
            WorkerStatus::Stopping => worker.stop(),
            WorkerStatus::Stopped => worker.terminate(),
            WorkerStatus::Failed(ref reason) => worker.fail(reason.clone()),
            _ => {}, // Starting status is handled during creation
        }

        tracing::debug!(
            worker_id = ?worker.id,
            old_status = ?old_status,
            new_status = ?new_status,
            "Worker status updated"
        );

        Ok(())
    }

    /// Get worker health status
    pub async fn check_worker_health(&self, worker: &Worker) -> Result<bool> {
        // Check if process is still alive
        let is_alive = self.is_process_alive(worker.id.host_pid).await?;
        
        if !is_alive {
            return Ok(false);
        }

        // Additional health checks could be added here:
        // - GPU utilization within expected ranges
        // - Memory usage not exceeding limits
        // - Response to health check signals
        
        Ok(true)
    }

    /// Set worker timeout duration
    pub fn set_worker_timeout(&mut self, timeout: Duration) {
        self.worker_timeout = timeout;
    }

    // Private helper methods
    
    /// Wait for a process to exit
    async fn wait_for_process_exit(&self, host_pid: u32) -> Result<()> {
        let check_interval = Duration::from_millis(100);
        
        loop {
            if !self.is_process_alive(host_pid).await? {
                break;
            }
            tokio::time::sleep(check_interval).await;
        }
        
        Ok(())
    }

    /// Check if a process is still alive
    async fn is_process_alive(&self, host_pid: u32) -> Result<bool> {
        // Use procfs to check if process exists
        let proc_path = format!("/proc/{}", host_pid);
        Ok(tokio::fs::metadata(proc_path).await.is_ok())
    }
}

impl Default for WorkerService {
    fn default() -> Self {
        // This implementation requires actual dependencies, so we can't provide a meaningful default
        // In practice, this would be constructed via dependency injection
        panic!("WorkerService requires dependencies and cannot be created with default()")
    }
}

// Helper functions

/// Extract pod name from PodId (assumes format: tf_shm_namespace_podname)
fn extract_pod_name_from_id(pod_id: &PodId) -> Result<String> {
    let parts: Vec<&str> = pod_id.as_str().split('_').collect();
    if parts.len() >= 4 && parts[0] == "tf" && parts[1] == "shm" {
        Ok(parts[3..].join("_"))
    } else {
        Err(PodManagementError::InvalidConfiguration(
            format!("Invalid pod ID format: {}", pod_id)
        ))
    }
}

/// Extract namespace from PodId
fn extract_namespace_from_id(pod_id: &PodId) -> Result<String> {
    let parts: Vec<&str> = pod_id.as_str().split('_').collect();
    if parts.len() >= 3 && parts[0] == "tf" && parts[1] == "shm" {
        Ok(parts[2].to_string())
    } else {
        Err(PodManagementError::InvalidConfiguration(
            format!("Invalid pod ID format: {}", pod_id)
        ))
    }
}