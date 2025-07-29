//! Pod registry for thread-safe pod management

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::domain::pod_management::types::{Pod, PodId, Worker};
use super::error::{PodManagementError, Result};

/// Thread-safe pod registry
#[derive(Debug, Clone)]
pub struct PodRegistry {
    pods: Arc<RwLock<HashMap<PodId, Pod>>>,
    pid_to_pod: Arc<RwLock<HashMap<u32, PodId>>>, // host_pid -> PodId
}

impl Default for PodRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PodRegistry {
    /// Create a new pod registry
    pub fn new() -> Self {
        Self {
            pods: Arc::new(RwLock::new(HashMap::new())),
            pid_to_pod: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new pod
    pub async fn register_pod(&self, pod: Pod) -> Result<()> {
        let pod_id = pod.id.clone();
        let mut pods = self.pods.write().await;
        
        if pods.contains_key(&pod_id) {
            return Err(PodManagementError::Other(format!(
                "Pod {} already exists", pod_id
            )));
        }
        
        pods.insert(pod_id, pod);
        Ok(())
    }

    /// Unregister a pod and all its workers
    pub async fn unregister_pod(&self, pod_id: &PodId) -> Result<Pod> {
        let mut pods = self.pods.write().await;
        let mut pid_to_pod = self.pid_to_pod.write().await;
        
        let pod = pods.remove(pod_id)
            .ok_or_else(|| PodManagementError::PodNotFound(pod_id.to_string()))?;
        
        // Remove all PIDs associated with this pod
        for host_pid in pod.all_host_pids() {
            pid_to_pod.remove(&host_pid);
        }
        
        Ok(pod)
    }

    /// Get a pod by ID
    pub async fn get_pod(&self, pod_id: &PodId) -> Option<Pod> {
        let pods = self.pods.read().await;
        pods.get(pod_id).cloned()
    }

    /// Get a pod by worker PID
    pub async fn get_pod_by_pid(&self, host_pid: u32) -> Option<Pod> {
        let pid_to_pod = self.pid_to_pod.read().await;
        if let Some(pod_id) = pid_to_pod.get(&host_pid) {
            let pods = self.pods.read().await;
            pods.get(pod_id).cloned()
        } else {
            None
        }
    }

    /// Add a worker to a pod
    pub async fn add_worker(
        &self,
        pod_id: &PodId,
        container_name: &str,
        host_pid: u32,
        worker: Worker,
    ) -> Result<()> {
        let mut pods = self.pods.write().await;
        let mut pid_to_pod = self.pid_to_pod.write().await;
        
        let pod = pods.get_mut(pod_id)
            .ok_or_else(|| PodManagementError::PodNotFound(pod_id.to_string()))?;
        
        if !pod.add_worker(container_name, host_pid, worker) {
            return Err(PodManagementError::ContainerNotFound(container_name.to_string()));
        }
        
        // Update PID mapping
        pid_to_pod.insert(host_pid, pod_id.clone());
        
        Ok(())
    }

    /// Remove a worker from a pod
    pub async fn remove_worker(
        &self,
        pod_id: &PodId,
        container_name: &str,
        host_pid: u32,
    ) -> Result<Option<Worker>> {
        let mut pods = self.pods.write().await;
        let mut pid_to_pod = self.pid_to_pod.write().await;
        
        let pod = pods.get_mut(pod_id)
            .ok_or_else(|| PodManagementError::PodNotFound(pod_id.to_string()))?;
        
        let worker = pod.remove_worker(container_name, host_pid);
        
        if worker.is_some() {
            // Remove PID mapping
            pid_to_pod.remove(&host_pid);
        }
        
        Ok(worker)
    }

    /// List all pods
    pub async fn list_pods(&self) -> Vec<Pod> {
        let pods = self.pods.read().await;
        pods.values().cloned().collect()
    }

    /// Get statistics
    pub async fn stats(&self) -> RegistryStats {
        let pods = self.pods.read().await;
        let pid_to_pod = self.pid_to_pod.read().await;
        
        let total_pods = pods.len();
        let total_workers = pid_to_pod.len();
        let active_pods = pods.values()
            .filter(|p| !p.is_empty())
            .count();
        
        RegistryStats {
            total_pods,
            active_pods,
            total_workers,
        }
    }

    /// Update pod status (for termination, etc.)
    pub async fn update_pod_status<F>(&self, pod_id: &PodId, update_fn: F) -> Result<()>
    where
        F: FnOnce(&mut Pod),
    {
        let mut pods = self.pods.write().await;
        let pod = pods.get_mut(pod_id)
            .ok_or_else(|| PodManagementError::PodNotFound(pod_id.to_string()))?;
        
        update_fn(pod);
        Ok(())
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total_pods: usize,
    pub active_pods: usize,
    pub total_workers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::pod_management::types::{
        Pod, PodId, Worker, WorkerId, ProcessInfo, DeviceAllocation,
    };
    use crate::domain::process::worker::TensorFusionWorker;
    use api_types::{WorkerInfo, QosLevel};
    use std::sync::Arc;

    fn create_test_worker_info() -> WorkerInfo {
        WorkerInfo {
            namespace: "test-namespace".to_string(),
            pod_name: "test-pod".to_string(),
            containers: Some(vec!["container1".to_string()]),
            qos_level: Some(QosLevel::Medium),
            gpu_uuids: Some(vec!["test-gpu-uuid".to_string()]),
            tflops_limit: None,
            vram_limit: None,
        }
    }

    fn create_test_worker(host_pid: u32, container_pid: u32) -> Worker {
        let worker_instance = Arc::new(TensorFusionWorker::new(
            format!("test-worker-{}", host_pid),
            QosLevel::Medium,
            host_pid,
        ));

        Worker::new(
            host_pid,
            container_pid,
            "container1".to_string(),
            QosLevel::Medium,
            worker_instance,
        )
    }

    fn create_test_pod() -> Pod {
        let worker_info = create_test_worker_info();
        let device_allocation = DeviceAllocation::new(vec![]); // Empty for test
        Pod::new(worker_info, device_allocation)
    }

    #[tokio::test]
    async fn test_register_pod() {
        let registry = PodRegistry::new();
        let pod = create_test_pod();
        let pod_id = pod.id.clone();

        // Register pod should succeed
        assert!(registry.register_pod(pod).await.is_ok());

        // Pod should exist in registry
        assert!(registry.get_pod(&pod_id).await.is_some());

        // Registering same pod again should fail
        let duplicate_pod = create_test_pod();
        assert!(registry.register_pod(duplicate_pod).await.is_err());
    }

    #[tokio::test]
    async fn test_unregister_pod() {
        let registry = PodRegistry::new();
        let pod = create_test_pod();
        let pod_id = pod.id.clone();

        // Register pod first
        registry.register_pod(pod).await.unwrap();

        // Unregister should succeed and return the pod
        let removed_pod = registry.unregister_pod(&pod_id).await.unwrap();
        assert_eq!(removed_pod.id, pod_id);

        // Pod should no longer exist
        assert!(registry.get_pod(&pod_id).await.is_none());

        // Unregistering non-existent pod should fail
        assert!(registry.unregister_pod(&pod_id).await.is_err());
    }

    #[tokio::test]
    async fn test_add_worker() {
        let registry = PodRegistry::new();
        let pod = create_test_pod();
        let pod_id = pod.id.clone();

        // Register pod first
        registry.register_pod(pod).await.unwrap();

        // Add worker should succeed
        let worker = create_test_worker(1234, 5678);
        assert!(registry
            .add_worker(&pod_id, "container1", 1234, worker)
            .await
            .is_ok());

        // Worker should be findable by PID
        let found_pod = registry.get_pod_by_pid(1234).await;
        assert!(found_pod.is_some());
        assert_eq!(found_pod.unwrap().id, pod_id);

        // Adding worker to non-existent container should fail
        let worker2 = create_test_worker(9999, 8888);
        assert!(registry
            .add_worker(&pod_id, "non-existent-container", 9999, worker2)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_remove_worker() {
        let registry = PodRegistry::new();
        let pod = create_test_pod();
        let pod_id = pod.id.clone();

        // Register pod and add worker
        registry.register_pod(pod).await.unwrap();
        let worker = create_test_worker(1234, 5678);
        registry
            .add_worker(&pod_id, "container1", 1234, worker)
            .await
            .unwrap();

        // Remove worker should succeed
        let removed_worker = registry
            .remove_worker(&pod_id, "container1", 1234)
            .await
            .unwrap();
        assert!(removed_worker.is_some());

        // Worker should no longer be findable by PID
        assert!(registry.get_pod_by_pid(1234).await.is_none());

        // Removing non-existent worker should return None
        let not_found = registry
            .remove_worker(&pod_id, "container1", 9999)
            .await
            .unwrap();
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn test_list_pods() {
        let registry = PodRegistry::new();

        // Initially empty
        assert_eq!(registry.list_pods().await.len(), 0);

        // Add pods
        let pod1 = create_test_pod();
        let pod2 = {
            let mut worker_info = create_test_worker_info();
            worker_info.pod_name = "test-pod-2".to_string();
            let device_allocation = DeviceAllocation::new(vec![]);
            Pod::new(worker_info, device_allocation)
        };

        registry.register_pod(pod1).await.unwrap();
        registry.register_pod(pod2).await.unwrap();

        // Should list all pods
        let pods = registry.list_pods().await;
        assert_eq!(pods.len(), 2);
    }

    #[tokio::test]
    async fn test_stats() {
        let registry = PodRegistry::new();

        // Initially empty
        let stats = registry.stats().await;
        assert_eq!(stats.total_pods, 0);
        assert_eq!(stats.active_pods, 0);
        assert_eq!(stats.total_workers, 0);

        // Add pod with worker
        let pod = create_test_pod();
        let pod_id = pod.id.clone();
        registry.register_pod(pod).await.unwrap();

        let worker = create_test_worker(1234, 5678);
        registry
            .add_worker(&pod_id, "container1", 1234, worker)
            .await
            .unwrap();

        // Check updated stats
        let stats = registry.stats().await;
        assert_eq!(stats.total_pods, 1);
        assert_eq!(stats.active_pods, 1);
        assert_eq!(stats.total_workers, 1);
    }

    #[tokio::test]
    async fn test_update_pod_status() {
        let registry = PodRegistry::new();
        let pod = create_test_pod();
        let pod_id = pod.id.clone();

        registry.register_pod(pod).await.unwrap();

        // Update pod status
        registry
            .update_pod_status(&pod_id, |pod| {
                pod.mark_terminating();
            })
            .await
            .unwrap();

        // Verify status was updated
        let updated_pod = registry.get_pod(&pod_id).await.unwrap();
        assert_eq!(updated_pod.status, crate::domain::pod_management::types::PodStatus::Terminating);
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let registry = PodRegistry::new();
        let pod = create_test_pod();
        let pod_id = pod.id.clone();

        registry.register_pod(pod).await.unwrap();

        // Simulate concurrent worker additions
        let mut tasks = Vec::new();
        for i in 0..10 {
            let registry_clone = registry.clone();
            let pod_id_clone = pod_id.clone();
            
            let task = tokio::spawn(async move {
                let worker = create_test_worker(1000 + i, 2000 + i);
                registry_clone
                    .add_worker(&pod_id_clone, "container1", 1000 + i, worker)
                    .await
            });
            tasks.push(task);
        }

        // Wait for all tasks to complete
        for task in tasks {
            assert!(task.await.unwrap().is_ok());
        }

        // Verify all workers were added
        let stats = registry.stats().await;
        assert_eq!(stats.total_workers, 10);
    }
}