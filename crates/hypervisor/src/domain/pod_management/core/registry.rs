//! Pod registry for thread-safe pod management with high-performance concurrent access

use std::sync::Arc;
use dashmap::DashMap;

use crate::domain::pod_management::types::{Pod, PodId, Worker};
use super::error::{PodManagementError, Result};

/// High-performance thread-safe pod registry using DashMap for optimal concurrent access
#[derive(Debug, Clone)]
pub struct PodRegistry {
    pods: Arc<DashMap<PodId, Pod>>,
    pid_to_pod: Arc<DashMap<u32, PodId>>, // host_pid -> PodId
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
            pods: Arc::new(DashMap::new()),
            pid_to_pod: Arc::new(DashMap::new()),
        }
    }

    /// Register a new pod
    pub fn register_pod(&self, pod: Pod) -> Result<()> {
        let pod_id = pod.id.clone();
        
        // Register all worker PIDs
        for worker in pod.workers.values() {
            if let Some(host_pid) = worker.process_info.host_pid {
                self.pid_to_pod.insert(host_pid, pod_id.clone());
            }
        }
        
        self.pods.insert(pod_id, pod);
        Ok(())
    }

    /// Unregister a pod and clean up all associated data
    pub fn unregister_pod(&self, pod_id: &PodId) -> Result<Pod> {
        let pod = self.pods.remove(pod_id)
            .ok_or_else(|| PodManagementError::PodNotFound(pod_id.clone()))?
            .1;

        // Clean up PID mappings
        for worker in pod.workers.values() {
            if let Some(host_pid) = worker.process_info.host_pid {
                self.pid_to_pod.remove(&host_pid);
            }
        }

        Ok(pod)
    }

    /// Get a pod by ID (returns a clone for safety)
    pub fn get_pod(&self, pod_id: &PodId) -> Option<Pod> {
        self.pods.get(pod_id).map(|entry| entry.value().clone())
    }

    /// Get pod ID by process PID
    pub fn get_pod_by_pid(&self, host_pid: u32) -> Option<PodId> {
        self.pid_to_pod.get(&host_pid).map(|entry| entry.value().clone())
    }

    /// Add a worker to an existing pod
    pub fn add_worker(&self, pod_id: &PodId, worker: Worker) -> Result<()> {
        let mut pod_ref = self.pods.get_mut(pod_id)
            .ok_or_else(|| PodManagementError::PodNotFound(pod_id.clone()))?;

        // Register PID mapping if available
        if let Some(host_pid) = worker.process_info.host_pid {
            self.pid_to_pod.insert(host_pid, pod_id.clone());
        }

        pod_ref.workers.insert(worker.id.clone(), worker);
        Ok(())
    }

    /// Remove a worker from a pod
    pub fn remove_worker(&self, pod_id: &PodId, worker_id: &crate::domain::pod_management::types::WorkerId) -> Result<Worker> {
        let mut pod_ref = self.pods.get_mut(pod_id)
            .ok_or_else(|| PodManagementError::PodNotFound(pod_id.clone()))?;

        let worker = pod_ref.workers.remove(worker_id)
            .ok_or_else(|| PodManagementError::WorkerNotFound(worker_id.clone()))?;

        // Clean up PID mapping
        if let Some(host_pid) = worker.process_info.host_pid {
            self.pid_to_pod.remove(&host_pid);
        }

        Ok(worker)
    }

    /// List all pods (returns clones for safety)
    pub fn list_pods(&self) -> Vec<Pod> {
        self.pods.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Get registry statistics
    pub fn stats(&self) -> RegistryStats {
        let total_pods = self.pods.len();
        let total_workers: usize = self.pods.iter()
            .map(|entry| entry.value().workers.len())
            .sum();
        let total_pids = self.pid_to_pod.len();

        RegistryStats {
            total_pods,
            total_workers,
            total_pids,
        }
    }

    /// Update pod status
    pub fn update_pod_status(&self, pod_id: &PodId, status: crate::domain::pod_management::types::PodStatus) -> Result<()> {
        let mut pod_ref = self.pods.get_mut(pod_id)
            .ok_or_else(|| PodManagementError::PodNotFound(pod_id.clone()))?;
        pod_ref.status = status;
        Ok(())
    }

    /// Check if a pod exists
    pub fn contains_pod(&self, pod_id: &PodId) -> bool {
        self.pods.contains_key(pod_id)
    }

    /// Get the number of pods
    pub fn pod_count(&self) -> usize {
        self.pods.len()
    }

    /// Get all pod IDs
    pub fn pod_ids(&self) -> Vec<PodId> {
        self.pods.iter().map(|entry| entry.key().clone()).collect()
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total_pods: usize,
    pub total_workers: usize,
    pub total_pids: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::pod_management::types::*;
    use std::collections::HashMap;

    fn create_test_pod(id: &str) -> Pod {
        Pod {
            id: PodId::new(id.to_string()),
            name: format!("test-pod-{}", id),
            namespace: "default".to_string(),
            containers: HashMap::new(),
            workers: HashMap::new(),
            status: PodStatus::Running,
        }
    }

    fn create_test_worker(id: &str, host_pid: Option<u32>) -> Worker {
        Worker {
            id: WorkerId::new(format!("worker-{}", id)),
            container_id: ContainerId::new("test".to_string(), "test".to_string()),
            name: format!("worker-{}", id),
            qos_level: api_types::QosLevel::Guaranteed,
            process_info: ProcessInfo {
                host_pid,
                container_pid: Some(1234),
                start_time: std::time::SystemTime::now(),
            },
            status: WorkerStatus::Running,
        }
    }

    #[test]
    fn test_register_pod() {
        let registry = PodRegistry::new();
        let pod = create_test_pod("test-1");
        
        registry.register_pod(pod.clone()).unwrap();
        
        let retrieved = registry.get_pod(&pod.id).unwrap();
        assert_eq!(retrieved.id, pod.id);
        assert_eq!(retrieved.name, pod.name);
    }

    #[test]
    fn test_unregister_pod() {
        let registry = PodRegistry::new();
        let pod = create_test_pod("test-1");
        let pod_id = pod.id.clone();
        
        registry.register_pod(pod).unwrap();
        assert!(registry.contains_pod(&pod_id));
        
        let removed_pod = registry.unregister_pod(&pod_id).unwrap();
        assert_eq!(removed_pod.id, pod_id);
        assert!(!registry.contains_pod(&pod_id));
    }

    #[test]
    fn test_add_worker() {
        let registry = PodRegistry::new();
        let pod = create_test_pod("test-1");
        let pod_id = pod.id.clone();
        
        registry.register_pod(pod).unwrap();
        
        let worker = create_test_worker("w1", Some(5678));
        registry.add_worker(&pod_id, worker.clone()).unwrap();
        
        let updated_pod = registry.get_pod(&pod_id).unwrap();
        assert!(updated_pod.workers.contains_key(&worker.id));
        
        // Check PID mapping
        assert_eq!(registry.get_pod_by_pid(5678), Some(pod_id));
    }

    #[test]
    fn test_remove_worker() {
        let registry = PodRegistry::new();
        let pod = create_test_pod("test-1");
        let pod_id = pod.id.clone();
        
        registry.register_pod(pod).unwrap();
        
        let worker = create_test_worker("w1", Some(5678));
        let worker_id = worker.id.clone();
        registry.add_worker(&pod_id, worker).unwrap();
        
        let removed_worker = registry.remove_worker(&pod_id, &worker_id).unwrap();
        assert_eq!(removed_worker.id, worker_id);
        
        // Check PID mapping is cleaned up
        assert_eq!(registry.get_pod_by_pid(5678), None);
    }

    #[test]
    fn test_list_pods() {
        let registry = PodRegistry::new();
        
        let pod1 = create_test_pod("test-1");
        let pod2 = create_test_pod("test-2");
        
        registry.register_pod(pod1).unwrap();
        registry.register_pod(pod2).unwrap();
        
        let pods = registry.list_pods();
        assert_eq!(pods.len(), 2);
    }

    #[test]
    fn test_stats() {
        let registry = PodRegistry::new();
        
        let pod = create_test_pod("test-1");
        let pod_id = pod.id.clone();
        registry.register_pod(pod).unwrap();
        
        let worker1 = create_test_worker("w1", Some(1001));
        let worker2 = create_test_worker("w2", Some(1002));
        
        registry.add_worker(&pod_id, worker1).unwrap();
        registry.add_worker(&pod_id, worker2).unwrap();
        
        let stats = registry.stats();
        assert_eq!(stats.total_pods, 1);
        assert_eq!(stats.total_workers, 2);
        assert_eq!(stats.total_pids, 2);
    }

    #[test]
    fn test_update_pod_status() {
        let registry = PodRegistry::new();
        let pod = create_test_pod("test-1");
        let pod_id = pod.id.clone();
        
        registry.register_pod(pod).unwrap();
        
        registry.update_pod_status(&pod_id, PodStatus::Terminating).unwrap();
        
        let updated_pod = registry.get_pod(&pod_id).unwrap();
        assert_eq!(updated_pod.status, PodStatus::Terminating);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;
        use std::sync::Arc;
        
        let registry = Arc::new(PodRegistry::new());
        let mut handles = vec![];
        
        // Spawn multiple threads to test concurrent access
        for i in 0..10 {
            let registry_clone = Arc::clone(&registry);
            let handle = thread::spawn(move || {
                let pod = create_test_pod(&format!("test-{}", i));
                registry_clone.register_pod(pod).unwrap();
                
                let worker = create_test_worker(&format!("w{}", i), Some(2000 + i as u32));
                let pod_id = PodId::new(format!("test-{}", i));
                registry_clone.add_worker(&pod_id, worker).unwrap();
            });
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        let stats = registry.stats();
        assert_eq!(stats.total_pods, 10);
        assert_eq!(stats.total_workers, 10);
        assert_eq!(stats.total_pids, 10);
    }
}