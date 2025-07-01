use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use tracing::error;
use tracing::info;

use super::types::PodResourceInfo;
use crate::k8s::annotations::TensorFusionAnnotations;

/// Pod storage for API queries
pub type PodStorage = Arc<RwLock<HashMap<String, PodResourceInfo>>>;

/// Update pod information in storage
pub fn update_pod_storage(
    storage: &PodStorage,
    pod_name: String,
    namespace: String,
    node_name: Option<String>,
    annotations: TensorFusionAnnotations,
) {
    let pod_key = format!("{namespace}/{pod_name}");
    let pod_info = PodResourceInfo::from((pod_name, namespace, node_name, annotations));

    match storage.write() {
        Ok(mut storage_guard) => {
            storage_guard.insert(pod_key.clone(), pod_info);
            info!("Updated pod storage for {pod_key}");
        }
        Err(e) => {
            error!("Failed to acquire write lock for pod storage: {e}");
        }
    }
}

/// Remove pod information from storage
pub fn remove_pod_from_storage(storage: &PodStorage, pod_name: &str, namespace: &str) {
    let pod_key = format!("{namespace}/{pod_name}");

    match storage.write() {
        Ok(mut storage_guard) => {
            if storage_guard.remove(&pod_key).is_some() {
                info!("Removed pod from storage: {pod_key}");
            }
        }
        Err(e) => {
            error!("Failed to acquire write lock for pod storage removal: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;

    use super::*;

    fn create_test_storage() -> PodStorage {
        Arc::new(RwLock::new(HashMap::new()))
    }

    fn create_test_annotations() -> TensorFusionAnnotations {
        TensorFusionAnnotations {
            tflops_request: Some(10.0),
            vram_request: Some(8_000_000_000),
            tflops_limit: Some(20.0),
            vram_limit: Some(16_000_000_000),
            gpu_uuids: Some(vec!["gpu-123".to_string()]),
        }
    }

    #[test]
    fn update_pod_storage_adds_new_pod() {
        // Arrange
        let storage = create_test_storage();
        let pod_name = "test-pod".to_string();
        let namespace = "default".to_string();
        let node_name = Some("worker-1".to_string());
        let annotations = create_test_annotations();

        // Act
        update_pod_storage(
            &storage,
            pod_name.clone(),
            namespace.clone(),
            node_name.clone(),
            annotations,
        );

        // Assert
        let storage_guard = storage.read().expect("should acquire read lock");
        let pod_key = format!("{namespace}/{pod_name}");
        let stored_pod = storage_guard.get(&pod_key).expect("should find stored pod");

        assert_eq!(
            stored_pod.pod_name, pod_name,
            "Pod name should be stored correctly"
        );
        assert_eq!(
            stored_pod.namespace, namespace,
            "Namespace should be stored correctly"
        );
        assert_eq!(
            stored_pod.node_name, node_name,
            "Node name should be stored correctly"
        );
        assert_eq!(
            stored_pod.tflops_request,
            Some(10.0),
            "TFLOPS request should be stored correctly"
        );
        assert_eq!(
            stored_pod.vram_request,
            Some(8_000_000_000),
            "VRAM request should be stored correctly"
        );
    }

    #[test]
    fn update_pod_storage_overwrites_existing_pod() {
        // Arrange
        let storage = create_test_storage();
        let pod_name = "test-pod".to_string();
        let namespace = "default".to_string();
        let node_name = Some("worker-1".to_string());

        // Add initial pod
        let initial_annotations = TensorFusionAnnotations {
            tflops_request: Some(5.0),
            vram_request: Some(4_000_000_000),
            tflops_limit: None,
            vram_limit: None,
            gpu_uuids: None,
        };
        update_pod_storage(
            &storage,
            pod_name.clone(),
            namespace.clone(),
            node_name.clone(),
            initial_annotations,
        );

        // Act - Update with new annotations
        let updated_annotations = create_test_annotations();
        update_pod_storage(
            &storage,
            pod_name.clone(),
            namespace.clone(),
            node_name.clone(),
            updated_annotations,
        );

        // Assert
        let storage_guard = storage.read().expect("should acquire read lock");
        let pod_key = format!("{namespace}/{pod_name}");
        let stored_pod = storage_guard
            .get(&pod_key)
            .expect("should find updated pod");

        assert_eq!(
            stored_pod.tflops_request,
            Some(10.0),
            "TFLOPS request should be updated"
        );
        assert_eq!(
            stored_pod.vram_request,
            Some(8_000_000_000),
            "VRAM request should be updated"
        );
        assert_eq!(
            stored_pod.tflops_limit,
            Some(20.0),
            "TFLOPS limit should be updated"
        );
        assert_eq!(
            stored_pod.vram_limit,
            Some(16_000_000_000),
            "VRAM limit should be updated"
        );
    }

    #[test]
    fn update_pod_storage_handles_different_namespaces() {
        // Arrange
        let storage = create_test_storage();
        let pod_name = "same-pod-name".to_string();
        let namespace1 = "namespace1".to_string();
        let namespace2 = "namespace2".to_string();
        let node_name = Some("worker-1".to_string());
        let annotations = create_test_annotations();

        // Act - Add same pod name in different namespaces
        update_pod_storage(
            &storage,
            pod_name.clone(),
            namespace1.clone(),
            node_name.clone(),
            annotations.clone(),
        );
        update_pod_storage(
            &storage,
            pod_name.clone(),
            namespace2.clone(),
            node_name.clone(),
            annotations,
        );

        // Assert
        let storage_guard = storage.read().expect("should acquire read lock");
        let pod_key1 = format!("{namespace1}/{pod_name}");
        let pod_key2 = format!("{namespace2}/{pod_name}");

        assert!(
            storage_guard.contains_key(&pod_key1),
            "Should store pod in first namespace"
        );
        assert!(
            storage_guard.contains_key(&pod_key2),
            "Should store pod in second namespace"
        );
        assert_eq!(storage_guard.len(), 2, "Should have two separate entries");
    }

    #[test]
    fn remove_pod_from_storage_removes_existing_pod() {
        // Arrange
        let storage = create_test_storage();
        let pod_name = "test-pod";
        let namespace = "default";
        let annotations = create_test_annotations();

        // Add a pod first
        update_pod_storage(
            &storage,
            pod_name.to_string(),
            namespace.to_string(),
            None,
            annotations,
        );

        // Act
        remove_pod_from_storage(&storage, pod_name, namespace);

        // Assert
        let storage_guard = storage.read().expect("should acquire read lock");
        let pod_key = format!("{namespace}/{pod_name}");
        assert!(
            !storage_guard.contains_key(&pod_key),
            "Pod should be removed from storage"
        );
        assert!(
            storage_guard.is_empty(),
            "Storage should be empty after removal"
        );
    }

    #[test]
    fn remove_pod_from_storage_handles_nonexistent_pod() {
        // Arrange
        let storage = create_test_storage();
        let pod_name = "nonexistent-pod";
        let namespace = "default";

        // Act - Try to remove a pod that doesn't exist
        remove_pod_from_storage(&storage, pod_name, namespace);

        // Assert
        let storage_guard = storage.read().expect("should acquire read lock");
        assert!(storage_guard.is_empty(), "Storage should remain empty");
    }

    #[test]
    fn remove_pod_from_storage_only_removes_correct_pod() {
        // Arrange
        let storage = create_test_storage();
        let annotations = create_test_annotations();

        // Add multiple pods
        update_pod_storage(
            &storage,
            "pod1".to_string(),
            "default".to_string(),
            None,
            annotations.clone(),
        );
        update_pod_storage(
            &storage,
            "pod2".to_string(),
            "default".to_string(),
            None,
            annotations.clone(),
        );
        update_pod_storage(
            &storage,
            "pod1".to_string(),
            "other".to_string(),
            None,
            annotations,
        );

        // Act - Remove only one specific pod
        remove_pod_from_storage(&storage, "pod1", "default");

        // Assert
        let storage_guard = storage.read().expect("should acquire read lock");
        assert!(
            !storage_guard.contains_key("default/pod1"),
            "Specific pod should be removed"
        );
        assert!(
            storage_guard.contains_key("default/pod2"),
            "Other pod in same namespace should remain"
        );
        assert!(
            storage_guard.contains_key("other/pod1"),
            "Pod with same name in different namespace should remain"
        );
        assert_eq!(storage_guard.len(), 2, "Should have two remaining pods");
    }

    #[test]
    fn storage_operations_are_thread_safe() {
        // Arrange
        let storage = create_test_storage();
        let storage_clone1 = Arc::clone(&storage);
        let storage_clone2 = Arc::clone(&storage);
        let storage_clone3 = Arc::clone(&storage);

        // Act - Perform concurrent operations
        let handle1 = thread::spawn(move || {
            for i in 0..10 {
                let annotations = create_test_annotations();
                update_pod_storage(
                    &storage_clone1,
                    format!("pod{i}"),
                    "ns1".to_string(),
                    None,
                    annotations,
                );
                thread::sleep(Duration::from_millis(1));
            }
        });

        let handle2 = thread::spawn(move || {
            for i in 0..10 {
                let annotations = create_test_annotations();
                update_pod_storage(
                    &storage_clone2,
                    format!("pod{i}"),
                    "ns2".to_string(),
                    None,
                    annotations,
                );
                thread::sleep(Duration::from_millis(1));
            }
        });

        let handle3 = thread::spawn(move || {
            thread::sleep(Duration::from_millis(5));
            for i in 0..5 {
                remove_pod_from_storage(&storage_clone3, &format!("pod{i}"), "ns1");
                thread::sleep(Duration::from_millis(1));
            }
        });

        // Wait for all threads to complete
        handle1.join().expect("should complete thread 1");
        handle2.join().expect("should complete thread 2");
        handle3.join().expect("should complete thread 3");

        // Assert
        let storage_guard = storage.read().expect("should acquire read lock");

        // Should have 5 pods from ns1 (5 removed) and 10 pods from ns2
        assert_eq!(
            storage_guard.len(),
            15,
            "Should have correct number of pods after concurrent operations"
        );

        // Verify that the remaining ns1 pods are the correct ones
        for i in 5..10 {
            let key = format!("ns1/pod{i}");
            assert!(
                storage_guard.contains_key(&key),
                "Pod {i} should exist in ns1"
            );
        }

        // Verify all ns2 pods exist
        for i in 0..10 {
            let key = format!("ns2/pod{i}");
            assert!(
                storage_guard.contains_key(&key),
                "Pod {i} should exist in ns2"
            );
        }
    }

    #[test]
    fn storage_can_be_read_concurrently() {
        // Arrange
        let storage = create_test_storage();
        let annotations = create_test_annotations();

        // Add some test data
        for i in 0..5 {
            update_pod_storage(
                &storage,
                format!("pod{i}"),
                "default".to_string(),
                None,
                annotations.clone(),
            );
        }

        let storage_clone1 = Arc::clone(&storage);
        let storage_clone2 = Arc::clone(&storage);

        // Act - Perform concurrent reads
        let handle1 = thread::spawn(move || {
            for _ in 0..100 {
                let storage_guard = storage_clone1.read().expect("should acquire read lock");
                assert_eq!(storage_guard.len(), 5, "Should consistently read 5 pods");
                thread::sleep(Duration::from_millis(1));
            }
        });

        let handle2 = thread::spawn(move || {
            for _ in 0..100 {
                let storage_guard = storage_clone2.read().expect("should acquire read lock");
                assert!(
                    storage_guard.contains_key("default/pod2"),
                    "Should consistently find pod2"
                );
                thread::sleep(Duration::from_millis(1));
            }
        });

        // Wait for all threads to complete
        handle1
            .join()
            .expect("should complete concurrent read thread 1");
        handle2
            .join()
            .expect("should complete concurrent read thread 2");

        // Assert - Final verification
        let storage_guard = storage.read().expect("should acquire read lock");
        assert_eq!(
            storage_guard.len(),
            5,
            "Storage should remain consistent after concurrent reads"
        );
    }
}
