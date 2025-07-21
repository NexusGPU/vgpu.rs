use std::sync::Arc;
use std::time::Duration;

use poem::handler;
use poem::web::Data;
use poem::web::Query;
use poem::Request;
use serde::Deserialize;
use tokio::time::timeout;
use tracing::info;

use super::types::JwtPayload;
use crate::api::types::WorkerQueryResponse;
use crate::gpu_observer::GpuObserver;
use crate::worker_manager::WorkerManager;
use crate::worker_manager::WorkerRegistry;

/// Query parameters for worker lookup
#[derive(Debug, Deserialize)]
pub struct WorkerQuery {
    pub container_name: String,
    pub container_pid: u32,
}

/// Get pod resource information using JWT token info and container details
#[handler]
pub async fn get_worker_info(
    req: &Request,
    query: Query<WorkerQuery>,
    worker_registry: Data<&WorkerRegistry>,
    worker_manager: Data<&Arc<WorkerManager>>,
    gpu_observer: Data<&Arc<GpuObserver>>,
) -> poem::Result<poem::web::Json<WorkerQueryResponse>> {
    // Extract JWT payload from request extensions
    let jwt_payload = req.extensions().get::<JwtPayload>().ok_or_else(|| {
        poem::Error::from_string(
            "JWT payload not found in request",
            poem::http::StatusCode::UNAUTHORIZED,
        )
    })?;

    let pod_name = &jwt_payload.kubernetes.pod.name;
    let namespace = &jwt_payload.kubernetes.namespace;
    let container_name = &query.container_name;
    let container_pid = query.container_pid;

    info!(
        pod_name = pod_name,
        namespace = namespace,
        container_name = container_name,
        container_pid = container_pid,
        "Querying worker info using JWT and container details"
    );

    let registry = worker_registry.read().await;
    let worker_key = format!("{namespace}_{pod_name}");

    // First, check if the worker exists for this pod
    let Some(worker_entry) = registry.get(&worker_key) else {
        info!(
            pod_name = pod_name,
            namespace = namespace,
            "Worker not found in registry"
        );
        return Ok(poem::web::Json(WorkerQueryResponse {
            success: false,
            data: None,
            message: format!("Worker {pod_name} not found in namespace {namespace}"),
        }));
    };
    info!(pod_name = pod_name, "Worker found in registry");

    // Then check if the container exists
    let Some(container_info) = worker_entry.get_container(container_name) else {
        info!(
            pod_name = pod_name,
            namespace = namespace,
            container_name = container_name,
            "Worker found but container not found"
        );
        return Ok(poem::web::Json(WorkerQueryResponse {
            success: false,
            data: None,
            message: format!(
                "Container {container_name} not found for pod {pod_name} in namespace {namespace}"
            ),
        }));
    };

    // Check if the container has the requested PID
    let host_pid = if let Some(&pid) = container_info.container_pid_to_host_pid.get(&container_pid)
    {
        pid
    } else {
        info!(
            pod_name = pod_name,
            namespace = namespace,
            container_name = container_name,
            container_pid = container_pid,
            "Container PID not found in cache, attempting to discover..."
        );

        // release the registry lock
        drop(registry);

        // Discover worker PID with a timeout
        let discovery_timeout = Duration::from_secs(5);
        match timeout(
            discovery_timeout,
            worker_manager.discover_worker_pid(
                pod_name,
                namespace,
                container_name,
                container_pid,
                gpu_observer.clone(),
            ),
        )
        .await
        {
            Ok(Ok(process_info)) => {
                info!(
                    pod_name = pod_name,
                    namespace = namespace,
                    host_pid = process_info.host_pid,
                    "Successfully discovered worker PID"
                );
                process_info.host_pid
            }
            Ok(Err(e)) => {
                info!(
                    pod_name = pod_name,
                    namespace = namespace,
                    container_pid = container_pid,
                    "Failed to discover PID: {}",
                    e
                );
                return Ok(poem::web::Json(WorkerQueryResponse {
                    success: false,
                    data: None,
                    message: format!(
                        "Failed to discover host PID for container PID {container_pid}: {e}"
                    ),
                }));
            }
            Err(_) => {
                info!(
                    pod_name = pod_name,
                    namespace = namespace,
                    container_pid = container_pid,
                    "PID discovery timed out after {} seconds",
                    discovery_timeout.as_secs()
                );
                return Ok(poem::web::Json(WorkerQueryResponse {
                    success: false,
                    data: None,
                    message: format!(
                        "Discovery for container PID {container_pid} timed out after {} seconds",
                        discovery_timeout.as_secs()
                    ),
                }));
            }
        }
    };

    info!(
        pod_name = pod_name,
        namespace = namespace,
        container_name = container_name,
        container_pid = container_pid,
        host_pid = host_pid,
        "Found worker by JWT pod info and container details"
    );

    // re-acquire the registry lock
    let registry = worker_registry.read().await;
    let worker_entry = registry
        .get(&worker_key)
        .expect("Worker should exist after PID discovery");

    // Update the WorkerInfo with host_pid
    let mut worker_info = worker_entry.info.clone();
    worker_info.host_pid = host_pid;

    Ok(poem::web::Json(WorkerQueryResponse {
        success: true,
        data: Some(worker_info),
        message: format!(
            "Worker found for pod {pod_name} in namespace {namespace}, container {container_name} with container PID {container_pid}, host PID {host_pid}"
        ),
    }))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::collections::HashMap;
    use std::sync::Arc;

    use api_types::*;
    use tokio::sync::RwLock;

    use crate::worker_manager::ContainerInfo;
    use crate::worker_manager::WorkerEntry;

    fn create_test_worker_entry() -> WorkerEntry {
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
            workload_name: Some("test-workload".to_string()),
            labels: BTreeMap::new(),
        };

        WorkerEntry {
            info: worker_info.clone(),
            containers: {
                let mut containers = HashMap::new();
                containers.insert(
                    "container1".to_string(),
                    ContainerInfo {
                        container_pid_to_host_pid: {
                            let mut map = HashMap::new();
                            map.insert(100, 1234);
                            map.insert(101, 1235);
                            map
                        },
                        worker: None,
                    },
                );
                containers.insert(
                    "container2".to_string(),
                    ContainerInfo {
                        container_pid_to_host_pid: {
                            let mut map = HashMap::new();
                            map.insert(200, 2234);
                            map.insert(201, 2235);
                            map
                        },
                        worker: None,
                    },
                );
                containers
            },
        }
    }

    #[tokio::test]
    async fn test_worker_registry_lookup() {
        let mut registry = HashMap::new();
        let worker_key = "test-namespace/test-pod".to_string();
        registry.insert(worker_key.clone(), create_test_worker_entry());

        let worker_registry = Arc::new(RwLock::new(registry));

        // Test successful lookup
        {
            let guard = worker_registry.read().await;
            let entry = guard.get(&worker_key).unwrap();

            let container1 = entry.get_container("container1").unwrap();
            assert_eq!(container1.container_pid_to_host_pid.get(&100), Some(&1234));

            let container2 = entry.get_container("container2").unwrap();
            assert_eq!(container2.container_pid_to_host_pid.get(&200), Some(&2234));
        }

        // Test non-existent container
        {
            let guard = worker_registry.read().await;
            let entry = guard.get(&worker_key).unwrap();

            let non_existent = entry.get_container("non-existent");
            assert!(non_existent.is_none());
        }
    }

    #[test]
    fn container_info_multiple_pids() {
        let worker_info = WorkerInfo {
            pod_name: "test-pod".to_string(),
            namespace: "test-namespace".to_string(),
            node_name: Some("test-node".to_string()),
            containers: Some(vec!["test-container".to_string()]),
            tflops_request: Some(1.0),
            vram_request: Some(1024),
            tflops_limit: Some(2.0),
            vram_limit: Some(2048),
            gpu_uuids: Some(vec!["gpu1".to_string()]),
            qos_level: Some(QosLevel::High),
            host_pid: 0,
            workload_name: Some("test-workload".to_string()),
            labels: BTreeMap::new(),
        };

        let entry = WorkerEntry {
            info: worker_info.clone(),
            containers: {
                let mut containers = HashMap::new();
                containers.insert(
                    "test-container".to_string(),
                    ContainerInfo {
                        container_pid_to_host_pid: {
                            let mut map = HashMap::new();
                            map.insert(100, 1234);
                            map.insert(101, 1235);
                            map.insert(102, 1236);
                            map
                        },
                        worker: None,
                    },
                );
                containers
            },
        };

        let container_info = entry.get_container("test-container").unwrap();
        assert_eq!(container_info.container_pid_to_host_pid.len(), 3);
        assert_eq!(
            container_info.container_pid_to_host_pid.get(&100),
            Some(&1234)
        );
        assert_eq!(
            container_info.container_pid_to_host_pid.get(&101),
            Some(&1235)
        );
        assert_eq!(
            container_info.container_pid_to_host_pid.get(&102),
            Some(&1236)
        );
    }

    #[test]
    fn complete_worker_identification_flow() {
        // Test the complete flow: pod + namespace + container + container_pid
        let entry = create_test_worker_entry();

        // Simulate the lookup flow
        let pod_name = "test-pod";
        let namespace = "test-namespace";
        let container_name = "container1";
        let container_pid = 100u32;

        let worker_key = format!("{namespace}_{pod_name}");
        assert_eq!(worker_key, "test-namespace_test-pod");

        let container_info = entry.get_container(container_name).unwrap();
        let host_pid = container_info
            .container_pid_to_host_pid
            .get(&container_pid)
            .unwrap();
        assert_eq!(*host_pid, 1234);

        // Test different container
        let container2_info = entry.get_container("container2").unwrap();
        let host_pid2 = container2_info.container_pid_to_host_pid.get(&200).unwrap();
        assert_eq!(*host_pid2, 2234);
    }
}
