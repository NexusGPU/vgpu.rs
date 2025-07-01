use poem::handler;
use poem::web::Data;
use poem::Request;
use tracing::error;
use tracing::info;

use super::storage::PodStorage;
use super::types::JwtPayload;
use super::types::PodQueryResponse;

/// Core logic for getting pod resource information
async fn get_pod_info_impl(
    jwt_payload: &JwtPayload,
    pod_storage: &PodStorage,
) -> poem::Result<PodQueryResponse> {
    let pod_name = &jwt_payload.kubernetes.pod.name;
    let namespace = &jwt_payload.kubernetes.namespace;
    let pod_key = format!("{namespace}/{pod_name}");

    info!(
        pod_name = pod_name,
        namespace = namespace,
        "Querying pod info"
    );

    match pod_storage.read() {
        Ok(storage_guard) => {
            if let Some(pod_info) = storage_guard.get(&pod_key) {
                info!(pod_name = pod_name, "Pod found in storage");
                Ok(PodQueryResponse {
                    success: true,
                    data: Some(pod_info.clone()),
                    message: "Pod information retrieved successfully".to_string(),
                })
            } else {
                info!(pod_name = pod_name, "Pod not found in storage");
                Ok(PodQueryResponse {
                    success: false,
                    data: None,
                    message: format!("Pod {pod_name} not found in namespace {namespace}"),
                })
            }
        }
        Err(e) => {
            error!("Failed to acquire read lock on pod storage: {e}");
            Err(poem::Error::from_string(
                "Internal server error",
                poem::http::StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
    }
}

/// Get pod resource information
#[handler]
pub async fn get_pod_info(
    req: &Request,
    pod_storage: Data<&PodStorage>,
) -> poem::Result<poem::web::Json<PodQueryResponse>> {
    // Extract JWT payload from request extensions
    let jwt_payload = req.extensions().get::<JwtPayload>().ok_or_else(|| {
        error!("JWT payload not found in request extensions");
        poem::Error::from_string(
            "Authentication information missing",
            poem::http::StatusCode::INTERNAL_SERVER_ERROR,
        )
    })?;

    let response = get_pod_info_impl(jwt_payload, &pod_storage).await?;
    Ok(poem::web::Json(response))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::RwLock;

    use super::super::types::KubernetesInfo;
    use super::super::types::KubernetesNode;
    use super::super::types::KubernetesPod;
    use super::super::types::KubernetesServiceAccount;
    use super::super::types::PodResourceInfo;
    use super::*;

    fn create_test_jwt_payload(pod_name: &str, namespace: &str) -> JwtPayload {
        JwtPayload {
            kubernetes: KubernetesInfo {
                namespace: namespace.to_string(),
                node: KubernetesNode {
                    name: "test-node".to_string(),
                    uid: "node-uuid-123".to_string(),
                },
                pod: KubernetesPod {
                    name: pod_name.to_string(),
                    uid: "pod-uuid-456".to_string(),
                },
                serviceaccount: KubernetesServiceAccount {
                    name: "test-sa".to_string(),
                    uid: "sa-uuid-789".to_string(),
                },
            },
            nbf: 1751311081,
            sub: "test-subject".to_string(),
        }
    }

    fn create_test_pod_info(pod_name: &str, namespace: &str) -> PodResourceInfo {
        PodResourceInfo {
            pod_name: pod_name.to_string(),
            namespace: namespace.to_string(),
            node_name: Some("worker-1".to_string()),
            tflops_request: Some(10.0),
            vram_request: Some(8_000_000_000),
            tflops_limit: Some(20.0),
            vram_limit: Some(16_000_000_000),
            gpu_uuids: Some(vec!["gpu-123".to_string()]),
        }
    }

    async fn test_get_pod_info_with_payload(
        pod_storage: PodStorage,
        jwt_payload: Option<JwtPayload>,
    ) -> poem::Result<PodQueryResponse> {
        if let Some(payload) = jwt_payload {
            get_pod_info_impl(&payload, &pod_storage).await
        } else {
            Err(poem::Error::from_string(
                "Authentication information missing",
                poem::http::StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
    }

    #[tokio::test]
    async fn get_pod_info_returns_pod_when_found() {
        // Arrange
        let mut storage_map = HashMap::new();
        let pod_name = "test-pod";
        let namespace = "default";
        let pod_key = format!("{namespace}/{pod_name}");
        let pod_info = create_test_pod_info(pod_name, namespace);
        storage_map.insert(pod_key, pod_info.clone());

        let pod_storage: PodStorage = Arc::new(RwLock::new(storage_map));
        let jwt_payload = create_test_jwt_payload(pod_name, namespace);

        // Act
        let result = test_get_pod_info_with_payload(pod_storage, Some(jwt_payload)).await;

        // Assert
        let response = result.expect("should return successful response");
        assert!(response.success, "Response should indicate success");
        assert!(response.data.is_some(), "Response should contain pod data");

        let returned_pod = response.data.unwrap();
        assert_eq!(returned_pod.pod_name, pod_name, "Pod name should match");
        assert_eq!(returned_pod.namespace, namespace, "Namespace should match");
        assert_eq!(
            returned_pod.tflops_request,
            Some(10.0),
            "TFLOPS request should match"
        );
        assert!(
            response.message.contains("successfully"),
            "Message should indicate success"
        );
    }

    #[tokio::test]
    async fn get_pod_info_returns_not_found_when_pod_missing() {
        // Arrange
        let pod_storage: PodStorage = Arc::new(RwLock::new(HashMap::new()));
        let jwt_payload = create_test_jwt_payload("missing-pod", "default");

        // Act
        let result = test_get_pod_info_with_payload(pod_storage, Some(jwt_payload)).await;

        // Assert
        let response = result.expect("should return response even when pod not found");
        assert!(!response.success, "Response should indicate failure");
        assert!(
            response.data.is_none(),
            "Response should not contain pod data"
        );
        assert!(
            response.message.contains("not found"),
            "Message should indicate pod not found"
        );
        assert!(
            response.message.contains("missing-pod"),
            "Message should contain pod name"
        );
        assert!(
            response.message.contains("default"),
            "Message should contain namespace"
        );
    }

    #[tokio::test]
    async fn get_pod_info_returns_error_when_jwt_payload_missing() {
        // Arrange
        let pod_storage: PodStorage = Arc::new(RwLock::new(HashMap::new()));

        // Act
        let result = test_get_pod_info_with_payload(pod_storage, None).await;

        // Assert
        let error = result.expect_err("should return error when JWT payload missing");
        assert_eq!(
            error.status(),
            poem::http::StatusCode::INTERNAL_SERVER_ERROR,
            "Should return 500 error"
        );
    }

    #[tokio::test]
    async fn get_pod_info_finds_pod_in_correct_namespace() {
        // Arrange
        let mut storage_map = HashMap::new();
        let pod_name = "same-name";

        // Add pods with same name in different namespaces
        let pod_info_ns1 = create_test_pod_info(pod_name, "namespace1");
        let pod_info_ns2 = create_test_pod_info(pod_name, "namespace2");
        storage_map.insert("namespace1/same-name".to_string(), pod_info_ns1);
        storage_map.insert("namespace2/same-name".to_string(), pod_info_ns2);

        let pod_storage: PodStorage = Arc::new(RwLock::new(storage_map));
        let jwt_payload = create_test_jwt_payload(pod_name, "namespace2");

        // Act
        let result = test_get_pod_info_with_payload(pod_storage, Some(jwt_payload)).await;

        // Assert
        let response = result.expect("should return successful response");
        assert!(response.success, "Response should indicate success");

        let returned_pod = response.data.expect("should contain pod data");
        assert_eq!(
            returned_pod.namespace, "namespace2",
            "Should return pod from correct namespace"
        );
    }

    #[tokio::test]
    async fn get_pod_info_handles_different_pod_configurations() {
        // Arrange
        let mut storage_map = HashMap::new();
        let pod_name = "minimal-pod";
        let namespace = "test";

        // Create pod with minimal configuration
        let minimal_pod = PodResourceInfo {
            pod_name: pod_name.to_string(),
            namespace: namespace.to_string(),
            node_name: None,
            tflops_request: None,
            vram_request: None,
            tflops_limit: None,
            vram_limit: None,
            gpu_uuids: None,
        };

        let pod_key = format!("{namespace}/{pod_name}");
        storage_map.insert(pod_key, minimal_pod);

        let pod_storage: PodStorage = Arc::new(RwLock::new(storage_map));
        let jwt_payload = create_test_jwt_payload(pod_name, namespace);

        // Act
        let result = test_get_pod_info_with_payload(pod_storage, Some(jwt_payload)).await;

        // Assert
        let response = result.expect("should return successful response");
        assert!(response.success, "Response should indicate success");

        let returned_pod = response.data.expect("should contain pod data");
        assert_eq!(returned_pod.pod_name, pod_name, "Pod name should match");
        assert_eq!(returned_pod.namespace, namespace, "Namespace should match");
        assert_eq!(returned_pod.node_name, None, "Node name should be None");
        assert_eq!(
            returned_pod.tflops_request, None,
            "TFLOPS request should be None"
        );
        assert_eq!(returned_pod.gpu_uuids, None, "GPU UUIDs should be None");
    }
}
