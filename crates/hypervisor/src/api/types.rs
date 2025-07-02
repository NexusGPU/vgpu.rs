// reexport api-types
pub use api_types::JwtAuthConfig;
pub use api_types::JwtPayload;
pub use api_types::PodQueryResponse;
pub use api_types::PodResourceInfo;

use crate::k8s::annotations::TensorFusionAnnotations;

/// Create PodResourceInfo from TensorFusionAnnotations
///
/// This helper function converts TensorFusionAnnotations to PodResourceInfo
/// while handling the conversion of annotation data to the shared API type.
pub fn pod_resource_info_from_annotations(
    pod_name: String,
    namespace: String,
    node_name: Option<String>,
    annotations: TensorFusionAnnotations,
) -> PodResourceInfo {
    PodResourceInfo {
        pod_name,
        namespace,
        node_name,
        tflops_request: annotations.tflops_request,
        vram_request: annotations.vram_request,
        tflops_limit: annotations.tflops_limit,
        vram_limit: annotations.vram_limit,
        gpu_uuids: annotations.gpu_uuids,
    }
}

#[cfg(test)]
mod tests {
    use api_types::KubernetesInfo;
    use api_types::KubernetesNode;
    use api_types::KubernetesPod;
    use api_types::KubernetesServiceAccount;
    use serde_json::json;

    use super::*;

    #[test]
    fn jwt_payload_serializes_correctly() {
        // Arrange
        let payload = JwtPayload {
            kubernetes: KubernetesInfo {
                namespace: "default".to_string(),
                node: KubernetesNode {
                    name: "worker-1".to_string(),
                    uid: "node-123".to_string(),
                },
                pod: KubernetesPod {
                    name: "test-pod".to_string(),
                    uid: "pod-456".to_string(),
                },
                serviceaccount: KubernetesServiceAccount {
                    name: "default".to_string(),
                    uid: "sa-789".to_string(),
                },
            },
            nbf: 1751311081,
            sub: "test-subject".to_string(),
        };

        // Act
        let serialized =
            serde_json::to_value(&payload).expect("should serialize JWT payload successfully");

        // Assert
        let expected = json!({
            "kubernetes.io": {
                "namespace": "default",
                "node": {
                    "name": "worker-1",
                    "uid": "node-123"
                },
                "pod": {
                    "name": "test-pod",
                    "uid": "pod-456"
                },
                "serviceaccount": {
                    "name": "default",
                    "uid": "sa-789"
                }
            },
            "nbf": 1751311081,
            "sub": "test-subject"
        });

        assert_eq!(
            serialized, expected,
            "Serialized payload should match expected structure"
        );
    }

    #[test]
    fn jwt_payload_deserializes_correctly() {
        // Arrange
        let json_data = json!({
            "kubernetes.io": {
                "namespace": "production",
                "node": {
                    "name": "worker-2",
                    "uid": "node-abc"
                },
                "pod": {
                    "name": "api-pod",
                    "uid": "pod-def"
                },
                "serviceaccount": {
                    "name": "api-service",
                    "uid": "sa-ghi"
                }
            },
            "nbf": 1751311082,
            "sub": "api-user"
        });

        // Act
        let payload: JwtPayload =
            serde_json::from_value(json_data).expect("should deserialize JWT payload successfully");

        // Assert
        assert_eq!(
            payload.kubernetes.namespace, "production",
            "Namespace should be correctly deserialized"
        );
        assert_eq!(
            payload.kubernetes.node.name, "worker-2",
            "Node name should be correctly deserialized"
        );
        assert_eq!(
            payload.kubernetes.pod.name, "api-pod",
            "Pod name should be correctly deserialized"
        );
        assert_eq!(
            payload.kubernetes.pod.uid, "pod-def",
            "Pod UID should be correctly deserialized"
        );
        assert_eq!(
            payload.nbf, 1751311082,
            "NBF should be correctly deserialized"
        );
        assert_eq!(
            payload.sub, "api-user",
            "Subject should be correctly deserialized"
        );
    }

    #[test]
    fn pod_resource_info_from_annotations_conversion() {
        // Arrange
        let pod_name = "test-pod".to_string();
        let namespace = "default".to_string();
        let node_name = Some("worker-1".to_string());
        let annotations = TensorFusionAnnotations {
            tflops_request: Some(10.5),
            vram_request: Some(8_000_000_000),
            tflops_limit: Some(20.0),
            vram_limit: Some(16_000_000_000),
            gpu_uuids: Some(vec!["gpu-123".to_string(), "gpu-456".to_string()]),
        };

        // Act
        let resource_info =
            pod_resource_info_from_annotations(pod_name, namespace, node_name, annotations);

        // Assert
        assert_eq!(
            resource_info.pod_name, "test-pod",
            "Pod name should be set correctly"
        );
        assert_eq!(
            resource_info.namespace, "default",
            "Namespace should be set correctly"
        );
        assert_eq!(
            resource_info.node_name,
            Some("worker-1".to_string()),
            "Node name should be set correctly"
        );
        assert_eq!(
            resource_info.tflops_request,
            Some(10.5),
            "TFLOPS request should be set correctly"
        );
        assert_eq!(
            resource_info.vram_request,
            Some(8_000_000_000),
            "VRAM request should be set correctly"
        );
        assert_eq!(
            resource_info.tflops_limit,
            Some(20.0),
            "TFLOPS limit should be set correctly"
        );
        assert_eq!(
            resource_info.vram_limit,
            Some(16_000_000_000),
            "VRAM limit should be set correctly"
        );
        assert_eq!(
            resource_info.gpu_uuids,
            Some(vec!["gpu-123".to_string(), "gpu-456".to_string()]),
            "GPU UUIDs should be set correctly"
        );
    }

    #[test]
    fn pod_resource_info_with_minimal_annotations() {
        // Arrange
        let pod_name = "minimal-pod".to_string();
        let namespace = "test".to_string();
        let node_name = None;
        let annotations = TensorFusionAnnotations {
            tflops_request: None,
            vram_request: None,
            tflops_limit: None,
            vram_limit: None,
            gpu_uuids: None,
        };

        // Act
        let resource_info =
            pod_resource_info_from_annotations(pod_name, namespace, node_name, annotations);

        // Assert
        assert_eq!(
            resource_info.pod_name, "minimal-pod",
            "Pod name should be set correctly"
        );
        assert_eq!(
            resource_info.namespace, "test",
            "Namespace should be set correctly"
        );
        assert_eq!(resource_info.node_name, None, "Node name should be None");
        assert_eq!(
            resource_info.tflops_request, None,
            "TFLOPS request should be None"
        );
        assert_eq!(
            resource_info.vram_request, None,
            "VRAM request should be None"
        );
        assert_eq!(
            resource_info.tflops_limit, None,
            "TFLOPS limit should be None"
        );
        assert_eq!(resource_info.vram_limit, None, "VRAM limit should be None");
        assert_eq!(resource_info.gpu_uuids, None, "GPU UUIDs should be None");
    }

    #[test]
    fn pod_query_response_serializes_correctly() {
        // Arrange
        let pod_info = PodResourceInfo {
            pod_name: "test-pod".to_string(),
            namespace: "default".to_string(),
            node_name: Some("worker-1".to_string()),
            tflops_request: Some(5.0),
            vram_request: Some(4_000_000_000),
            tflops_limit: Some(10.0),
            vram_limit: Some(8_000_000_000),
            gpu_uuids: Some(vec!["gpu-abc".to_string()]),
        };

        let response = PodQueryResponse {
            success: true,
            data: Some(pod_info),
            message: "Pod found".to_string(),
        };

        // Act
        let serialized = serde_json::to_value(&response)
            .expect("should serialize pod query response successfully");

        // Assert
        assert_eq!(serialized["success"], true, "Success field should be true");
        assert!(
            serialized["data"].is_object(),
            "Data field should contain pod information"
        );
        assert_eq!(
            serialized["message"], "Pod found",
            "Message should be correctly set"
        );
        assert_eq!(
            serialized["data"]["pod_name"], "test-pod",
            "Pod name in data should be correct"
        );
    }

    #[test]
    fn pod_query_response_not_found_serializes_correctly() {
        // Arrange
        let response = PodQueryResponse {
            success: false,
            data: None,
            message: "Pod not found".to_string(),
        };

        // Act
        let serialized = serde_json::to_value(&response)
            .expect("should serialize not found response successfully");

        // Assert
        assert_eq!(
            serialized["success"], false,
            "Success field should be false"
        );
        assert!(
            serialized["data"].is_null(),
            "Data field should be null when pod not found"
        );
        assert_eq!(
            serialized["message"], "Pod not found",
            "Message should indicate pod not found"
        );
    }

    #[test]
    fn jwt_auth_config_can_be_cloned() {
        // Arrange
        let config = JwtAuthConfig {
            public_key: "test-key".to_string(),
        };

        // Act
        let cloned_config = config.clone();

        // Assert
        assert_eq!(
            config.public_key, cloned_config.public_key,
            "Cloned config should have the same public key"
        );
    }
}
