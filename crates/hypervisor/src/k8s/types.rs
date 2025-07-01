use thiserror::Error;

use crate::k8s::TensorFusionAnnotations;

/// Represents updates to workers based on Kubernetes pod events.
#[derive(Debug, Clone)]
#[allow(clippy::enum_variant_names)]
pub(crate) enum WorkerUpdate {
    /// A new pod was created with tensor-fusion annotations
    PodCreated {
        pod_name: String,
        namespace: String,
        annotations: TensorFusionAnnotations,
        node_name: Option<String>,
    },
    /// An existing pod's annotations were updated
    #[allow(dead_code)]
    PodUpdated {
        pod_name: String,
        namespace: String,
        annotations: TensorFusionAnnotations,
        node_name: Option<String>,
    },
    /// A pod was deleted
    PodDeleted { pod_name: String, namespace: String },
}

/// Errors that can occur during Kubernetes operations.
#[derive(Debug, Error)]
pub(crate) enum KubernetesError {
    #[error("Failed to connect to Kubernetes API: {message}")]
    ConnectionFailed { message: String },
    #[error("Failed to watch pods: {message}")]
    WatchFailed { message: String },
    #[error("Failed to parse annotations: {message}")]
    AnnotationParseError { message: String },
    #[error("Pod not found: {pod_name} in namespace {namespace}")]
    #[allow(dead_code)]
    PodNotFound { pod_name: String, namespace: String },
}
