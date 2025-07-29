use core::error::Error;

use crate::infrastructure::k8s::TensorFusionPodInfo;

/// Represents updates to workers based on Kubernetes pod events.
#[derive(Debug, Clone)]
#[allow(clippy::enum_variant_names)]
pub enum WorkerUpdate {
    /// A new pod was created with tensor-fusion annotations
    PodCreated { pod_info: TensorFusionPodInfo },
    /// An existing pod's annotations were updated
    #[allow(dead_code)]
    PodUpdated {
        pod_name: String,
        namespace: String,
        pod_info: TensorFusionPodInfo,
        node_name: Option<String>,
    },
    /// A pod was deleted
    PodDeleted { pod_name: String, namespace: String },
}

/// Errors that can occur during Kubernetes operations.
#[derive(Debug, derive_more::Display)]
pub(crate) enum KubernetesError {
    #[display("Failed to connect to Kubernetes API: {message}")]
    ConnectionFailed { message: String },
    #[display("Failed to watch pods: {message}")]
    WatchFailed { message: String },
    #[display("Failed to parse annotations: {message}")]
    AnnotationParseError { message: String },
    #[display("Pod not found: {pod_name} in namespace {namespace}")]
    #[allow(dead_code)]
    PodNotFound { pod_name: String, namespace: String },
}

impl Error for KubernetesError {}
