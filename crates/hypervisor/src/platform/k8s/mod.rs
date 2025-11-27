//! Kubernetes integration module.
//!
//! This module provides functionality for watching Kubernetes pods and extracting
//! tensor-fusion annotations to manage GPU resource allocation for workers.
//!
//! The main components are:
//! - [`PodWatcher`]: Watches for pod creation/update/deletion events
//! - [`TensorFusionAnnotations`]: Represents parsed tensor-fusion annotations
//! - [`WorkerUpdate`]: Events sent when pod changes affect workers

use core::error::Error;

pub mod device_plugin;
pub mod pod_info;
pub mod pod_info_cache;

pub use pod_info::TensorFusionPodInfo;
pub use pod_info_cache::PodInfoCache;

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
pub enum KubernetesError {
    #[display("Failed to connect to Kubernetes API: {message}")]
    ConnectionFailed { message: String },
    #[display("Failed to watch pods: {message}")]
    WatchFailed { message: String },
    #[display("Failed to parse annotations: {message}")]
    AnnotationParseError { message: String },
}

impl Error for KubernetesError {}
