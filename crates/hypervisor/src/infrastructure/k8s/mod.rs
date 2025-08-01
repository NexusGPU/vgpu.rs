//! Kubernetes integration module.
//!
//! This module provides functionality for watching Kubernetes pods and extracting
//! tensor-fusion annotations to manage GPU resource allocation for workers.
//!
//! The main components are:
//! - [`PodWatcher`]: Watches for pod creation/update/deletion events
//! - [`TensorFusionAnnotations`]: Represents parsed tensor-fusion annotations
//! - [`WorkerUpdate`]: Events sent when pod changes affect workers

// pub(crate) mod device_plugin;
pub mod pod_info;
pub mod pod_info_cache;
pub mod types;

pub use pod_info::TensorFusionPodInfo;
pub use pod_info_cache::PodInfoCache;
pub use types::WorkerUpdate;
