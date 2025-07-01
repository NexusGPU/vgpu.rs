//! Kubernetes integration module.
//!
//! This module provides functionality for watching Kubernetes pods and extracting
//! tensor-fusion annotations to manage GPU resource allocation for workers.
//!
//! The main components are:
//! - [`PodWatcher`]: Watches for pod creation/update/deletion events
//! - [`TensorFusionAnnotations`]: Represents parsed tensor-fusion annotations
//! - [`WorkerUpdate`]: Events sent when pod changes affect workers

pub(crate) mod annotations;
pub(crate) mod pod_watcher;
pub(crate) mod types;

pub(crate) use annotations::TensorFusionAnnotations;
pub(crate) use pod_watcher::PodWatcher;
pub(crate) use types::WorkerUpdate;
