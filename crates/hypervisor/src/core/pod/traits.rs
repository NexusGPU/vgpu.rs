//! Traits for pod state management and device operations

use std::path::PathBuf;
use std::sync::Arc;

use super::utilization::DeviceSnapshot;
use utils::shared_memory::{DeviceConfig, PodIdentifier};

/// Trait for accessing pod state information
pub trait PodStateRepository: Send + Sync {
    /// Get list of pod paths using a specific device
    fn get_pods_using_device(&self, device_idx: u32) -> Vec<PodIdentifier>;

    /// Get host PIDs for a specific pod
    fn get_host_pids_for_pod(&self, pod_identifier: &PodIdentifier) -> Option<Vec<u32>>;

    /// Get device configuration for a pod and device index
    fn get_device_config_for_pod(
        &self,
        pod_identifier: &PodIdentifier,
        device_idx: u32,
    ) -> Option<Arc<DeviceConfig>>;

    /// Check if a pod exists in the store
    fn contains_pod(&self, pod_identifier: &PodIdentifier) -> bool;

    /// List all pod identifiers
    fn list_pod_identifiers(&self) -> Vec<PodIdentifier>;

    /// Get pod path for a pod identifier
    fn pod_path(&self, pod_identifier: &PodIdentifier) -> PathBuf;
}

/// Trait for getting device snapshots
pub trait DeviceSnapshotProvider: Send + Sync {
    type Error: std::fmt::Debug + std::fmt::Display + Send + Sync + 'static;

    /// Get a device snapshot for the given device index and last seen timestamp
    fn get_device_snapshot(
        &self,
        device_idx: u32,
        last_seen_ts: u64,
    ) -> Result<DeviceSnapshot, Self::Error>;
}

/// Trait for getting current time
pub trait TimeSource: Send + Sync {
    /// Get current Unix timestamp in seconds
    fn now_unix_secs(&self) -> u64;
}
