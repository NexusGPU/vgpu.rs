//! Traits for pod state management and device operations

use super::{pod_state_store::DeviceConfigRef, utilization::DeviceSnapshot};

/// Trait for accessing pod state information
pub trait PodStateRepository: Send + Sync {
    /// Get list of pod paths using a specific device
    fn get_pods_using_device(&self, device_idx: u32) -> Vec<String>;

    /// Get host PIDs for a specific pod
    fn get_host_pids_for_pod(&self, pod_path: &str) -> Option<Vec<u32>>;

    /// Get device configuration for a pod and device index
    fn get_device_config_for_pod(
        &self,
        pod_path: &str,
        device_idx: u32,
    ) -> Option<DeviceConfigRef<'_>>;

    /// Check if a pod exists in the store
    fn contains_pod(&self, pod_path: &str) -> bool;

    /// List all pod paths
    fn list_pod_identifiers(&self) -> Vec<String>;
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
