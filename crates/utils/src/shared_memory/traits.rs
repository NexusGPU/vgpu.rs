//! Traits for shared memory access and related operations

use std::{
    fmt,
    path::{Path, PathBuf},
};

use super::{DeviceConfig, SharedDeviceState};

/// Trait for shared memory access operations  
pub trait SharedMemoryAccess: Send + Sync {
    type Error: fmt::Debug + fmt::Display + Send + Sync + 'static;

    /// Find shared memory files matching a glob pattern
    fn find_shared_memory_files(&self, glob: &str) -> Result<Vec<PathBuf>, Self::Error>;

    /// Extract identifier from a shared memory file path
    fn extract_identifier_from_path(&self, path: &Path) -> Result<String, Self::Error>;

    /// Create shared memory for a pod with given device configurations
    fn create_shared_memory(
        &self,
        pod_identifier: &str,
        cfgs: &[DeviceConfig],
    ) -> Result<(), Self::Error>;

    /// Get shared memory pointer for a pod
    fn get_shared_memory(
        &self,
        pod_identifier: &str,
    ) -> Result<*const SharedDeviceState, Self::Error>;

    /// Add a PID to shared memory for a pod
    fn add_pid(&self, pod_identifier: &str, host_pid: usize) -> Result<(), Self::Error>;

    /// Remove a PID from shared memory for a pod
    fn remove_pid(&self, pod_identifier: &str, host_pid: usize) -> Result<(), Self::Error>;

    /// Cleanup orphaned shared memory files
    fn cleanup_orphaned_files<F>(
        &self,
        glob: &str,
        should_remove: F,
    ) -> Result<Vec<String>, Self::Error>
    where
        F: Fn(&str) -> bool;

    /// Cleanup unused shared memory segments
    fn cleanup_unused<F>(&self, should_keep: F) -> Result<Vec<String>, Self::Error>
    where
        F: Fn(&str) -> bool;
}
