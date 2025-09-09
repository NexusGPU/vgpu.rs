//! Traits for shared memory access and related operations

use std::{
    fmt,
    path::{Path, PathBuf},
};

use crate::shared_memory::PodIdentifier;

use super::{DeviceConfig, SharedDeviceState};

/// Trait for shared memory access operations  
pub trait SharedMemoryAccess: Send + Sync {
    type Error: fmt::Debug + fmt::Display + Send + Sync + 'static;

    /// Find shared memory files matching a glob pattern
    fn find_shared_memory_files(&self, glob: &str) -> Result<Vec<PathBuf>, Self::Error>;

    /// Extract identifier from a shared memory file path
    fn extract_identifier_from_path(
        &self,
        base_path: impl AsRef<Path>,
        path: impl AsRef<Path>,
    ) -> Result<PodIdentifier, Self::Error>;

    /// Create shared memory for a pod with given device configurations
    fn create_shared_memory(
        &self,
        pod_path: impl AsRef<Path>,
        cfgs: &[DeviceConfig],
    ) -> Result<(), Self::Error>;

    /// Get shared memory pointer for a pod
    fn get_shared_memory(
        &self,
        pod_path: impl AsRef<Path>,
    ) -> Result<*const SharedDeviceState, Self::Error>;

    /// Add a PID to shared memory for a pod
    fn add_pid(&self, pod_path: impl AsRef<Path>, host_pid: usize) -> Result<(), Self::Error>;

    /// Remove a PID from shared memory for a pod
    fn remove_pid(&self, pod_path: impl AsRef<Path>, host_pid: usize) -> Result<(), Self::Error>;

    /// Cleanup orphaned shared memory files
    fn cleanup_orphaned_files<F, P>(
        &self,
        glob: &str,
        should_remove: F,
        base_path: P,
    ) -> Result<Vec<PodIdentifier>, Self::Error>
    where
        F: Fn(&PodIdentifier) -> bool,
        P: AsRef<Path>;

    /// Cleanup unused shared memory segments
    fn cleanup_unused<F, P>(
        &self,
        should_keep: F,
        base_path: P,
    ) -> Result<Vec<PodIdentifier>, Self::Error>
    where
        F: Fn(&PodIdentifier) -> bool,
        P: AsRef<Path>;
}
