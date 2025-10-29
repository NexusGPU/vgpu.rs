//! Traits for shared memory access and related operations

use std::{
    fmt,
    path::{Path, PathBuf},
};

use crate::shared_memory::PodIdentifier;

use super::{DeviceConfig, SharedDeviceState};

/// Trait for shared memory access operations
#[async_trait::async_trait]
pub trait SharedMemoryAccess: Send + Sync {
    type Error: fmt::Debug + fmt::Display + Send + Sync + 'static;

    /// Find shared memory files matching a glob pattern
    async fn find_shared_memory_files(&self, glob: &str) -> Result<Vec<PathBuf>, Self::Error>;

    /// Extract identifier from a shared memory file path
    fn extract_identifier_from_path(
        &self,
        base_path: impl AsRef<Path>,
        path: impl AsRef<Path>,
    ) -> Result<PodIdentifier, Self::Error>;

    /// Create shared memory for a pod with given device configurations
    async fn create_shared_memory(
        &self,
        pod_path: impl AsRef<Path> + Send,
        cfgs: &[DeviceConfig],
    ) -> Result<(), Self::Error>;

    /// Get shared memory pointer for a pod
    async fn get_shared_memory(
        &self,
        pod_path: impl AsRef<Path> + Send,
    ) -> Result<*const SharedDeviceState, Self::Error>;

    /// Add a PID to shared memory for a pod
    async fn add_pid(
        &self,
        pod_path: impl AsRef<Path> + Send,
        host_pid: usize,
    ) -> Result<(), Self::Error>;

    /// Remove a PID from shared memory for a pod
    async fn remove_pid(
        &self,
        pod_path: impl AsRef<Path> + Send,
        host_pid: usize,
    ) -> Result<(), Self::Error>;

    /// Cleanup orphaned shared memory files
    async fn cleanup_orphaned_files<F, P>(
        &self,
        glob: &str,
        should_remove: F,
        base_path: P,
    ) -> Result<Vec<PodIdentifier>, Self::Error>
    where
        F: Fn(&PodIdentifier) -> bool + Send,
        P: AsRef<Path> + Send;

    /// Cleanup unused shared memory segments
    async fn cleanup_unused<F, P>(
        &self,
        should_keep: F,
        base_path: P,
    ) -> Result<Vec<PodIdentifier>, Self::Error>
    where
        F: Fn(&PodIdentifier) -> bool + Send,
        P: AsRef<Path> + Send;
}
