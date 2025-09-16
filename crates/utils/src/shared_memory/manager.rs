use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use glob;
use shared_memory::ShmemConf;
use spin::RwLock;
use tracing::info;
use tracing::warn;

use crate::shared_memory::PodIdentifier;

use super::{handle::SharedMemoryHandle, DeviceConfig, SharedDeviceState};

/// A thread-safe shared memory manager.
pub struct ThreadSafeSharedMemoryManager {
    /// Active shared memory segments: path -> Shmem
    active_memories: RwLock<HashMap<PathBuf, SharedMemoryHandle>>,
}

impl Default for ThreadSafeSharedMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadSafeSharedMemoryManager {
    /// Creates a new thread-safe shared memory manager.
    pub fn new() -> Self {
        Self {
            active_memories: RwLock::new(HashMap::new()),
        }
    }

    /// Creates a shared memory segment.
    pub fn create_shared_memory(
        &self,
        path: impl AsRef<Path>,
        configs: &[DeviceConfig],
    ) -> Result<()> {
        let mut memories = self.active_memories.write();
        // Check if the segment already exists.
        if memories.contains_key(path.as_ref()) {
            return Ok(());
        }
        // Create a new shared memory segment.
        let shmem = SharedMemoryHandle::create(&path, configs)?;

        // Store the Shmem object and configuration.
        memories.insert(path.as_ref().to_path_buf(), shmem);

        Ok(())
    }

    /// Gets a pointer to the shared memory by its identifier.
    pub fn get_shared_memory(&self, path: impl AsRef<Path>) -> Result<*mut SharedDeviceState> {
        let memories = self.active_memories.read();
        if let Some(shmem) = memories.get(path.as_ref()) {
            Ok(shmem.get_ptr())
        } else {
            drop(memories);
            let mut memories = self.active_memories.write();
            let handle = SharedMemoryHandle::open(path.as_ref())?;
            let ptr = handle.get_ptr();
            memories.insert(path.as_ref().to_path_buf(), handle);
            Ok(ptr)
        }
    }

    pub fn add_pid(&self, path: impl AsRef<Path>, pid: usize) -> Result<()> {
        self.with_memory_handle(path, |shmem| {
            let state = shmem.get_state();
            state.add_pid(pid);
            Ok(())
        })
    }

    pub fn remove_pid(&self, path: impl AsRef<Path>, pid: usize) -> Result<()> {
        self.with_memory_handle(path, |shmem| {
            let state = shmem.get_state();
            state.remove_pid(pid);
            Ok(())
        })
    }

    /// Helper method to safely access shared memory handles with error handling
    fn with_memory_handle<T, F>(&self, path: impl AsRef<Path>, f: F) -> Result<T>
    where
        F: FnOnce(&SharedMemoryHandle) -> Result<T>,
    {
        let memories = self.active_memories.read();
        let shmem = memories.get(path.as_ref()).context(format!(
            "Shared memory not found: {}",
            path.as_ref().display()
        ))?;
        f(shmem)
    }

    /// Cleans up a shared memory segment.
    pub fn cleanup(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut memories = self.active_memories.write();

        if let Some(shmem) = memories.remove(path.as_ref()) {
            // Drop the Shmem object to release the shared memory.
            drop(shmem);
            info!(path = %path.as_ref().display(), "Cleaned up shared memory segment");
        } else {
            warn!(path = %path.as_ref().display(), "Attempted to cleanup non-existent shared memory");
        }

        Ok(())
    }

    /// Cleanup orphaned shared memory files that match our naming pattern.
    /// This should be called at startup to clean up files left by crashed processes.
    pub fn cleanup_orphaned_files(
        &self,
        glob_pattern: &str,
        is_pod_tracking: impl Fn(&PodIdentifier) -> bool,
        base_path: impl AsRef<Path>,
    ) -> Result<Vec<PodIdentifier>> {
        let mut cleaned_pid_ids = Vec::new();
        let file_paths = self.find_shared_memory_files(glob_pattern)?;

        for file_path in file_paths {
            let identifier = self.extract_identifier_from_path(&base_path, &file_path)?;
            if is_pod_tracking(&identifier) {
                continue;
            }
            if self.is_shared_memory_orphaned(&file_path)? {
                if let Err(e) = self.remove_orphaned_file(&file_path, base_path.as_ref()) {
                    warn!(
                        "Failed to remove orphaned file {}: {}",
                        file_path.display(),
                        e
                    );
                } else {
                    cleaned_pid_ids.push(identifier);
                }
            }
        }

        Ok(cleaned_pid_ids)
    }

    /// Find shared memory files matching the glob pattern
    pub fn find_shared_memory_files(&self, glob_pattern: &str) -> Result<Vec<PathBuf>> {
        let paths = glob::glob(glob_pattern).context("Failed to compile glob pattern")?;

        let mut file_paths = Vec::new();
        for path_result in paths {
            let file_path = path_result.context("Failed to read glob path")?;
            if file_path.is_file() {
                file_paths.push(file_path);
            }
        }

        Ok(file_paths)
    }

    /// Check if a shared memory segment is orphaned
    fn is_shared_memory_orphaned(&self, path: impl AsRef<Path>) -> Result<bool> {
        match ShmemConf::new()
            .size(std::mem::size_of::<SharedDeviceState>())
            .flink(path.as_ref())
            .open()
        {
            Ok(shmem) => {
                let ptr = shmem.as_ptr() as *const SharedDeviceState;
                let is_orphaned = unsafe {
                    let state = &*ptr;
                    // Clean up orphaned locks before accessing shared data
                    state.cleanup_orphaned_locks();
                    let pids = state.get_all_pids();
                    let is_healthy = state.is_healthy(Duration::from_secs(10));
                    // Check if it has active processes and is healthy
                    let all_dead = pids.iter().all(|pid| libc::kill(*pid as i32, 0) != 0);
                    all_dead && !is_healthy
                };
                Ok(is_orphaned)
            }
            Err(_) => {
                // Failed to open as shared memory, but file exists - likely corrupted
                Ok(true)
            }
        }
    }

    /// Remove an orphaned shared memory file
    fn remove_orphaned_file(&self, file_path: &Path, base_path: &Path) -> Result<()> {
        self.active_memories.write().remove(file_path);
        std::fs::remove_file(file_path)
            .context(format!("Failed to remove file {}", file_path.display()))?;

        // Try to clean up empty parent directories (stop at base_path)
        if let Err(e) = super::cleanup_empty_parent_directories(file_path, Some(base_path)) {
            tracing::warn!("Failed to cleanup empty directories: {}", e);
        }

        info!(
            "Cleaned up orphaned shared memory file: {}",
            file_path.display()
        );
        Ok(())
    }

    /// Checks if a shared memory segment should be cleaned up based on reference count.
    /// Returns true if the segment exists and has zero references.
    pub fn should_cleanup(&self, path: impl AsRef<Path>) -> bool {
        self.with_memory_handle(path, |shmem| {
            Ok(shmem.get_state().get_all_pids().is_empty())
        })
        .unwrap_or(false)
    }

    /// Attempt to cleanup shared memory segments with zero reference count.
    pub fn cleanup_unused(
        &self,
        is_pod_tracking: impl Fn(&PodIdentifier) -> bool,
        base_path: impl AsRef<Path>,
    ) -> Result<Vec<PodIdentifier>> {
        let mut cleaned_up = Vec::new();
        let paths: Vec<PathBuf> = {
            let memories = self.active_memories.read();
            memories.keys().cloned().collect()
        };

        for path in paths {
            let identifier = self.extract_identifier_from_path(&base_path, &path)?;
            if is_pod_tracking(&identifier) {
                continue;
            }
            if self.should_cleanup(&path) {
                match self.cleanup(&path) {
                    Ok(_) => {
                        cleaned_up.push(identifier.clone());
                        info!(path = %path.display(), "Cleaned up unused shared memory segment");
                    }
                    Err(e) => {
                        warn!(path = %path.display(), error = %e, "Failed to cleanup unused shared memory segment");
                    }
                }
            }
        }
        Ok(cleaned_up)
    }

    fn extract_identifier_from_path(
        &self,
        base_path: impl AsRef<Path>,
        path: impl AsRef<Path>,
    ) -> Result<PodIdentifier> {
        let relative_path = path
            .as_ref()
            .strip_prefix(base_path.as_ref())
            .with_context(|| format!("Failed to strip prefix from {}", path.as_ref().display()))?;
        let identifier = relative_path.to_string_lossy().to_string();
        PodIdentifier::from_path(&identifier).ok_or_else(|| {
            anyhow::anyhow!("Failed to parse PodIdentifier from path: {}", identifier)
        })
    }

    /// Checks if a shared memory segment exists.
    pub fn contains(&self, path: impl AsRef<Path>) -> bool {
        let memories = self.active_memories.read();
        memories.contains_key(path.as_ref())
    }
}

// Implement SharedMemoryAccess trait directly
impl super::traits::SharedMemoryAccess for ThreadSafeSharedMemoryManager {
    type Error = anyhow::Error;

    fn find_shared_memory_files(&self, glob: &str) -> Result<Vec<PathBuf>, Self::Error> {
        self.find_shared_memory_files(glob)
    }

    fn create_shared_memory(
        &self,
        pod_path: impl AsRef<Path>,
        cfgs: &[super::DeviceConfig],
    ) -> Result<(), Self::Error> {
        self.create_shared_memory(pod_path.as_ref(), cfgs)
    }

    fn get_shared_memory(
        &self,
        pod_path: impl AsRef<Path>,
    ) -> Result<*const super::SharedDeviceState, Self::Error> {
        self.get_shared_memory(pod_path.as_ref())
            .map(|ptr| ptr as *const _)
    }

    fn add_pid(&self, pod_path: impl AsRef<Path>, host_pid: usize) -> Result<(), Self::Error> {
        self.add_pid(pod_path.as_ref(), host_pid)
    }

    fn remove_pid(&self, pod_path: impl AsRef<Path>, host_pid: usize) -> Result<(), Self::Error> {
        self.remove_pid(pod_path.as_ref(), host_pid)
    }

    fn cleanup_orphaned_files<F, P>(
        &self,
        glob: &str,
        should_remove: F,
        base_path: P,
    ) -> Result<Vec<PodIdentifier>, Self::Error>
    where
        F: Fn(&PodIdentifier) -> bool,
        P: AsRef<Path>,
    {
        self.cleanup_orphaned_files(glob, should_remove, base_path)
    }

    fn cleanup_unused<F, P>(
        &self,
        should_keep: F,
        base_path: P,
    ) -> Result<Vec<PodIdentifier>, Self::Error>
    where
        F: Fn(&PodIdentifier) -> bool,
        P: AsRef<Path>,
    {
        self.cleanup_unused(should_keep, base_path)
    }

    fn extract_identifier_from_path(
        &self,
        base_path: impl AsRef<Path>,
        path: impl AsRef<Path>,
    ) -> Result<PodIdentifier, Self::Error> {
        self.extract_identifier_from_path(base_path, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared_memory::PodIdentifier;
    use std::path::Path;

    #[test]
    fn extract_identifier_from_path_basic() {
        let manager = ThreadSafeSharedMemoryManager::new();
        let base_path = Path::new("/dev/shm");

        // Success case
        let full_path = Path::new("/dev/shm/kube-system/my-pod");
        let result = manager
            .extract_identifier_from_path(base_path, full_path)
            .unwrap();
        assert_eq!(result.namespace, "kube-system");
        assert_eq!(result.name, "my-pod");

        // Error: insufficient components
        let full_path = Path::new("/dev/shm/namespace");
        assert!(manager
            .extract_identifier_from_path(base_path, full_path)
            .is_err());
    }

    #[test]
    fn extract_identifier_consistent_with_pod_identifier() {
        let manager = ThreadSafeSharedMemoryManager::new();
        let base_path = Path::new("/dev/shm");

        let original = PodIdentifier::new("test-namespace", "test-pod");
        let full_path = original.to_path(base_path);

        let extracted = manager
            .extract_identifier_from_path(base_path, &full_path)
            .unwrap();
        assert_eq!(extracted, original);
    }
}
