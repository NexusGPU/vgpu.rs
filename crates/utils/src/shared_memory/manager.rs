use std::collections::HashMap;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use glob;
use shared_memory::ShmemConf;
use spin::RwLock;
use tracing::info;
use tracing::warn;

use super::{handle::SharedMemoryHandle, DeviceConfig, SharedDeviceState};

/// A thread-safe shared memory manager.
pub struct ThreadSafeSharedMemoryManager {
    /// Active shared memory segments: identifier -> Shmem
    active_memories: RwLock<HashMap<String, SharedMemoryHandle>>,
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
    pub fn create_shared_memory(&self, identifier: &str, configs: &[DeviceConfig]) -> Result<()> {
        let mut memories = self.active_memories.write();

        // Check if the segment already exists.
        if memories.contains_key(identifier) {
            return Ok(());
        }
        // Create a new shared memory segment.
        let shmem = SharedMemoryHandle::create(identifier, configs)?;

        // Store the Shmem object and configuration.
        memories.insert(identifier.to_string(), shmem);

        Ok(())
    }

    /// Gets a pointer to the shared memory by its identifier.
    pub fn get_shared_memory(&self, identifier: &str) -> Result<*mut SharedDeviceState> {
        let memories = self.active_memories.read();
        if let Some(shmem) = memories.get(identifier) {
            Ok(shmem.get_ptr())
        } else {
            drop(memories);
            let mut memories = self.active_memories.write();
            let handle = SharedMemoryHandle::open(identifier)?;
            let ptr = handle.get_ptr();
            memories.insert(identifier.to_string(), handle);
            Ok(ptr)
        }
    }

    pub fn add_pid(&self, identifier: &str, pid: usize) -> Result<()> {
        self.with_memory_handle(identifier, |shmem| {
            let state = shmem.get_state();
            state.add_pid(pid);
            Ok(())
        })
    }

    pub fn remove_pid(&self, identifier: &str, pid: usize) -> Result<()> {
        self.with_memory_handle(identifier, |shmem| {
            let state = shmem.get_state();
            state.remove_pid(pid);
            Ok(())
        })
    }

    /// Helper method to safely access shared memory handles with error handling
    fn with_memory_handle<T, F>(&self, identifier: &str, f: F) -> Result<T>
    where
        F: FnOnce(&SharedMemoryHandle) -> Result<T>,
    {
        let memories = self.active_memories.read();
        let shmem = memories
            .get(identifier)
            .context(format!("Shared memory not found: {identifier}"))?;
        f(shmem)
    }

    /// Cleans up a shared memory segment.
    pub fn cleanup(&self, identifier: &str) -> Result<()> {
        let mut memories = self.active_memories.write();

        if let Some(shmem) = memories.remove(identifier) {
            // Drop the Shmem object to release the shared memory.
            drop(shmem);
            info!(identifier = %identifier, "Cleaned up shared memory segment");
        } else {
            warn!(identifier = %identifier, "Attempted to cleanup non-existent shared memory");
        }

        Ok(())
    }

    /// Cleanup orphaned shared memory files from /dev/shm that match our naming pattern.
    /// This should be called at startup to clean up files left by crashed processes.
    pub fn cleanup_orphaned_files(
        &self,
        glob_pattern: &str,
        is_pod_tracking: impl Fn(&str) -> bool,
    ) -> Result<Vec<String>> {
        let mut cleaned_files = Vec::new();
        let file_paths = self.find_shared_memory_files(glob_pattern)?;

        for file_path in file_paths {
            let identifier = self.extract_identifier_from_path(&file_path)?;
            if is_pod_tracking(&identifier) {
                continue;
            }
            if self.is_shared_memory_orphaned(&identifier)? {
                if let Err(e) = self.remove_orphaned_file(&file_path, &identifier) {
                    warn!(
                        "Failed to remove orphaned file {}: {}",
                        file_path.display(),
                        e
                    );
                } else {
                    cleaned_files.push(identifier);
                }
            }
        }

        Ok(cleaned_files)
    }

    /// Find shared memory files matching the glob pattern
    pub fn find_shared_memory_files(&self, glob_pattern: &str) -> Result<Vec<std::path::PathBuf>> {
        let paths = glob::glob(&format!("/dev/shm/{glob_pattern}"))
            .context("Failed to compile glob pattern")?;

        let mut file_paths = Vec::new();
        for path_result in paths {
            let file_path = path_result.context("Failed to read glob path")?;
            if file_path.is_file() {
                file_paths.push(file_path);
            }
        }

        Ok(file_paths)
    }

    /// Extract identifier from file path by removing /dev/shm/ prefix
    pub fn extract_identifier_from_path(&self, file_path: &std::path::Path) -> Result<String> {
        file_path
            .strip_prefix("/dev/shm/")
            .map(|relative_path| relative_path.to_string_lossy().to_string())
            .context(format!(
                "Failed to strip /dev/shm/ prefix from {}",
                file_path.display()
            ))
    }

    /// Check if a shared memory segment is orphaned
    fn is_shared_memory_orphaned(&self, identifier: &str) -> Result<bool> {
        match ShmemConf::new()
            .size(std::mem::size_of::<SharedDeviceState>())
            .os_id(identifier)
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
    fn remove_orphaned_file(&self, file_path: &std::path::Path, identifier: &str) -> Result<()> {
        self.active_memories.write().remove(identifier);
        std::fs::remove_file(file_path)
            .context(format!("Failed to remove file {}", file_path.display()))?;
        info!(
            "Cleaned up orphaned shared memory file: {}",
            file_path.display()
        );
        Ok(())
    }

    /// Checks if a shared memory segment should be cleaned up based on reference count.
    /// Returns true if the segment exists and has zero references.
    pub fn should_cleanup(&self, identifier: &str) -> bool {
        self.with_memory_handle(identifier, |shmem| {
            Ok(shmem.get_state().get_all_pids().is_empty())
        })
        .unwrap_or(false)
    }

    /// Attempt to cleanup shared memory segments with zero reference count.
    pub fn cleanup_unused(&self, is_pod_tracking: impl Fn(&str) -> bool) -> Result<Vec<String>> {
        let mut cleaned_up = Vec::new();
        let identifiers: Vec<String> = {
            let memories = self.active_memories.read();
            memories.keys().cloned().collect()
        };

        for identifier in identifiers {
            if is_pod_tracking(&identifier) {
                continue;
            }
            if self.should_cleanup(&identifier) {
                match self.cleanup(&identifier) {
                    Ok(_) => {
                        cleaned_up.push(identifier.clone());
                        info!(identifier = %identifier, "Cleaned up unused shared memory segment");
                    }
                    Err(e) => {
                        warn!(identifier = %identifier, error = %e, "Failed to cleanup unused shared memory segment");
                    }
                }
            }
        }

        Ok(cleaned_up)
    }

    /// Checks if a shared memory segment exists.
    pub fn contains(&self, identifier: &str) -> bool {
        let memories = self.active_memories.read();
        memories.contains_key(identifier)
    }
}

// Implement SharedMemoryAccess trait directly
impl super::traits::SharedMemoryAccess for ThreadSafeSharedMemoryManager {
    type Error = anyhow::Error;

    fn find_shared_memory_files(&self, glob: &str) -> Result<Vec<std::path::PathBuf>, Self::Error> {
        self.find_shared_memory_files(glob)
    }

    fn extract_identifier_from_path(&self, path: &std::path::Path) -> Result<String, Self::Error> {
        self.extract_identifier_from_path(path)
    }

    fn create_shared_memory(
        &self,
        pod_identifier: &str,
        cfgs: &[super::DeviceConfig],
    ) -> Result<(), Self::Error> {
        self.create_shared_memory(pod_identifier, cfgs)
    }

    fn get_shared_memory(
        &self,
        pod_identifier: &str,
    ) -> Result<*const super::SharedDeviceState, Self::Error> {
        self.get_shared_memory(pod_identifier)
            .map(|ptr| ptr as *const _)
    }

    fn add_pid(&self, pod_identifier: &str, host_pid: usize) -> Result<(), Self::Error> {
        self.add_pid(pod_identifier, host_pid)
    }

    fn remove_pid(&self, pod_identifier: &str, host_pid: usize) -> Result<(), Self::Error> {
        self.remove_pid(pod_identifier, host_pid)
    }

    fn cleanup_orphaned_files<F>(
        &self,
        glob: &str,
        should_remove: F,
    ) -> Result<Vec<String>, Self::Error>
    where
        F: Fn(&str) -> bool,
    {
        self.cleanup_orphaned_files(glob, should_remove)
    }

    fn cleanup_unused<F>(&self, should_keep: F) -> Result<Vec<String>, Self::Error>
    where
        F: Fn(&str) -> bool,
    {
        self.cleanup_unused(should_keep)
    }
}
