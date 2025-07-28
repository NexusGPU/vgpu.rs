use std::collections::HashMap;

use anyhow::Context;
use anyhow::Result;
use glob;
use shared_memory::ShmemConf;
use spin::RwLock;
use tracing::info;
use tracing::warn;

/// A thread-safe shared memory manager.
pub struct ThreadSafeSharedMemoryManager {
    /// Active shared memory segments: identifier -> Shmem
    active_memories: RwLock<HashMap<String, super::handle::SharedMemoryHandle>>,
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

    /// Creates or gets a shared memory segment.
    pub fn create_or_get_shared_memory(
        &self,
        identifier: &str,
        configs: &[super::DeviceConfig],
    ) -> Result<()> {
        let mut memories = self.active_memories.write();

        // Check if the segment already exists.
        if memories.contains_key(identifier) {
            return Ok(());
        }
        // Create a new shared memory segment.
        let shmem = super::handle::SharedMemoryHandle::create(identifier, configs)?;

        // Store the Shmem object and configuration.
        memories.insert(identifier.to_string(), shmem);

        Ok(())
    }

    /// Gets a pointer to the shared memory by its identifier.
    pub fn get_shared_memory(&self, identifier: &str) -> Result<*mut super::SharedDeviceState> {
        let memories = self.active_memories.read();

        if let Some(shmem) = memories.get(identifier) {
            let ptr = shmem.get_ptr();
            Ok(ptr)
        } else {
            Err(anyhow::anyhow!("Shared memory not found: {}", identifier))
        }
    }

    pub fn add_pid(&self, identifier: &str, pid: usize) {
        let memories = self.active_memories.read();
        if let Some(shmem) = memories.get(identifier) {
            shmem.get_state().add_pid(pid);
        }
    }

    pub fn remove_pid(&self, identifier: &str, pid: usize) {
        let memories = self.active_memories.read();
        if let Some(shmem) = memories.get(identifier) {
            shmem.get_state().remove_pid(pid);
        }
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
    pub fn cleanup_orphaned_files(&self, glob_pattern: &str) -> Result<Vec<String>> {
        let mut cleaned_files = Vec::new();

        // Use glob to find matching files
        let paths = glob::glob(&format!("/dev/shm/{glob_pattern}"))
            .context("Failed to compile glob pattern")?;

        for path_result in paths {
            let file_path = path_result.context("Failed to read glob path")?;

            if !file_path.is_file() {
                continue;
            }

            // Extract identifier by removing /dev/shm/ prefix
            let identifier = if let Ok(relative_path) = file_path.strip_prefix("/dev/shm/") {
                relative_path.to_string_lossy().to_string()
            } else {
                warn!(
                    "Failed to strip /dev/shm/ prefix from {}",
                    file_path.display()
                );
                continue;
            };

            // Try to determine if this file is actually in use by attempting to open it
            let is_orphaned = match ShmemConf::new()
                .size(std::mem::size_of::<super::SharedDeviceState>())
                .os_id(&identifier)
                .open()
            {
                Ok(shmem) => {
                    // Successfully opened, check if it has an active process
                    let ptr = shmem.as_ptr() as *const super::SharedDeviceState;
                    unsafe {
                        let state = &*ptr;
                        // If reference count is 0, it's likely orphaned
                        // If no heartbeat within 1 hour, it's likely orphaned
                        state.get_all_pids().is_empty() && !state.is_healthy(100)
                    }
                }
                Err(_) => {
                    // Failed to open as shared memory, but file exists - likely corrupted
                    true
                }
            };

            if is_orphaned {
                match std::fs::remove_file(&file_path) {
                    Ok(_) => {
                        cleaned_files.push(identifier.clone());
                        info!(
                            "Cleaned up orphaned shared memory file: {}",
                            file_path.display()
                        );
                    }
                    Err(e) => {
                        warn!(
                            "Failed to remove orphaned file {}: {}",
                            file_path.display(),
                            e
                        );
                    }
                }
            }
        }

        Ok(cleaned_files)
    }

    /// Checks if a shared memory segment should be cleaned up based on reference count.
    /// Returns true if the segment exists and has zero references.
    pub fn should_cleanup(&self, identifier: &str) -> bool {
        let memories = self.active_memories.read();
        if let Some(shmem) = memories.get(identifier) {
            let state = shmem.get_state();
            state.get_all_pids().is_empty()
        } else {
            false
        }
    }

    /// Attempt to cleanup shared memory segments with zero reference count.
    pub fn cleanup_unused(&self) -> Result<Vec<String>> {
        let mut cleaned_up = Vec::new();
        let identifiers: Vec<String> = {
            let memories = self.active_memories.read();
            memories.keys().cloned().collect()
        };

        for identifier in identifiers {
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
