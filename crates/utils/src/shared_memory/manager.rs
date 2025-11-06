use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use glob;
use shared_memory::ShmemConf;
use tokio::sync::RwLock;
use tracing::info;
use tracing::warn;

use crate::keyed_lock::KeyedAsyncLock;
use crate::shared_memory::handle::SHM_PATH_SUFFIX;
use crate::shared_memory::PodIdentifier;

use super::{handle::SharedMemoryHandle, DeviceConfig, SharedDeviceState};

/// Shared memory manager.
pub struct MemoryManager {
    /// Active shared memory segments: path -> Shmem
    active_memories: RwLock<HashMap<PathBuf, SharedMemoryHandle>>,
    /// Per-path locks to serialize creation/opening
    creation_locks: KeyedAsyncLock<PathBuf>,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryManager {
    /// Creates a new shared memory manager.
    pub fn new() -> Self {
        Self {
            active_memories: RwLock::new(HashMap::new()),
            creation_locks: KeyedAsyncLock::<PathBuf>::new(),
        }
    }

    /// Creates a shared memory segment.
    pub async fn create_shared_memory(
        &self,
        path: impl AsRef<Path>,
        configs: &[DeviceConfig],
    ) -> Result<()> {
        let path_buf = path.as_ref().to_path_buf();

        // Fast path: check if already exists
        {
            let memories = self.active_memories.read().await;
            if memories.contains_key(&path_buf) {
                return Ok(());
            }
        }

        // Serialize creation for the same path
        let _guard = self.creation_locks.lock(&path_buf).await;

        // Double-check after acquiring lock
        {
            let memories = self.active_memories.read().await;
            if memories.contains_key(&path_buf) {
                return Ok(());
            }
        }

        // Create shared memory outside the active_memories lock (blocking operation)
        let configs_clone = configs.to_vec();
        let path_clone = path_buf.clone();
        let shmem = tokio::task::spawn_blocking(move || {
            SharedMemoryHandle::create(&path_clone, &configs_clone)
        })
        .await
        .context("Failed to spawn blocking task for shared memory creation")??;

        // Insert into map with write lock
        let mut memories = self.active_memories.write().await;

        // Final check (shouldn't happen due to per-path lock, but safe)
        if memories.contains_key(&path_buf) {
            return Ok(());
        }

        memories.insert(path_buf, shmem);

        Ok(())
    }

    /// Gets a pointer to the shared memory by its identifier.
    pub async fn get_shared_memory(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<*mut SharedDeviceState> {
        let path_buf = path.as_ref().to_path_buf();

        // Fast path: check if already exists
        {
            let memories = self.active_memories.read().await;
            if let Some(shmem) = memories.get(&path_buf) {
                return Ok(shmem.get_ptr());
            }
        }

        // Serialize opening for the same path
        let _guard = self.creation_locks.lock(&path_buf).await;

        // Double-check after acquiring lock
        {
            let memories = self.active_memories.read().await;
            if let Some(shmem) = memories.get(&path_buf) {
                return Ok(shmem.get_ptr());
            }
        }

        // Open shared memory outside the active_memories lock (blocking operation)
        let path_clone = path_buf.clone();
        let (handle, ptr_addr) = tokio::task::spawn_blocking(move || {
            let handle = SharedMemoryHandle::open(&path_clone)?;
            let ptr = handle.get_ptr();
            // Convert pointer to usize to make it Send-safe
            Ok::<_, anyhow::Error>((handle, ptr as usize))
        })
        .await
        .context("Failed to spawn blocking task for shared memory open")??;

        // Insert into map with write lock
        let mut memories = self.active_memories.write().await;

        // Final check (shouldn't happen due to per-path lock, but safe)
        if let Some(existing) = memories.get(&path_buf) {
            return Ok(existing.get_ptr());
        }

        memories.insert(path_buf, handle);

        // Convert back to pointer after all awaits
        Ok(ptr_addr as *mut _)
    }

    pub async fn add_pid(&self, path: impl AsRef<Path>, pid: usize) -> Result<()> {
        self.with_memory_handle(path, |shmem| {
            let state = shmem.get_state();
            state.add_pid(pid);
            Ok(())
        })
        .await
    }

    pub async fn remove_pid(&self, path: impl AsRef<Path>, pid: usize) -> Result<()> {
        self.with_memory_handle(path, |shmem| {
            let state = shmem.get_state();
            state.remove_pid(pid);
            Ok(())
        })
        .await
    }

    /// Helper method to safely access shared memory handles with error handling
    async fn with_memory_handle<T, F>(&self, path: impl AsRef<Path>, f: F) -> Result<T>
    where
        F: FnOnce(&SharedMemoryHandle) -> Result<T>,
    {
        let memories = self.active_memories.read().await;
        let shmem = memories.get(path.as_ref()).context(format!(
            "Shared memory not found: {}",
            path.as_ref().display()
        ))?;
        f(shmem)
    }

    /// remove a shared memory segment.
    pub async fn remove_shared_memory(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut memories = self.active_memories.write().await;
        if let Some(shmem) = memories.remove(path.as_ref()) {
            drop(shmem);
            std::fs::remove_file(path.as_ref().join(SHM_PATH_SUFFIX))
                .context(format!("Failed to remove directory {:?}", path.as_ref()))?;
            info!(path = %path.as_ref().display(), "Removed shared memory segment");
        } else {
            warn!(path = %path.as_ref().display(), "Attempted to remove non-existent shared memory");
        }
        Ok(())
    }

    /// Cleanup orphaned shared memory files that match our naming pattern.
    /// This should be called at startup to clean up files left by crashed processes.
    pub async fn cleanup_orphaned_files(
        &self,
        glob_pattern: &str,
        is_pod_tracking: impl Fn(&PodIdentifier) -> bool + Send,
        base_path: impl AsRef<Path> + Send,
    ) -> Result<Vec<PodIdentifier>> {
        let mut cleaned_pid_ids = Vec::new();
        let file_paths = self.find_shared_memory_files(glob_pattern)?;

        for file_path in file_paths {
            if let Err(e) = super::cleanup_empty_parent_directories(
                file_path.join(SHM_PATH_SUFFIX).as_ref(),
                Some(base_path.as_ref()),
            ) {
                tracing::warn!("Failed to cleanup empty directories: {}", e);
            }

            let shm_file_path = file_path.join(SHM_PATH_SUFFIX);
            let identifier = self.extract_identifier_from_path(&base_path, &shm_file_path)?;
            if is_pod_tracking(&identifier) {
                continue;
            }

            if self.is_shared_memory_orphaned(file_path)? {
                if let Err(e) = self
                    .remove_orphaned_file(&shm_file_path, base_path.as_ref())
                    .await
                {
                    warn!(
                        "Failed to remove orphaned file {}: {}",
                        shm_file_path.display(),
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
        glob::glob(glob_pattern)
            .context("Failed to compile glob pattern")?
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to read glob path")
    }

    /// Check if a shared memory segment is orphaned
    fn is_shared_memory_orphaned(&self, path: impl AsRef<Path>) -> Result<bool> {
        match ShmemConf::new()
            .size(std::mem::size_of::<SharedDeviceState>())
            .use_tmpfs_with_dir(path.as_ref())
            .os_id(SHM_PATH_SUFFIX)
            .open()
            .context("Failed to open shared memory")
        {
            Ok(mut shmem) => {
                shmem.set_owner(false);
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
    async fn remove_orphaned_file(&self, file_path: &Path, base_path: &Path) -> Result<()> {
        self.active_memories.write().await.remove(file_path);
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
    pub async fn should_cleanup(&self, path: impl AsRef<Path>) -> bool {
        self.with_memory_handle(path, |shmem| {
            Ok(shmem.get_state().get_all_pids().is_empty())
        })
        .await
        .unwrap_or(false)
    }
    /// Attempt to cleanup shared memory segments with zero reference count.
    pub async fn cleanup_unused(
        &self,
        is_pod_tracking: impl Fn(&PodIdentifier) -> bool + Send,
        base_path: impl AsRef<Path> + Send,
    ) -> Result<Vec<PodIdentifier>> {
        let mut cleaned_up = Vec::new();
        let paths: Vec<PathBuf> = {
            let memories = self.active_memories.read().await;
            memories.keys().cloned().collect()
        };

        for path in paths {
            let identifier =
                self.extract_identifier_from_path(&base_path, path.join(SHM_PATH_SUFFIX))?;
            if is_pod_tracking(&identifier) {
                continue;
            }
            if self.should_cleanup(&path).await {
                match self.remove_shared_memory(&path).await {
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
        PodIdentifier::from_shm_file_path(&identifier)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse PodIdentifier from path: {identifier}"))
    }

    /// Checks if a shared memory segment exists.
    pub async fn contains(&self, path: impl AsRef<Path>) -> bool {
        let memories = self.active_memories.read().await;
        memories.contains_key(path.as_ref())
    }
}

#[async_trait::async_trait]
impl super::traits::SharedMemoryAccess for MemoryManager {
    type Error = anyhow::Error;

    async fn find_shared_memory_files(&self, glob: &str) -> Result<Vec<PathBuf>, Self::Error> {
        self.find_shared_memory_files(glob)
    }

    async fn create_shared_memory(
        &self,
        pod_path: impl AsRef<Path> + Send,
        cfgs: &[super::DeviceConfig],
    ) -> Result<(), Self::Error> {
        self.create_shared_memory(pod_path.as_ref(), cfgs).await
    }

    async fn remove_shared_memory(&self, path: impl AsRef<Path> + Send) -> Result<(), Self::Error> {
        self.remove_shared_memory(path).await
    }

    async fn get_shared_memory(
        &self,
        pod_path: impl AsRef<Path> + Send,
    ) -> Result<super::traits::SharedMemoryPtr, Self::Error> {
        self.get_shared_memory(pod_path.as_ref())
            .await
            .map(|ptr| super::traits::SharedMemoryPtr::new(ptr as *const _))
    }

    async fn add_pid(
        &self,
        pod_path: impl AsRef<Path> + Send,
        host_pid: usize,
    ) -> Result<(), Self::Error> {
        self.add_pid(pod_path.as_ref(), host_pid).await
    }

    async fn remove_pid(
        &self,
        pod_path: impl AsRef<Path> + Send,
        host_pid: usize,
    ) -> Result<(), Self::Error> {
        self.remove_pid(pod_path.as_ref(), host_pid).await
    }

    async fn cleanup_orphaned_files<F, P>(
        &self,
        glob: &str,
        is_pod_tracking: F,
        base_path: P,
    ) -> Result<Vec<PodIdentifier>, Self::Error>
    where
        F: Fn(&PodIdentifier) -> bool + Send,
        P: AsRef<Path> + Send,
    {
        self.cleanup_orphaned_files(glob, is_pod_tracking, base_path)
            .await
    }

    async fn cleanup_unused<F, P>(
        &self,
        should_keep: F,
        base_path: P,
    ) -> Result<Vec<PodIdentifier>, Self::Error>
    where
        F: Fn(&PodIdentifier) -> bool + Send,
        P: AsRef<Path> + Send,
    {
        self.cleanup_unused(should_keep, base_path).await
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
    use crate::shared_memory::{handle::SHM_PATH_SUFFIX, PodIdentifier};
    use std::path::Path;

    #[test]
    fn extract_identifier_from_path_basic() {
        let manager = MemoryManager::new();
        let base_path = Path::new("/dev/shm");

        // Success case
        let full_path = Path::new("/dev/shm/kube-system/my-pod/shm");
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
        let manager = MemoryManager::new();
        let base_path = Path::new("/dev/shm");

        let original = PodIdentifier::new("test-namespace", "test-pod");
        let full_path = original.to_path(base_path);

        let extracted = manager
            .extract_identifier_from_path(base_path, full_path.join(SHM_PATH_SUFFIX))
            .unwrap();
        assert_eq!(extracted, original);
    }
}
