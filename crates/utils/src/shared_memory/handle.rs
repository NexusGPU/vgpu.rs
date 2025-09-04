use std::cell::RefCell;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use shared_memory::Mode;
use shared_memory::Shmem;
use shared_memory::ShmemConf;
use shared_memory::ShmemError;
use tracing::info;

use super::{DeviceConfig, SharedDeviceState};

/// Safely access shared memory, automatically handling the segment's lifecycle.
pub struct SharedMemoryHandle {
    shmem: RefCell<Shmem>,
    ptr: *mut SharedDeviceState,
}

impl SharedMemoryHandle {
    /// Creates a mock SharedMemoryHandle with predefined test data.
    /// This function is useful for testing without requiring actual shared memory.
    pub fn mock(shm_name: String, gpu_idx_uuids: Vec<(usize, String)>) -> Self {
        // Create mock configs for testing
        let mock_configs: Vec<_> = gpu_idx_uuids
            .iter()
            .map(|(idx, uuid)| {
                DeviceConfig {
                    device_idx: *idx as u32,
                    device_uuid: uuid.clone(),
                    up_limit: 80,
                    mem_limit: 8 * 1024 * 1024 * 1024, // 8GB
                    sm_count: 82,
                    max_thread_per_sm: 1536,
                    total_cuda_cores: 2048,
                }
            })
            .collect();

        // Create actual shared memory to get a valid pointer
        let shmem = match ShmemConf::new()
            .size(std::mem::size_of::<SharedDeviceState>())
            .os_id(&shm_name)
            .open()
        {
            Ok(shmem) => shmem,
            Err(e) => {
                tracing::warn!("failed to open shared memory: {:?}, creating new one", e);
                let shmem = ShmemConf::new()
                    .size(std::mem::size_of::<SharedDeviceState>())
                    .os_id(&shm_name)
                    .create()
                    .expect("Failed to create mock shared memory");

                let ptr = shmem.as_ptr() as *mut SharedDeviceState;

                // Initialize with mock data
                unsafe {
                    ptr.write(SharedDeviceState::new(&mock_configs));
                }
                shmem
            }
        };

        let ptr = shmem.as_ptr() as *mut SharedDeviceState;

        Self {
            shmem: RefCell::new(shmem),
            ptr,
        }
    }

    /// Opens an existing shared memory segment.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let shmem = ShmemConf::new()
            .size(std::mem::size_of::<SharedDeviceState>())
            .use_tmpfs_with_dir(path.as_ref())
            .os_id("shm")
            .open()
            .context("Failed to open shared memory")?;

        let ptr = shmem.as_ptr() as *mut SharedDeviceState;

        Ok(Self {
            shmem: RefCell::new(shmem),
            ptr,
        })
    }

    /// Creates a new shared memory segment.
    pub fn create(path: impl AsRef<Path>, configs: &[DeviceConfig]) -> Result<Self> {
        let old_umask = unsafe { libc::umask(0) };

        let shmem = match ShmemConf::new()
            .size(std::mem::size_of::<SharedDeviceState>())
            .use_tmpfs_with_dir(path.as_ref())
            .os_id("shm")
            .mode(
                Mode::S_IRUSR
                    | Mode::S_IWUSR
                    | Mode::S_IRGRP
                    | Mode::S_IWGRP
                    | Mode::S_IROTH
                    | Mode::S_IWOTH,
            )
            .create()
        {
            Ok(shmem) => shmem,
            Err(ShmemError::LinkExists) => {
                // If it already exists, try to open it.
                ShmemConf::new()
                    .size(std::mem::size_of::<SharedDeviceState>())
                    .use_tmpfs_with_dir(path.as_ref())
                    .os_id("shm")
                    .open()
                    .context("Failed to open existing shared memory")?
            }
            Err(e) => return Err(anyhow::anyhow!("Failed to create shared memory: {}", e)),
        };

        unsafe {
            libc::umask(old_umask);
        }

        let ptr = shmem.as_ptr() as *mut SharedDeviceState;

        // Initialize the shared memory data.
        unsafe {
            ptr.write(SharedDeviceState::new(configs));
        }

        info!(
            path = ?path.as_ref(),
            "Created shared memory segment"
        );

        Ok(Self {
            shmem: RefCell::new(shmem),
            ptr,
        })
    }

    /// Gets a pointer to the shared device state.
    pub fn get_ptr(&self) -> *mut SharedDeviceState {
        self.ptr
    }

    pub fn set_owner(&self, is_owner: bool) {
        self.shmem.borrow_mut().set_owner(is_owner);
    }

    /// Gets a reference to the shared device state.
    pub fn get_state(&self) -> &SharedDeviceState {
        unsafe { &*self.ptr }
    }
}

// Implement Send and Sync because SharedDeviceState uses atomic operations.
unsafe impl Send for SharedMemoryHandle {}
unsafe impl Sync for SharedMemoryHandle {}

impl Drop for SharedMemoryHandle {
    fn drop(&mut self) {
        if self.shmem.borrow().is_owner() {
            let need_cleanup = self.get_state().get_all_pids().is_empty();

            if !need_cleanup {
                // Don't clean up - other processes are still using it
                self.shmem.borrow_mut().set_owner(false);
            } else {
                let path = self.shmem.borrow().get_tmpfs_file_path();
                info!(
                    path = ?path,
                    "No other processes using shared memory, allowing cleanup"
                );
                // Clean up - no other processes are using it
                self.shmem.borrow_mut().set_owner(true);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process;

    #[test]
    fn test_shared_memory_preserved_with_pids() {
        let test_id = format!("test_preserved_{}", process::id());
        let configs = vec![];

        // Test case: Create shared memory, add PID, then test cleanup behavior
        let handle = SharedMemoryHandle::create(&test_id, &configs).unwrap();

        // Initially no PIDs
        assert!(handle.get_state().get_all_pids().is_empty());
        assert!(handle.shmem.borrow().is_owner());

        // Add a PID to simulate another process using the memory
        handle.get_state().add_pid(12345);
        assert!(!handle.get_state().get_all_pids().is_empty());
        assert_eq!(handle.get_state().get_all_pids().len(), 1);

        drop(handle);

        let handle2 = SharedMemoryHandle::open(&test_id);

        assert!(handle2.is_ok());

        {
            std::fs::remove_file(format!("/dev/shm/{test_id}")).unwrap();
        }
    }

    #[test]
    fn test_shared_memory_cleanup() {
        let test_id = format!("test_cleanup_{}", process::id());
        let configs = vec![];

        // Test case: Create shared memory, add PID, then test cleanup behavior
        let handle = SharedMemoryHandle::create(&test_id, &configs).unwrap();

        // Initially no PIDs
        assert!(handle.get_state().get_all_pids().is_empty());
        assert!(handle.shmem.borrow().is_owner());

        drop(handle);

        let handle2 = SharedMemoryHandle::open(&test_id);

        assert!(handle2.is_err());
    }
}
