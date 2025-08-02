use std::cell::RefCell;

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
    identifier: String,
}

impl SharedMemoryHandle {
    /// Opens an existing shared memory segment.
    pub fn open(identifier: &str) -> Result<Self> {
        let shmem = ShmemConf::new()
            .size(std::mem::size_of::<SharedDeviceState>())
            .os_id(identifier)
            .open()
            .context("Failed to open shared memory")?;

        let ptr = shmem.as_ptr() as *mut SharedDeviceState;

        Ok(Self {
            shmem: RefCell::new(shmem),
            ptr,
            identifier: identifier.to_string(),
        })
    }

    /// Creates a new shared memory segment.
    pub fn create(identifier: &str, configs: &[DeviceConfig]) -> Result<Self> {
        let old_umask = unsafe { libc::umask(0) };

        let shmem = match ShmemConf::new()
            .size(std::mem::size_of::<SharedDeviceState>())
            .os_id(identifier)
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
                    .os_id(identifier)
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
            identifier = %identifier,
            "Created shared memory segment"
        );

        Ok(Self {
            shmem: RefCell::new(shmem),
            ptr,
            identifier: identifier.to_string(),
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

    /// Gets the shared memory identifier.
    pub fn get_identifier(&self) -> &str {
        &self.identifier
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
                info!(
                    identifier = %self.identifier,
                    pid_count = self.get_state().get_all_pids().len(),
                    "Other processes still using shared memory, preserving it"
                );
                // Don't clean up - other processes are still using it
                self.shmem.borrow_mut().set_owner(false);
            } else {
                info!(
                    identifier = %self.identifier,
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
