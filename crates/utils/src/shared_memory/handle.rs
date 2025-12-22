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

/// Shared memory file name constant
pub const SHM_PATH_SUFFIX: &str = "shm";

/// Safely access shared memory, automatically handling the segment's lifecycle.
pub struct SharedMemoryHandle {
    shmem: RefCell<Shmem>,
    ptr: *mut SharedDeviceState,
}

impl SharedMemoryHandle {
    /// Creates a mock SharedMemoryHandle with predefined test data.
    /// This function is useful for testing without requiring actual shared memory.
    pub fn mock(shm_path: impl AsRef<Path>, gpu_idx_uuids: Vec<(usize, String)>) -> Self {
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
            .use_tmpfs_with_dir(shm_path.as_ref())
            .os_id(SHM_PATH_SUFFIX)
            .open()
        {
            Ok(shmem) => shmem,
            Err(e) => {
                tracing::warn!(
                    "failed to open shared memory shm_name: {:?}, err: {:?}, creating new one",
                    shm_path.as_ref(),
                    e
                );

                std::fs::create_dir_all(shm_path.as_ref())
                    .expect("Failed to create mock shared memory directory");

                let shmem = ShmemConf::new()
                    .size(std::mem::size_of::<SharedDeviceState>())
                    .use_tmpfs_with_dir(shm_path.as_ref())
                    .os_id(SHM_PATH_SUFFIX)
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

    /// Opens an existing shared memory segment, creates one if it doesn't exist.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        match ShmemConf::new()
            .size(std::mem::size_of::<SharedDeviceState>())
            .use_tmpfs_with_dir(path.as_ref())
            .os_id(SHM_PATH_SUFFIX)
            .open()
        {
            Ok(mut shmem) => {
                shmem.set_owner(false);
                let ptr = shmem.as_ptr() as *mut SharedDeviceState;
                Ok(Self {
                    shmem: RefCell::new(shmem),
                    ptr,
                })
            }
            Err(e) => {
                info!(
                    path = ?path.as_ref(),
                    err = ?e,
                    "Shared memory not found, creating new one"
                );
                Self::create(path, &[])
            }
        }
    }

    /// Creates a new shared memory segment.
    pub fn create(path: impl AsRef<Path>, configs: &[DeviceConfig]) -> Result<Self> {
        std::fs::create_dir_all(path.as_ref())?;
        let old_umask = unsafe { libc::umask(0) };
        let mut shmem = match ShmemConf::new()
            .size(std::mem::size_of::<SharedDeviceState>())
            .use_tmpfs_with_dir(path.as_ref())
            .os_id(SHM_PATH_SUFFIX)
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
                    .os_id(SHM_PATH_SUFFIX)
                    .open()
                    .context("Failed to open existing shared memory")?
            }
            Err(e) => return Err(anyhow::anyhow!("Failed to create shared memory: {e}")),
        };
        // avoid cleanup by drop
        shmem.set_owner(false);
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_open_creates_when_not_exists() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let shm_path = temp_dir.path().join("test_open_create");

        let handle = SharedMemoryHandle::open(&shm_path).expect("Failed to open/create");

        let state = handle.get_state();
        assert_eq!(state.device_count(), 0);
    }

    #[test]
    fn test_open_existing_shared_memory() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let shm_path = temp_dir.path().join("test_open_existing");

        let configs = vec![DeviceConfig {
            device_idx: 0,
            device_uuid: "GPU-test-uuid".to_string(),
            up_limit: 75,
            mem_limit: 4 * 1024 * 1024 * 1024,
            sm_count: 64,
            max_thread_per_sm: 1024,
            total_cuda_cores: 1024,
        }];

        let handle1 = SharedMemoryHandle::create(&shm_path, &configs).expect("Failed to create");
        assert_eq!(handle1.get_state().device_count(), 1);

        let handle2 = SharedMemoryHandle::open(&shm_path).expect("Failed to open existing");
        assert_eq!(handle2.get_state().device_count(), 1);

        let device_info = handle2
            .get_state()
            .get_device_info(0)
            .expect("Device should exist");
        assert_eq!(device_info.0, "GPU-test-uuid");
    }

    #[test]
    fn test_open_multiple_times() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let shm_path = temp_dir.path().join("test_open_multiple");

        let handle1 = SharedMemoryHandle::open(&shm_path).expect("Failed to open first time");
        assert_eq!(handle1.get_state().device_count(), 0);

        let handle2 = SharedMemoryHandle::open(&shm_path).expect("Failed to open second time");
        assert_eq!(handle2.get_state().device_count(), 0);

        drop(handle1);

        let handle3 = SharedMemoryHandle::open(&shm_path).expect("Failed to open third time");
        assert_eq!(handle3.get_state().device_count(), 0);
    }

    #[test]
    fn test_open_with_nested_path() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let shm_path = temp_dir.path().join("nested").join("path").join("test");

        let handle = SharedMemoryHandle::open(&shm_path).expect("Failed to open with nested path");
        assert_eq!(handle.get_state().device_count(), 0);

        assert!(shm_path.exists(), "Nested directories should be created");
    }
}
