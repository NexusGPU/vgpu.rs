use anyhow::Context;
use anyhow::Result;
use shared_memory::Mode;
use shared_memory::Shmem;
use shared_memory::ShmemConf;
use shared_memory::ShmemError;
use tracing::info;
use tracing::warn;

/// Safely access shared memory, automatically handling the segment's lifecycle.
pub struct SharedMemoryHandle {
    _shmem: Shmem,
    ptr: *mut super::SharedDeviceState,
    identifier: String,
}

impl SharedMemoryHandle {
    /// Opens an existing shared memory segment.
    pub fn open(identifier: &str) -> Result<Self> {
        let shmem = ShmemConf::new()
            .size(std::mem::size_of::<super::SharedDeviceState>())
            .os_id(identifier)
            .open()
            .context("Failed to open shared memory")?;

        let ptr = shmem.as_ptr() as *mut super::SharedDeviceState;

        // Increment reference count when opening existing shared memory
        unsafe {
            (*ptr).increment_ref_count();
        }

        Ok(Self {
            _shmem: shmem,
            ptr,
            identifier: identifier.to_string(),
        })
    }

    /// Creates a new shared memory segment.
    pub fn create(identifier: &str, configs: &[super::DeviceConfig]) -> Result<Self> {
        let old_umask = unsafe { libc::umask(0) };

        let shmem = match ShmemConf::new()
            .size(std::mem::size_of::<super::SharedDeviceState>())
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
                    .size(std::mem::size_of::<super::SharedDeviceState>())
                    .os_id(identifier)
                    .open()
                    .context("Failed to open existing shared memory")?
            }
            Err(e) => return Err(anyhow::anyhow!("Failed to create shared memory: {}", e)),
        };

        unsafe {
            libc::umask(old_umask);
        }

        let ptr = shmem.as_ptr() as *mut super::SharedDeviceState;

        // Initialize the shared memory data.
        unsafe {
            ptr.write(super::SharedDeviceState::new(configs));
        }

        info!(
            identifier = %identifier,
            "Created shared memory segment"
        );

        Ok(Self {
            _shmem: shmem,
            ptr,
            identifier: identifier.to_string(),
        })
    }

    /// Gets a pointer to the shared device state.
    pub fn get_ptr(&self) -> *mut super::SharedDeviceState {
        self.ptr
    }

    /// Gets a reference to the shared device state.
    pub fn get_state(&self) -> &super::SharedDeviceState {
        unsafe { &*self.ptr }
    }

    /// Gets the shared memory identifier.
    pub fn get_identifier(&self) -> &str {
        &self.identifier
    }
}

// Implement Drop to handle reference counting cleanup
impl Drop for SharedMemoryHandle {
    fn drop(&mut self) {
        unsafe {
            match (*self.ptr).try_decrement_ref_count() {
                Ok(ref_count) => {
                    info!(
                        identifier = %self.identifier,
                        "Dropped shared memory reference, remaining count: {}",
                        ref_count
                    );
                    if ref_count == 0 {
                        info!(
                            identifier = %self.identifier,
                            "Last reference to shared memory dropped, reference count: {}",
                            ref_count
                        );
                        // Note: We don't manually delete the file here. The shared_memory crate
                        // and OS will handle cleanup when the underlying Shmem is dropped.
                    }
                }
                Err(super::RefCountError::Underflow) => {
                    warn!(
                        identifier = %self.identifier,
                        "Attempted to decrement reference count below zero"
                    );
                }
            }
        }
    }
}

// Implement Send and Sync because SharedDeviceState uses atomic operations.
unsafe impl Send for SharedMemoryHandle {}
unsafe impl Sync for SharedMemoryHandle {}
