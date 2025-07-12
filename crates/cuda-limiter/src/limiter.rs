use std::collections::HashMap;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::time::Duration;

use cudarc::driver::sys::CUdevice;
use cudarc::driver::DriverError;
use thiserror::Error;
use trap::TrapError;
use utils::shared_memory::SharedMemoryHandle;

#[derive(Error, Debug)]
pub(crate) enum Error {
    #[error("CUDA driver error: `{0}`")]
    CuDriver(#[from] DriverError),

    #[error("Trap error: `{0}`")]
    Trap(#[from] TrapError),

    #[error("Invalid device index: {0}")]
    InvalidDevice(u32),

    #[error("Invalid CUDA device: {0}")]
    InvalidCuDevice(CUdevice),

    #[error("Pod name not found in environment")]
    PodNameNotFound,

    #[error("Shared memory access failed: {0}")]
    SharedMemoryError(#[from] anyhow::Error),

    #[error("Device {0} not configured")]
    DeviceNotConfigured(u32),

    #[error("NVML error: {0}")]
    Nvml(#[from] nvml_wrapper::error::NvmlError),
}

/// Configuration for a CUDA device
#[derive(Debug, Clone)]
pub(crate) struct DeviceConfig {
    /// Device index
    pub device_idx: u32,
    /// Utilization percentage limit (0-100)
    pub up_limit: u32,
    /// Memory limit in bytes (0 means no limit)
    pub mem_limit: u64,
    /// Total number of CUDA cores
    pub total_cuda_cores: u32,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_idx: 0,
            up_limit: 80,        // 80% default utilization limit
            mem_limit: 0,        // No memory limit by default
            total_cuda_cores: 0, // Will be calculated based on hardware
        }
    }
}

impl DeviceConfig {
    /// Create a DeviceConfig with specified parameters
    pub fn new(device_idx: u32, up_limit: u32, mem_limit: u64, total_cuda_cores: u32) -> Self {
        Self {
            device_idx,
            up_limit,
            mem_limit,
            total_cuda_cores,
        }
    }
}

/// Internal device information and state
#[derive(Debug)]
pub(crate) struct DeviceInfo {
    pub(crate) dev_idx: u32,
    /// Block dimensions set by cuFuncSetBlockShape
    pub(crate) block_x: AtomicU32,
    pub(crate) block_y: AtomicU32,
    pub(crate) block_z: AtomicU32,
    /// Condition variable for signaling when CUDA cores become available
    pub(crate) cores_condvar: Arc<(Mutex<()>, Condvar)>,
}

/// Main limiter struct that manages CUDA resource limits
pub(crate) struct Limiter {
    /// Process ID being monitored
    pub(crate) pid: u32,
    /// Pod name for shared memory access
    pod_name: String,
    /// Information about each device
    pub(crate) devices: Vec<DeviceInfo>,
    /// Shared memory handles for each device
    shared_memory_handles: HashMap<u32, SharedMemoryHandle>,
}

impl std::fmt::Debug for Limiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Limiter")
            .field("pid", &self.pid)
            .field("pod_name", &self.pod_name)
            .field("devices", &self.devices)
            .field(
                "shared_memory_handles_count",
                &self.shared_memory_handles.len(),
            )
            .finish()
    }
}

impl Limiter {
    /// Creates a new Limiter instance
    pub(crate) fn new(pid: u32, pod_name: String, device_indices: Vec<u32>) -> Result<Self, Error> {
        let mut devices = Vec::new();
        let mut shared_memory_handles = HashMap::new();

        for device_idx in device_indices {
            // Create device info
            let device_info = DeviceInfo {
                dev_idx: device_idx,
                block_x: AtomicU32::new(0),
                block_y: AtomicU32::new(0),
                block_z: AtomicU32::new(0),
                cores_condvar: Arc::new((Mutex::new(()), Condvar::new())),
            };

            devices.push(device_info);

            // Open shared memory for this device
            let shared_memory_handle =
                SharedMemoryHandle::open(&pod_name).map_err(Error::SharedMemoryError)?;
            shared_memory_handles.insert(device_idx, shared_memory_handle);
        }

        Ok(Limiter {
            pid,
            pod_name,
            devices,
            shared_memory_handles,
        })
    }

    /// Get device info by index
    fn get_device(&self, device_idx: u32) -> Result<&DeviceInfo, Error> {
        self.devices
            .iter()
            .find(|d| d.dev_idx == device_idx)
            .ok_or(Error::InvalidDevice(device_idx))
    }

    /// Rate limiter that waits for available CUDA cores
    pub(crate) fn rate_limiter(&self, device_idx: u32, grids: u32, _blocks: u32) {
        let device = self.get_device(device_idx).expect("get device");
        let kernel_size = grids as i32;

        // Get shared memory handle for this device
        let handle = self
            .shared_memory_handles
            .get(&device_idx)
            .expect("Shared memory handle not found");

        // Wait for available cores using condition variable
        let (lock, cvar) = &*device.cores_condvar;
        let mut guard = lock.lock().expect("poisoned");

        loop {
            // Try to consume cores from shared memory
            let state = handle.get_state();
            let available = state.get_available_cores();

            if available >= kernel_size {
                // Atomically decrease available cores
                let current = state
                    .available_cuda_cores
                    .fetch_sub(kernel_size, Ordering::AcqRel);
                if current >= kernel_size {
                    break; // Successfully consumed cores
                } else {
                    // Race condition occurred, restore cores and retry
                    state
                        .available_cuda_cores
                        .fetch_add(kernel_size, Ordering::AcqRel);
                }
            }

            // Wait for cores to become available
            guard = cvar
                .wait_timeout(guard, Duration::from_millis(10))
                .expect("poisoned")
                .0;
        }
    }

    /// Get pod memory usage from shared memory
    pub(crate) fn get_pod_memory_usage(&self, device_idx: u32) -> Result<(u64, u64), Error> {
        let handle = self
            .shared_memory_handles
            .get(&device_idx)
            .ok_or(Error::DeviceNotConfigured(device_idx))?;

        let state = handle.get_state();
        let used = state.get_pod_memory_used();
        let limit = state.get_mem_limit();

        Ok((used, limit))
    }

    /// Get memory info for NVML hooks (total, used, free)
    pub(crate) fn get_memory_info(&self, device_idx: u32) -> Result<(u64, u64, u64), Error> {
        let handle = self
            .shared_memory_handles
            .get(&device_idx)
            .ok_or(Error::DeviceNotConfigured(device_idx))?;

        let state = handle.get_state();
        let total = state.get_mem_limit();
        let used = state.get_pod_memory_used();
        let free = total.saturating_sub(used);

        Ok((total, used, free))
    }

    /// Get memory limit for a device
    pub(crate) fn get_mem_limit(&self, device_idx: u32) -> Result<u64, Error> {
        let handle = self
            .shared_memory_handles
            .get(&device_idx)
            .ok_or(Error::DeviceNotConfigured(device_idx))?;

        let state = handle.get_state();
        Ok(state.get_mem_limit())
    }

    /// Get block dimensions for a device
    pub(crate) fn get_block_dimensions(&self, device_idx: u32) -> Result<(u32, u32, u32), Error> {
        let device = self.get_device(device_idx)?;
        Ok((
            device.block_x.load(Ordering::Acquire),
            device.block_y.load(Ordering::Acquire),
            device.block_z.load(Ordering::Acquire),
        ))
    }

    /// Set block dimensions for a device
    pub(crate) fn set_block_dimensions(
        &self,
        device_idx: u32,
        x: u32,
        y: u32,
        z: u32,
    ) -> Result<(), Error> {
        let device = self.get_device(device_idx)?;

        device.block_x.store(x, Ordering::Release);
        device.block_y.store(y, Ordering::Release);
        device.block_z.store(z, Ordering::Release);

        Ok(())
    }
}

/// Get pod name from environment variable
pub(crate) fn get_pod_name() -> Result<String, Error> {
    std::env::var("POD_NAME").map_err(|_| Error::PodNameNotFound)
}
