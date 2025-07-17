use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use cudarc::driver::sys::CUdevice;
use cudarc::driver::CudaContext;
use cudarc::driver::DriverError;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;
use nvml_wrapper_sys::bindings::nvmlDevice_t;
use thiserror::Error;
use trap::TrapError;
use utils::shared_memory::SharedMemoryHandle;

use crate::detour;

#[derive(Error, Debug)]
pub(crate) enum Error {
    #[error("CUDA driver error: `{0}`")]
    CuDriver(#[from] DriverError),

    #[error("Trap error: `{0}`")]
    Trap(#[from] TrapError),

    #[error("Invalid CUDA device: {0}")]
    #[allow(dead_code)]
    InvalidCuDevice(CUdevice),

    #[error("Pod name or namespace not found in environment")]
    PodNameOrNamespaceNotFound,

    #[error("Shared memory access failed: {0}")]
    SharedMemory(#[from] anyhow::Error),

    #[error("Device {0} not configured")]
    DeviceNotConfigured(String),

    #[error("Device dim not configured: {0}")]
    DeviceDimNotConfigured(i32),

    #[error("NVML error: {0}")]
    Nvml(#[from] nvml_wrapper::error::NvmlError),
}

/// Internal device information and state
#[derive(Debug)]
pub(crate) struct DeviceDim {
    /// Block dimensions set by cuFuncSetBlockShape
    pub(crate) block_x: AtomicU32,
    pub(crate) block_y: AtomicU32,
    pub(crate) block_z: AtomicU32,
}

/// Main limiter struct that manages CUDA resource limits
pub(crate) struct Limiter {
    /// Process ID being monitored
    pub(crate) pid: u32,
    /// Pod name for shared memory access
    pod_identifier: String,
    /// Shared memory handle for each device
    shared_memory_handle: SharedMemoryHandle,
    /// NVML instance
    nvml: Nvml,
    /// Device dimensions
    current_devices_dim: HashMap<i32, DeviceDim>,
    /// CUDA device mapping (CUdevice -> (device_index, device_uuid))
    cu_device_mapping: BTreeMap<CUdevice, (u32, String)>,
}

impl std::fmt::Debug for Limiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Limiter")
            .field("pid", &self.pid)
            .field("identifier", &self.pod_identifier)
            .finish()
    }
}

impl Limiter {
    /// Creates a new Limiter instance
    pub(crate) fn new(
        pid: u32,
        nvml: Nvml,
        pod_identifier: String,
        gpu_uuids: &[String],
    ) -> Result<Self, Error> {
        let mut cu_device_mapping = BTreeMap::new();
        let mut uuid_mapping = HashMap::new();

        for i in 0..gpu_uuids.len() {
            let ctx = CudaContext::new(i)?;
            let cu_uuid = ctx.uuid()?;
            let cu_uuid = uuid_to_string_formatted(&cu_uuid.bytes);

            if gpu_uuids.contains(&cu_uuid) {
                let device = nvml.device_by_uuid(cu_uuid.as_str())?;
                let index = device.index()?;
                uuid_mapping.insert(i as i32, cu_uuid.clone());
                tracing::info!("Device {i} UUID: {}", cu_uuid);
                cu_device_mapping.insert(ctx.cu_device(), (index, cu_uuid.clone()));
            }
        }

        detour::GLOBAL_DEVICE_UUIDS
            .set(uuid_mapping)
            .expect("set GLOBAL_DEVICE_UUIDS");

        let shared_memory_handle = SharedMemoryHandle::open(&pod_identifier)?;
        Ok(Limiter {
            pid,
            pod_identifier,
            current_devices_dim: HashMap::new(),
            shared_memory_handle,
            nvml,
            cu_device_mapping,
        })
    }

    /// Rate limiter that waits for available CUDA cores with exponential backoff
    pub(crate) fn rate_limiter(
        &self,
        device_uuid: &str,
        grids: u32,
        _blocks: u32,
    ) -> Result<(), Error> {
        let kernel_size = grids as i32;

        // Get shared memory handle for this device
        let state = self.shared_memory_handle.get_state();

        // Check if device exists, return error instead of panic
        if !state.has_device(device_uuid) {
            return Err(Error::DeviceNotConfigured(device_uuid.to_string()));
        }

        // Exponential backoff parameters
        let mut backoff_ms = 1;
        const MAX_BACKOFF_MS: u64 = 100;
        const BACKOFF_MULTIPLIER: u64 = 2;

        loop {
            let available = state
                .with_device_by_uuid(device_uuid, |device| device.get_available_cores())
                .ok_or_else(|| Error::DeviceNotConfigured(device_uuid.to_string()))?;

            if available >= kernel_size {
                // Successfully reserved cores
                state
                    .with_device_by_uuid_mut(device_uuid, |device| {
                        device.fetch_sub_available_cores(kernel_size)
                    })
                    .ok_or_else(|| Error::DeviceNotConfigured(device_uuid.to_string()))?;
                break;
            }

            // Wait with exponential backoff to avoid busy-waiting
            std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
            backoff_ms = (backoff_ms * BACKOFF_MULTIPLIER).min(MAX_BACKOFF_MS);
        }

        Ok(())
    }

    pub(crate) fn get_device_count(&self) -> u32 {
        self.cu_device_mapping.len() as u32
    }

    /// Get pod memory usage from shared memory
    pub(crate) fn get_pod_memory_usage(&self, device_uuid: &str) -> Result<(u64, u64), Error> {
        let state = self.shared_memory_handle.get_state();

        if let Some((used, limit)) = state.with_device_by_uuid(device_uuid, |device| {
            (device.get_pod_memory_used(), device.get_mem_limit())
        }) {
            Ok((used, limit))
        } else {
            Err(Error::DeviceNotConfigured(device_uuid.to_string()))
        }
    }

    /// Get the memory limit for a specific device
    pub(crate) fn get_pod_memory_usage_cu(&self, cu_device: CUdevice) -> Result<(u64, u64), Error> {
        let device_uuid = self
            .cu_device_mapping
            .get(&cu_device)
            .ok_or(Error::InvalidCuDevice(cu_device))?;

        self.get_pod_memory_usage(device_uuid.1.as_str())
    }

    pub(crate) fn nvml_index_mapping(&self, index: usize) -> Result<u32, Error> {
        let key = self
            .cu_device_mapping
            .keys()
            .nth(index)
            .ok_or(Error::InvalidCuDevice(index as CUdevice))?;
        let nvml_idx = self
            .cu_device_mapping
            .get(key)
            .ok_or(Error::InvalidCuDevice(*key))?
            .0;
        Ok(nvml_idx)
    }

    /// Get the NVML device handle for a specific device
    pub(crate) fn device_uuid_by_handle(
        &self,
        device_handle: nvmlDevice_t,
    ) -> Result<Option<String>, NvmlError> {
        for (_, (_, gpu_uuid)) in self.cu_device_mapping.iter() {
            let dev = self.nvml.device_by_uuid(gpu_uuid.as_str());
            match dev {
                Ok(dev) => {
                    let handle = unsafe { dev.handle() };
                    if handle == device_handle {
                        return Ok(Some(dev.uuid()?));
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to get device by uuid {}: {}, skipped", gpu_uuid, e);
                    continue;
                }
            }
        }
        Ok(None)
    }

    /// Get block dimensions for a device
    pub(crate) fn get_block_dimensions(&self, device_idx: i32) -> Result<(u32, u32, u32), Error> {
        let device = self
            .current_devices_dim
            .get(&device_idx)
            .ok_or_else(|| Error::DeviceDimNotConfigured(device_idx))?;
        Ok((
            device.block_x.load(Ordering::Acquire),
            device.block_y.load(Ordering::Acquire),
            device.block_z.load(Ordering::Acquire),
        ))
    }

    /// Set block dimensions for a device
    pub(crate) fn set_block_dimensions(
        &self,
        device_idx: i32,
        x: u32,
        y: u32,
        z: u32,
    ) -> Result<(), Error> {
        let device = self
            .current_devices_dim
            .get(&device_idx)
            .ok_or_else(|| Error::DeviceDimNotConfigured(device_idx))?;

        device.block_x.store(x, Ordering::Release);
        device.block_y.store(y, Ordering::Release);
        device.block_z.store(z, Ordering::Release);

        Ok(())
    }
}

/// Get pod name from environment variable
pub(crate) fn get_pod_identifier() -> Result<String, Error> {
    let name = std::env::var("POD_NAME").map_err(|_| Error::PodNameOrNamespaceNotFound)?;
    let namespace =
        std::env::var("POD_NAMESPACE").map_err(|_| Error::PodNameOrNamespaceNotFound)?;
    Ok(format!("{namespace}_{name}"))
}

#[cfg(target_arch = "x86_64")]
fn uuid_to_string_formatted(uuid: &[i8; 16]) -> String {
    format!(
        "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        uuid[0], uuid[1], uuid[2], uuid[3],
        uuid[4], uuid[5],
        uuid[6], uuid[7],
        uuid[8], uuid[9],
        uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15],
    )
}

#[cfg(target_arch = "aarch64")]
fn uuid_to_string_formatted(uuid: &[u8; 16]) -> String {
    format!(
        "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        uuid[0], uuid[1], uuid[2], uuid[3],
        uuid[4], uuid[5],
        uuid[6], uuid[7],
        uuid[8], uuid[9],
        uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15],
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use utils::shared_memory::SharedDeviceInfo;

    /// Trait for shared memory operations
    trait SharedMemory {
        /// Gets a reference to the shared device state
        fn get_state(&self) -> &SharedDeviceInfo;
    }

    /// Mock implementation of SharedMemory for testing
    pub struct MockSharedMemory {
        state: Arc<SharedDeviceInfo>,
    }

    impl MockSharedMemory {
        /// Create a new MockSharedMemory with specified values
        pub fn new(total_cores: u32, up_limit: u32, mem_limit: u64) -> Self {
            let state = Arc::new(SharedDeviceInfo::new(total_cores, up_limit, mem_limit));
            Self { state }
        }

        /// Set available CUDA cores for testing
        pub fn set_available_cores(&self, cores: i32) {
            self.state.set_available_cores(cores);
        }

        /// Set memory limit for testing
        #[allow(dead_code)]
        pub fn set_mem_limit(&self, limit: u64) {
            self.state.set_mem_limit(limit);
        }

        /// Set pod memory used for testing
        pub fn set_pod_memory_used(&self, used: u64) {
            self.state.set_pod_memory_used(used);
        }
    }

    impl SharedMemory for MockSharedMemory {
        fn get_state(&self) -> &SharedDeviceInfo {
            &self.state
        }
    }

    #[test]
    fn test_memory_info_reporting() {
        let mock = MockSharedMemory::new(1000, 80, 1024 * 1024 * 1024); // 1GB limit
        mock.set_pod_memory_used(512 * 1024 * 1024); // 512MB used

        let state = mock.get_state();
        let total = state.get_mem_limit();
        let used = state.get_pod_memory_used();
        let free = total.saturating_sub(used);

        assert_eq!(total, 1024 * 1024 * 1024, "Total memory should be 1GB");
        assert_eq!(used, 512 * 1024 * 1024, "Used memory should be 512MB");
        assert_eq!(free, 512 * 1024 * 1024, "Free memory should be 512MB");
    }

    #[test]
    fn test_memory_info_reporting_edge_cases() {
        // Test when used memory equals total memory
        let mock = MockSharedMemory::new(1000, 80, 1024);
        mock.set_pod_memory_used(1024);

        let state = mock.get_state();
        let total = state.get_mem_limit();
        let used = state.get_pod_memory_used();
        let free = total.saturating_sub(used);

        assert_eq!(total, 1024, "Total memory should be 1024");
        assert_eq!(used, 1024, "Used memory should be 1024");
        assert_eq!(free, 0, "Free memory should be 0");

        // Test when used memory exceeds total memory (should saturate to 0)
        mock.set_pod_memory_used(2048);
        let used = state.get_pod_memory_used();
        let free = total.saturating_sub(used);

        assert_eq!(free, 0, "Free memory should saturate to 0");
    }

    #[test]
    fn test_shared_memory_trait_consistency() {
        let mock = MockSharedMemory::new(2048, 90, 2048 * 1024 * 1024);
        mock.set_available_cores(1500);
        mock.set_pod_memory_used(1024 * 1024 * 1024);

        let state = mock.get_state();

        assert_eq!(state.get_total_cores(), 2048, "Total cores should match");
        assert_eq!(state.get_up_limit(), 90, "Up limit should match");
        assert_eq!(
            state.get_mem_limit(),
            2048 * 1024 * 1024,
            "Memory limit should match"
        );
        assert_eq!(
            state.get_available_cores(),
            1500,
            "Available cores should match"
        );
        assert_eq!(
            state.get_pod_memory_used(),
            1024 * 1024 * 1024,
            "Pod memory used should match"
        );
    }
}
