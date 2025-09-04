use std::collections::HashMap;
use std::ffi::OsStr;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::time::Duration;

use cudarc::driver::sys::CUdevice;
use dashmap::DashMap;
use nvml_wrapper::error::nvml_try;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;
use nvml_wrapper_sys::bindings::nvmlDevice_t;
use once_cell::sync::OnceCell;
use trap::TrapError;
use utils::shared_memory::handle::SharedMemoryHandle;

use crate::culib;
use crate::detour::nvml::FN_NVML_DEVICE_GET_HANDLE_BY_INDEX_V2;

#[derive(thiserror::Error, Debug)]
pub(crate) enum Error {
    #[error("Trap error: `{0}`")]
    Trap(#[from] TrapError),

    #[error("Invalid CUDA device: {0}")]
    #[allow(dead_code)]
    InvalidCuDevice(CUdevice),

    #[error("Shared memory access failed: {0}")]
    SharedMemory(#[from] anyhow::Error),

    #[error("Pod name or namespace not found in environment")]
    PodNameOrNamespaceNotFound,

    #[error("Device {0} not configured")]
    DeviceNotConfigured(usize),

    #[error("Device dim not configured: {0}")]
    DeviceDimNotConfigured(usize),

    #[error("Device {0} not healthy")]
    DeviceNotHealthy(usize),

    #[error("NVML error: {0}")]
    Nvml(#[from] nvml_wrapper::error::NvmlError),

    #[error("CUDA error: {0:?}")]
    Cuda(cudarc::driver::sys::CUresult),

    #[error("Limiter not initialized")]
    LimiterNotInitialized,
}

#[derive(Debug)]
pub(crate) struct DeviceDim {
    /// Block dimensions set by cuFuncSetBlockShape
    pub(crate) block_x: AtomicU32,
    pub(crate) block_y: AtomicU32,
    pub(crate) block_z: AtomicU32,
}

/// Main limiter struct that manages CUDA resource limits
pub(crate) struct Limiter {
    /// Shared memory handle for each device (lazy initialized)
    shared_memory_handle: OnceCell<SharedMemoryHandle>,
    /// NVML instance
    nvml: Nvml,
    /// Device dimensions
    current_devices_dim: HashMap<usize, DeviceDim>,
    /// CUDA device mapping (CUdevice -> (device_index, device_uuid))
    cu_device_mapping: DashMap<CUdevice, (usize, String)>,
    /// GPU UUIDs
    gpu_idx_uuids: Vec<(usize, String)>,
}

impl std::fmt::Debug for Limiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Limiter").finish()
    }
}

impl Limiter {
    /// Creates a new Limiter instance
    pub(crate) fn new(nvml: Nvml, mut gpu_uuids: Vec<String>) -> Result<Self, Error> {
        gpu_uuids.sort();
        gpu_uuids.dedup();

        let mut gpu_idx_uuids = Vec::new();
        for uuid in gpu_uuids.into_iter() {
            let device = nvml.device_by_uuid(uuid.as_str())?;
            let index = device.index()?;
            gpu_idx_uuids.push((index as usize, uuid));
        }
        tracing::info!(
            "Limiter initialized with GPU UUIDs and indices: {:?}",
            gpu_idx_uuids
        );

        Ok(Self {
            shared_memory_handle: OnceCell::new(),
            current_devices_dim: HashMap::new(),
            nvml,
            cu_device_mapping: DashMap::new(),
            gpu_idx_uuids,
        })
    }

    /// Get or initialize the shared memory handle (lazy initialization)
    fn get_or_init_shared_memory(&self) -> Result<&SharedMemoryHandle, Error> {
        self.shared_memory_handle.get_or_try_init(|| {
            if let Some(shm_name) = crate::mock_shm_name() {
                Ok(SharedMemoryHandle::mock(
                    shm_name,
                    self.gpu_idx_uuids.clone(),
                ))
            } else {
                let pod_identifier = get_pod_identifier()?;
                SharedMemoryHandle::open(&pod_identifier).map_err(Error::SharedMemory)
            }
        })
    }

    pub(crate) fn insert_cu_device_if_not_exists(
        &self,
        cu_device: CUdevice,
        f: impl FnOnce() -> Result<String, Error>,
    ) -> Result<(), Error> {
        if !self.cu_device_mapping.contains_key(&cu_device) {
            let device_uuid = f()?;
            let nvml = Nvml::builder()
                .lib_path(&std::env::var_os("TF_NVML_LIB_PATH").unwrap_or(
                    OsStr::new("/lib/x86_64-linux-gnu/libnvidia-ml.so.1").to_os_string(),
                ))
                .init()
                .unwrap();

            let device = nvml.device_by_uuid(device_uuid.as_str())?;
            let raw_index = device.index()?;
            self.cu_device_mapping
                .insert(cu_device, (raw_index as usize, device_uuid));
        }
        Ok(())
    }

    pub(crate) fn device_raw_index_by_cu_device(
        &self,
        cu_device: CUdevice,
    ) -> Result<usize, Error> {
        if let Some(dev_idx_uuid) = self.cu_device_mapping.get(&cu_device) {
            Ok(dev_idx_uuid.0)
        } else {
            let uuid = culib::device_uuid(cu_device).map_err(Error::Cuda)?;
            let nvml = Nvml::builder()
                .lib_path(&std::env::var_os("TF_NVML_LIB_PATH").unwrap_or(
                    OsStr::new("/lib/x86_64-linux-gnu/libnvidia-ml.so.1").to_os_string(),
                ))
                .init()
                .unwrap();
            let device = nvml.device_by_uuid(uuid.as_str())?;
            let index: u32 = device.index()?;
            self.cu_device_mapping
                .insert(cu_device, (index as usize, uuid));
            Ok(index as usize)
        }
    }

    /// Rate limiter that waits for available CUDA cores with exponential backoff
    pub(crate) fn rate_limiter(
        &self,
        raw_device_index: usize,
        grids: u32,
        _blocks: u32,
    ) -> Result<(), Error> {
        let kernel_size = grids as i32;

        // Get shared memory handle for this device (lazy init)
        let handle = self.get_or_init_shared_memory()?;
        let state = handle.get_state();

        match state.with_device(raw_device_index, |device| device.device_info.get_up_limit()) {
            Some(up_limit) => {
                if up_limit >= 100 {
                    return Ok(());
                }
            }
            None => {
                return Err(Error::DeviceNotConfigured(raw_device_index));
            }
        }

        // Check if device exists, return error instead of panic
        if !state.has_device(raw_device_index) {
            return Err(Error::DeviceNotConfigured(raw_device_index));
        }
        if !state.is_healthy(Duration::from_secs(2)) {
            return Err(Error::DeviceNotHealthy(raw_device_index));
        }

        // Exponential backoff parameters
        let mut backoff_ms = 1;
        const MAX_BACKOFF_MS: u64 = 100;
        const BACKOFF_MULTIPLIER: u64 = 2;

        loop {
            let available = state
                .with_device(raw_device_index, |device| {
                    device.device_info.get_available_cores()
                })
                .ok_or_else(|| Error::DeviceNotConfigured(raw_device_index))?;

            if available >= kernel_size {
                // Successfully reserved cores
                state
                    .with_device(raw_device_index, |device| {
                        device.device_info.fetch_sub_available_cores(kernel_size)
                    })
                    .ok_or_else(|| Error::DeviceNotConfigured(raw_device_index))?;
                break;
            }

            // Wait with exponential backoff to avoid busy-waiting
            std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
            backoff_ms = (backoff_ms * BACKOFF_MULTIPLIER).min(MAX_BACKOFF_MS);
        }

        Ok(())
    }

    pub(crate) fn get_device_count(&self) -> u32 {
        self.gpu_idx_uuids.len() as u32
    }

    /// Get pod memory usage from shared memory
    pub(crate) fn get_pod_memory_usage(
        &self,
        raw_device_index: usize,
    ) -> Result<(u64, u64), Error> {
        let handle = self.get_or_init_shared_memory()?;
        let state = handle.get_state();

        if !state.is_healthy(Duration::from_secs(2)) {
            tracing::warn!("device {} is not healthy", raw_device_index);
        }

        if let Some((used, limit)) = state.with_device(raw_device_index, |device| {
            (
                device.device_info.get_pod_memory_used(),
                device.device_info.get_mem_limit(),
            )
        }) {
            Ok((used, limit))
        } else {
            Err(Error::DeviceNotConfigured(raw_device_index))
        }
    }

    /// Get the memory limit for a specific device
    pub(crate) fn get_pod_memory_usage_cu(&self, cu_device: CUdevice) -> Result<(u64, u64), Error> {
        let dev_idx = self.device_raw_index_by_cu_device(cu_device)?;
        self.get_pod_memory_usage(dev_idx)
    }

    pub(crate) fn device_raw_index_by_nvml_handle(
        &self,
        device_handle: nvmlDevice_t,
    ) -> Result<usize, NvmlError> {
        for (idx, uuid) in self.gpu_idx_uuids.iter() {
            let dev = if FN_NVML_DEVICE_GET_HANDLE_BY_INDEX_V2.is_none() {
                self.nvml
                    .device_by_index(*idx as u32)
                    .map(|dev| unsafe { dev.handle() })
            } else {
                let mut device_handle = device_handle;
                unsafe {
                    nvml_try(FN_NVML_DEVICE_GET_HANDLE_BY_INDEX_V2(
                        *idx as u32,
                        &mut device_handle,
                    ))
                    .map(|_| device_handle)
                }
            };

            match dev {
                Ok(dev) => {
                    if dev == device_handle {
                        return Ok(*idx);
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to get device by uuid {}: {}, skipped", uuid, e);
                    continue;
                }
            }
        }
        tracing::error!(
            "device_raw_index_by_nvml_handle: Failed to get device by handle: {:?} ",
            device_handle
        );
        Err(NvmlError::NotFound)
    }

    pub(crate) fn nvml_index_mapping(&self, index: usize) -> Result<usize, Error> {
        self.gpu_idx_uuids
            .get(index)
            .map(|(idx, _)| *idx)
            .ok_or_else(|| Error::DeviceNotConfigured(index))
    }

    /// Get block dimensions for a device
    pub(crate) fn get_block_dimensions(
        &self,
        raw_device_idx: usize,
    ) -> Result<(u32, u32, u32), Error> {
        let device = self
            .current_devices_dim
            .get(&raw_device_idx)
            .ok_or_else(|| Error::DeviceDimNotConfigured(raw_device_idx))?;
        Ok((
            device.block_x.load(Ordering::Acquire),
            device.block_y.load(Ordering::Acquire),
            device.block_z.load(Ordering::Acquire),
        ))
    }

    /// Set block dimensions for a device
    pub(crate) fn set_block_dimensions(
        &self,
        raw_device_idx: usize,
        x: u32,
        y: u32,
        z: u32,
    ) -> Result<(), Error> {
        let device = self
            .current_devices_dim
            .get(&raw_device_idx)
            .ok_or_else(|| Error::DeviceDimNotConfigured(raw_device_idx))?;

        device.block_x.store(x, Ordering::Release);
        device.block_y.store(y, Ordering::Release);
        device.block_z.store(z, Ordering::Release);

        Ok(())
    }
}

/// Get pod name from environment variable
fn get_pod_identifier() -> Result<String, Error> {
    let name = std::env::var("POD_NAME").map_err(|_| Error::PodNameOrNamespaceNotFound)?;
    let namespace =
        std::env::var("POD_NAMESPACE").map_err(|_| Error::PodNameOrNamespaceNotFound)?;
    Ok(format!("tf_shm_{namespace}_{name}"))
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
