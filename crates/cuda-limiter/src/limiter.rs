use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use cudarc::driver::sys::CUdevice;
use dashmap::DashMap;
use erl::KernelLimiter;
use nvml_wrapper::error::nvml_try;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;
use nvml_wrapper_sys::bindings::nvmlDevice_t;
use once_cell::sync::OnceCell;
use trap::TrapError;
use utils::shared_memory::{erl_adapter::ErlSharedMemoryAdapter, handle::SharedMemoryHandle};

use crate::culib;
use crate::detour::nvml::FN_NVML_DEVICE_GET_HANDLE_BY_INDEX_V2;
use crate::nvmllib::init_nvml;

/// Rate limiter configuration constants
const RATE_LIMITER_SLEEP_MS: u64 = 10;
const RATE_LIMITER_HEALTH_CHECK_INTERVAL: u64 = 10;
const RATE_LIMITER_LOG_INTERVAL: u64 = 50;

/// Cache TTL for up_limit to avoid frequent shared memory access
const UP_LIMIT_CACHE_TTL: Duration = Duration::from_secs(60);

struct CachedUpLimit {
    value: u32,
    last_refresh: Instant,
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum Error {
    #[error("Trap error: `{0}`")]
    Trap(#[from] TrapError),

    #[error("Invalid CUDA device: {0}")]
    #[allow(dead_code)]
    InvalidCuDevice(CUdevice),

    #[error("Shared memory access failed: {0}")]
    SharedMemory(#[from] anyhow::Error),

    #[error("Device {0} not configured")]
    DeviceNotConfigured(usize),

    #[error("Device dim not configured: {0}")]
    DeviceDimNotConfigured(usize),

    #[error("Device {device_idx} not healthy, last heartbeat {last_heartbeat}")]
    DeviceNotHealthy {
        device_idx: usize,
        last_heartbeat: u64,
    },

    #[error("ERL admission failed for device {device_idx}: {message}")]
    ErlAdmissionFailed { device_idx: usize, message: String },

    #[error("Unsupported shared memory version {version} for device {device_idx}")]
    UnsupportedVersion { version: u32, device_idx: usize },

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
    shared_memory_handle: OnceCell<Arc<SharedMemoryHandle>>,
    /// Cached ERL kernel limiter backed by shared memory
    erl_kernel_limiter: OnceCell<KernelLimiter<ErlSharedMemoryAdapter<Arc<SharedMemoryHandle>>>>,
    /// NVML instance
    nvml: Nvml,
    /// Device dimensions
    current_devices_dim: HashMap<usize, DeviceDim>,
    /// CUDA device mapping (CUdevice -> (device_index, device_uuid))
    cu_device_mapping: DashMap<CUdevice, (usize, String)>,
    /// GPU UUIDs
    gpu_idx_uuids: Vec<(usize, String)>,
    /// compute shard
    compute_shard: bool,
    /// isolation level
    isolation: Option<String>,
    /// Cached up_limit per device for fast path check
    /// device_index -> CachedUpLimit
    up_limit_cache: DashMap<usize, CachedUpLimit>,
}

impl std::fmt::Debug for Limiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Limiter").finish()
    }
}

impl Limiter {
    /// Creates a new Limiter instance
    pub(crate) fn new(
        nvml: Nvml,
        mut gpu_uuids: Vec<String>,
        compute_shard: bool,
        isolation: Option<String>,
    ) -> Result<Self, Error> {
        gpu_uuids.sort();
        gpu_uuids.dedup();

        let mut gpu_idx_uuids = Vec::new();
        for uuid in gpu_uuids.into_iter() {
            let device = nvml.device_by_uuid(uuid.replace("gpu-", "GPU-").as_str())?;
            let index = device.index()?;
            gpu_idx_uuids.push((index as usize, uuid));
        }
        tracing::info!(
            "Limiter initialized with GPU UUIDs and indices: {:?}",
            gpu_idx_uuids
        );

        Ok(Self {
            shared_memory_handle: OnceCell::new(),
            erl_kernel_limiter: OnceCell::new(),
            current_devices_dim: HashMap::new(),
            nvml,
            cu_device_mapping: DashMap::new(),
            gpu_idx_uuids,
            compute_shard,
            isolation,
            up_limit_cache: DashMap::new(),
        })
    }

    /// Get or initialize the shared memory handle (lazy initialization)
    fn get_or_init_shared_memory(&self) -> Result<&SharedMemoryHandle, Error> {
        Ok(self.get_or_init_shared_memory_arc()?.as_ref())
    }

    fn get_or_init_shared_memory_arc(&self) -> Result<&Arc<SharedMemoryHandle>, Error> {
        self.shared_memory_handle.get_or_try_init(|| {
            if let Some(shm_path) = crate::mock_shm_path() {
                Ok(Arc::new(SharedMemoryHandle::mock(
                    shm_path,
                    self.gpu_idx_uuids.clone(),
                )))
            } else {
                SharedMemoryHandle::open(shm_path())
                    .map(Arc::new)
                    .map_err(Error::SharedMemory)
            }
        })
    }

    fn get_or_init_kernel_limiter(
        &self,
    ) -> Result<&KernelLimiter<ErlSharedMemoryAdapter<Arc<SharedMemoryHandle>>>, Error> {
        self.erl_kernel_limiter.get_or_try_init(|| {
            let handle = Arc::clone(self.get_or_init_shared_memory_arc()?);
            let erl_adapter = ErlSharedMemoryAdapter::new(handle);
            Ok(KernelLimiter::new(erl_adapter))
        })
    }

    pub(crate) fn insert_cu_device_if_not_exists(
        &self,
        cu_device: CUdevice,
        f: impl FnOnce() -> Result<String, Error>,
    ) -> Result<(), Error> {
        if !self.cu_device_mapping.contains_key(&cu_device) {
            let device_uuid = f()?;
            // Create a new NVML instance to ensure proper initialization in this context
            let nvml = init_nvml()?;

            let device = nvml.device_by_uuid(device_uuid.replace("gpu-", "GPU-").as_str())?;
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
            // Create a new NVML instance to ensure proper initialization in this context
            let nvml = init_nvml()?;

            let device = nvml.device_by_uuid(uuid.replace("gpu-", "GPU-").as_str())?;
            let index: u32 = device.index()?;
            self.cu_device_mapping
                .insert(cu_device, (index as usize, uuid));
            Ok(index as usize)
        }
    }

    /// Rate limiter entry point - dispatches to V1 or V2 based on shared memory version
    pub(crate) fn rate_limiter(
        &self,
        raw_device_index: usize,
        grids: u32,
        blocks: u32,
    ) -> Result<(), Error> {
        // Fast path: check cached up_limit first (no shared memory access)
        if let Some(cached) = self.up_limit_cache.get(&raw_device_index) {
            if cached.value >= 100 && cached.last_refresh.elapsed() < UP_LIMIT_CACHE_TTL {
                return Ok(());
            }
        }

        // Slow path: access shared memory
        let handle = self.get_or_init_shared_memory()?;
        let state = handle.get_state();

        let up_limit = state
            .with_device(
                raw_device_index,
                |device| device.device_info.get_up_limit(),
                |device| device.device_info.get_up_limit(),
            )
            .ok_or(Error::DeviceNotConfigured(raw_device_index))?;

        // Update cache
        self.up_limit_cache.insert(
            raw_device_index,
            CachedUpLimit {
                value: up_limit,
                last_refresh: Instant::now(),
            },
        );

        if up_limit >= 100 {
            return Ok(());
        }

        self.check_device_health(state, raw_device_index)?;

        // Dispatch to appropriate version
        match state.version() {
            1 => self.rate_limiter_v1(state, raw_device_index, grids),
            2 => self.rate_limiter_v2(raw_device_index, grids, blocks),
            version => Err(Error::UnsupportedVersion {
                version,
                device_idx: raw_device_index,
            }),
        }
    }

    /// V1: Core-based rate limiting
    fn rate_limiter_v1(
        &self,
        state: &utils::shared_memory::SharedDeviceState,
        raw_device_index: usize,
        grids: u32,
    ) -> Result<(), Error> {
        let kernel_size = grids as i32;

        self.wait_and_retry(raw_device_index, || {
            let available = state
                .with_device_v1(raw_device_index, |device| {
                    device.device_info.get_available_cores()
                })
                .ok_or_else(|| Error::DeviceNotConfigured(raw_device_index))?;

            if available >= kernel_size {
                state
                    .with_device_v1(raw_device_index, |device| {
                        device.device_info.fetch_sub_available_cores(kernel_size)
                    })
                    .ok_or_else(|| Error::DeviceNotConfigured(raw_device_index))?;

                tracing::debug!(
                    device_idx = raw_device_index,
                    grids = grids,
                    available_cores = available,
                    "V1 core-based admission granted"
                );
                Ok(true)
            } else {
                Ok(false)
            }
        })
    }

    /// V2: ERL rate limiting
    fn rate_limiter_v2(
        &self,
        raw_device_index: usize,
        grids: u32,
        blocks: u32,
    ) -> Result<(), Error> {
        self.wait_and_retry(raw_device_index, || {
            match self.try_erl_admission_control(raw_device_index, grids, blocks) {
                Ok(true) => {
                    tracing::debug!(
                        device_idx = raw_device_index,
                        grids = grids,
                        blocks = blocks,
                        "V2 ERL admission granted"
                    );
                    Ok(true)
                }
                Ok(false) => Ok(false),
                Err(e) => {
                    tracing::error!(
                        device_idx = raw_device_index,
                        grids = grids,
                        blocks = blocks,
                        error = %e,
                        "ERL admission control failed"
                    );
                    Err(Error::ErlAdmissionFailed {
                        device_idx: raw_device_index,
                        message: e.to_string(),
                    })
                }
            }
        })
    }

    fn wait_and_retry<F>(&self, raw_device_index: usize, mut check: F) -> Result<(), Error>
    where
        F: FnMut() -> Result<bool, Error>,
    {
        let mut wait_times: u64 = 0;

        loop {
            match check() {
                Ok(true) => {
                    if wait_times > 0 {
                        tracing::debug!(
                            device_idx = raw_device_index,
                            wait_times,
                            total_wait_ms = wait_times * RATE_LIMITER_SLEEP_MS,
                            "Request approved after waiting"
                        );
                    }
                    return Ok(());
                }
                Ok(false) => {
                    thread::sleep(Duration::from_millis(RATE_LIMITER_SLEEP_MS));
                    wait_times += 1;

                    // Periodic health check
                    if wait_times.is_multiple_of(RATE_LIMITER_HEALTH_CHECK_INTERVAL) {
                        let handle = self.get_or_init_shared_memory()?;
                        let state = handle.get_state();
                        self.check_device_health(state, raw_device_index)?;
                    }

                    // Periodic logging when waiting too long
                    if wait_times.is_multiple_of(RATE_LIMITER_LOG_INTERVAL) {
                        tracing::warn!(
                            device_idx = raw_device_index,
                            wait_times,
                            total_wait_ms = wait_times * RATE_LIMITER_SLEEP_MS,
                            "Still waiting for tokens (possible starvation)"
                        );
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Check if device exists and is healthy
    fn check_device_health(
        &self,
        state: &utils::shared_memory::SharedDeviceState,
        raw_device_index: usize,
    ) -> Result<(), Error> {
        if !state.has_device(raw_device_index) {
            return Err(Error::DeviceNotConfigured(raw_device_index));
        }
        if !state.is_healthy(Duration::from_secs(2)) {
            let last_heartbeat = state.get_last_heartbeat();
            return Err(Error::DeviceNotHealthy {
                device_idx: raw_device_index,
                last_heartbeat,
            });
        }
        Ok(())
    }

    pub(crate) fn get_device_count(&self) -> u32 {
        self.gpu_idx_uuids.len() as u32
    }

    /// ERL workload-aware admission control
    fn try_erl_admission_control(
        &self,
        raw_device_index: usize,
        grids: u32,
        blocks: u32,
    ) -> Result<bool, Error> {
        let limiter = self.get_or_init_kernel_limiter()?;

        // Get current state before trying to acquire
        let current_tokens = limiter.current_tokens(raw_device_index).unwrap_or(0.0);

        match limiter.try_acquire(raw_device_index, grids, blocks) {
            Ok(true) => {
                let tokens_after = limiter.current_tokens(raw_device_index).unwrap_or(0.0);
                let cost = current_tokens - tokens_after;
                tracing::debug!(
                    device_idx = raw_device_index,
                    grids,
                    blocks,
                    tokens_before = %format!("{:.1}", current_tokens),
                    tokens_after = %format!("{:.1}", tokens_after),
                    cost = %format!("{:.1}", cost),
                    "ERL: Admission GRANTED"
                );
                Ok(true)
            }
            Ok(false) => {
                tracing::info!(
                    device_idx = raw_device_index,
                    grids,
                    blocks,
                    tokens = %format!("{:.1}", current_tokens),
                    "ERL: Admission DENIED (insufficient tokens)"
                );
                Ok(false)
            }
            Err(err) => {
                tracing::error!(
                    device_idx = raw_device_index,
                    grids,
                    blocks,
                    error = %err,
                    "ERL admission check failed"
                );
                Err(Error::ErlAdmissionFailed {
                    device_idx: raw_device_index,
                    message: err.to_string(),
                })
            }
        }
    }

    /// Get pod memory usage from shared memory
    pub(crate) fn get_pod_memory_usage(
        &self,
        raw_device_index: usize,
    ) -> Result<(u64, u64), Error> {
        let handle = self.get_or_init_shared_memory()?;
        let state = handle.get_state();

        if !state.is_healthy(Duration::from_secs(2)) {
            let last_heartbeat = state.get_last_heartbeat();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let err = Error::DeviceNotHealthy {
                device_idx: raw_device_index,
                last_heartbeat,
            };
            tracing::warn!(now = now, "{err}");
        }

        if let Some((used, limit)) = state.with_device(
            raw_device_index,
            |device| {
                (
                    device.device_info.get_pod_memory_used(),
                    device.device_info.get_mem_limit(),
                )
            },
            |device| {
                (
                    device.device_info.get_pod_memory_used(),
                    device.device_info.get_mem_limit(),
                )
            },
        ) {
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

    /// Check if the limiter is a compute shard
    pub(crate) fn is_compute_shard(&self) -> bool {
        self.compute_shard
    }

    pub(crate) fn isolation(&self) -> Option<&str> {
        self.isolation.as_deref()
    }

    /// Check if all devices have unlimited resources (both up_limit >= 100 and mem_limit == total_memory)
    pub(crate) fn all_devices_unlimited(&self) -> bool {
        let handle = match self.get_or_init_shared_memory() {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!("Failed to access shared memory: {}", e);
                return false;
            }
        };

        let state = handle.get_state();

        for (idx, _uuid) in &self.gpu_idx_uuids {
            let (up_limit, mem_limit) = state
                .with_device(
                    *idx,
                    |device| {
                        (
                            device.device_info.get_up_limit(),
                            device.device_info.get_mem_limit(),
                        )
                    },
                    |device| {
                        (
                            device.device_info.get_up_limit(),
                            device.device_info.get_mem_limit(),
                        )
                    },
                )
                .unwrap_or((0, 0));

            if up_limit < 100 {
                return false;
            }

            let device_total_memory = match self.nvml.device_by_index(*idx as u32) {
                Ok(device) => match device.memory_info() {
                    Ok(info) => info.total,
                    Err(e) => {
                        tracing::warn!("Failed to get memory info for device {}: {}", idx, e);
                        return false;
                    }
                },
                Err(e) => {
                    tracing::warn!("Failed to get device {} by index: {}", idx, e);
                    return false;
                }
            };

            if mem_limit < device_total_memory {
                return false;
            }
        }

        true
    }
}

fn shm_path() -> PathBuf {
    PathBuf::from(
        &std::env::var("SHM_PATH").unwrap_or_else(|_| "/run/tensor-fusion/shm".to_string()),
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
}
