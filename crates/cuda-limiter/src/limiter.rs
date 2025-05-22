use cudarc::driver::{sys::CUdevice_attribute, CudaContext, DriverError};
use nvml_wrapper::{enums::device::UsedGpuMemory, error::NvmlError, Nvml};
use std::{
    sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering},
    thread::sleep,
    time::{Duration, SystemTime},
};
use thiserror::Error;
use trap::TrapError;

use crate::detour::NvmlDeviceT;

// Configuration constant
const FACTOR: u32 = 1;
// Default sleep duration when waiting for resources
const DEFAULT_WAIT_SLEEP_MS: u64 = 10;
// Default backoff multiplier for retries
const DEFAULT_BACKOFF_MULTIPLIER: u32 = 2;

#[derive(Error, Debug)]
pub(crate) enum Error {
    #[error("NVML error: `{0}`")]
    Nvml(#[from] NvmlError),

    #[error("CUDA driver error: `{0}`")]
    CuDriver(#[from] DriverError),

    #[error("Trap error: `{0}`")]
    Trap(#[from] TrapError),

    #[error("Invalid device index: {0}")]
    InvalidDevice(u32),

    #[error("Resource not available: {0}")]
    ResourceNotAvailable(String),
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
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_idx: 0,
            up_limit: 80, // 80% default utilization limit
            mem_limit: 0, // No memory limit by default
        }
    }
}

/// Internal device information and state
#[derive(Debug)]
struct DeviceInfo {
    /// Number of streaming multiprocessors
    sm_count: u32,
    /// Maximum threads per streaming multiprocessor
    max_thread_per_sm: u32,
    /// Total CUDA cores available
    total_cuda_cores: u32,
    /// Currently available CUDA cores
    available_cuda_cores: AtomicI32,
    /// Utilization percentage limit
    up_limit: AtomicU32,
    /// Memory limit in bytes
    mem_limit: AtomicU64,
    /// Block dimensions set by cuFuncSetBlockShape
    pub(crate) block_x: AtomicU32,
    pub(crate) block_y: AtomicU32,
    pub(crate) block_z: AtomicU32,
}

/// Main limiter struct that manages CUDA resource limits
#[derive(Debug)]
pub(crate) struct Limiter {
    /// NVML interface for GPU monitoring
    nvml: Nvml,
    /// Process ID being monitored
    pub(crate) pid: u32,
    /// Information about each device
    devices: Vec<DeviceInfo>,
}

/// Builder for creating a Limiter with custom configuration
#[derive(Debug, Default)]
pub(crate) struct LimiterBuilder {
    pid: Option<u32>,
    device_configs: Vec<DeviceConfig>,
    nvml_lib_path: Option<String>,
}

/// GPU utilization statistics
#[derive(Debug, Default, Clone, Copy)]
struct Utilization {
    /// Current utilization by this process
    user_current: u32,
    /// Current system-wide utilization
    sys_current: u32,
    /// Number of processes using the GPU
    sys_process_num: u32,
}

impl LimiterBuilder {
    /// Create a new LimiterBuilder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the process ID to monitor
    pub fn with_pid(mut self, pid: u32) -> Self {
        self.pid = Some(pid);
        self
    }

    /// Add a device configuration
    pub fn with_device_config(mut self, config: DeviceConfig) -> Self {
        self.device_configs.push(config);
        self
    }

    /// Set the NVML library path
    #[allow(dead_code)]
    pub fn with_nvml_lib_path(mut self, path: impl Into<String>) -> Self {
        self.nvml_lib_path = Some(path.into());
        self
    }

    /// Build the Limiter
    pub fn build(self) -> Result<Limiter, Error> {
        let pid = self.pid.unwrap_or_else(std::process::id);

        // Initialize NVML
        let nvml = match Nvml::init() {
            Ok(nvml) => Ok(nvml),
            Err(_) => {
                let lib_path = self
                    .nvml_lib_path
                    .unwrap_or_else(|| "libnvidia-ml.so.1".to_string());
                Nvml::builder()
                    .lib_path(std::ffi::OsStr::new(&lib_path))
                    .init()
            }
        }?;

        let mut devices = vec![];
        let configs = if self.device_configs.is_empty() {
            vec![DeviceConfig::default()]
        } else {
            self.device_configs
        };

        for config in configs {
            // Fill any gaps in the device indices
            while devices.len() < config.device_idx as usize + 1 {
                let ctx = CudaContext::new(devices.len())?;

                let sm_count = ctx
                    .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?
                    as u32;
                let max_thread_per_sm = ctx.attribute(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                )? as u32;

                let total_cuda_cores = sm_count * max_thread_per_sm * FACTOR;

                tracing::info!(
                    "Device {}: sm_count: {}, max_thread_per_sm: {}, mem_limit: {} bytes",
                    devices.len(),
                    sm_count,
                    max_thread_per_sm,
                    config.mem_limit
                );

                let device_info = DeviceInfo {
                    sm_count,
                    max_thread_per_sm,
                    total_cuda_cores,
                    available_cuda_cores: AtomicI32::new(0),
                    up_limit: AtomicU32::new(config.up_limit),
                    mem_limit: AtomicU64::new(config.mem_limit),
                    block_x: AtomicU32::new(0),
                    block_y: AtomicU32::new(0),
                    block_z: AtomicU32::new(0),
                };

                devices.push(device_info);
            }
        }

        Ok(Limiter { nvml, pid, devices })
    }
}

impl Limiter {
    /// Create a new Limiter with the given process ID and device configurations
    pub fn new(pid: u32, device_configs: &[DeviceConfig]) -> Result<Self, Error> {
        let mut builder = LimiterBuilder::new().with_pid(pid);

        for config in device_configs {
            builder = builder.with_device_config(config.clone());
        }

        builder.build()
    }

    /// Helper function to get a device by index with proper error handling
    fn get_device(&self, device_idx: u32) -> Result<&DeviceInfo, Error> {
        self.devices
            .get(device_idx as usize)
            .ok_or(Error::InvalidDevice(device_idx))
    }
    /// Set the utilization limit for a specific device
    pub(crate) fn set_uplimit(&self, device_idx: u32, up_limit: u32) -> Result<(), Error> {
        let device = self.get_device(device_idx)?;
        device.up_limit.store(up_limit, Ordering::Release);
        Ok(())
    }

    /// Set the memory limit for a specific device
    pub(crate) fn set_mem_limit(&self, device_idx: u32, mem_limit: u64) -> Result<(), Error> {
        let device = self.get_device(device_idx)?;
        device.mem_limit.store(mem_limit, Ordering::Release);
        Ok(())
    }

    /// Limit the rate of kernel execution for a specific device
    pub(crate) fn rate_limiter(&self, device_idx: u32, grids: u32, _blocks: u32) {
        let device = self.get_device(device_idx).expect("get device");
        let kernel_size = grids;

        // Wait for available CUDA cores
        loop {
            if device.available_cuda_cores.load(Ordering::Acquire) > 0 {
                break;
            }
            sleep(Duration::from_millis(DEFAULT_WAIT_SLEEP_MS));
        }

        // Subtract the used cores
        device
            .available_cuda_cores
            .fetch_sub(kernel_size as i32, Ordering::Release);
    }

    /// Get the amount of GPU memory used by this process on a specific device
    pub(crate) fn get_used_gpu_memory(&self, device_idx: u32) -> Result<u64, Error> {
        // Validate device index
        if device_idx as usize >= self.devices.len() {
            return Err(Error::InvalidDevice(device_idx));
        }

        // Get device and process information
        let dev = self.nvml.device_by_index(device_idx).map_err(Error::from)?;

        let process_info = dev.running_compute_processes().map_err(|err| {
            tracing::warn!("Failed to get running compute processes: {:?}", err);
            Error::Nvml(NvmlError::Unknown)
        })?;

        // Find this process in the list
        for pi in process_info {
            if pi.pid == self.pid {
                return match pi.used_gpu_memory {
                    UsedGpuMemory::Used(bytes) => Ok(bytes),
                    UsedGpuMemory::Unavailable => {
                        tracing::warn!("GPU memory usage unavailable: NVML_VALUE_NOT_AVAILABLE");
                        Ok(0) // Return 0 when unavailable instead of breaking
                    }
                };
            }
        }

        // Process not found in the list
        Ok(0)
    }

    /// Run a watcher thread that adjusts available CUDA cores based on utilization
    ///
    /// This function runs in an infinite loop and should be called in a separate thread
    pub(crate) fn run_watcher(
        &self,
        device_idx: u32,
        watch_duration: Duration,
    ) -> Result<(), Error> {
        let device = self.get_device(device_idx)?;
        let mut share: i32 = 0;

        loop {
            // Sleep for the specified duration
            sleep(watch_duration);

            // Get current utilization with retry logic
            let util = self.get_utilization_with_retry(device_idx, watch_duration)?;

            // Get current available cores
            let available_cuda_cores = device.available_cuda_cores.load(Ordering::Acquire);

            // Calculate adjustment based on utilization
            share = self.delta(device_idx, util.user_current, share)?;

            // Apply the adjustment with bounds checking
            self.apply_core_adjustment(device, available_cuda_cores, share);
        }
    }

    /// Get utilization with retry logic
    fn get_utilization_with_retry(
        &self,
        device_idx: u32,
        retry_interval: Duration,
    ) -> Result<Utilization, Error> {
        let max_retries = 5;
        let mut retry_count = 0;

        loop {
            match self.get_used_gpu_utilization(device_idx) {
                Ok(Some(util)) => return Ok(util),
                Ok(None) => {
                    retry_count += 1;
                    if retry_count >= max_retries {
                        return Err(Error::ResourceNotAvailable(
                            "GPU utilization data not available after retries".to_string(),
                        ));
                    }
                    sleep(retry_interval);
                }
                Err(NvmlError::NotFound) => {
                    // not found means this gpu is not using by any process
                    sleep(retry_interval * DEFAULT_BACKOFF_MULTIPLIER);
                    continue;
                }
                Err(err) => {
                    tracing::warn!("failed to get_used_gpu_utilization, err: {err}");
                    retry_count += 1;
                    if retry_count >= max_retries {
                        return Err(Error::Nvml(err));
                    }
                    sleep(retry_interval * DEFAULT_BACKOFF_MULTIPLIER);
                }
            }
        }
    }

    /// Apply core adjustment with bounds checking
    fn apply_core_adjustment(&self, device: &DeviceInfo, available_cores: i32, adjustment: i32) {
        let total_cores = device.total_cuda_cores as i32;
        let new_value = available_cores + adjustment;

        if new_value >= total_cores {
            // Cap at maximum
            device
                .available_cuda_cores
                .store(total_cores, Ordering::Release);
        } else if new_value <= 0 {
            // Don't go below zero
            device.available_cuda_cores.store(0, Ordering::Release);
        } else {
            // Apply the adjustment
            device
                .available_cuda_cores
                .fetch_add(adjustment, Ordering::Release);
        }
    }

    /// Calculate the delta adjustment for available CUDA cores based on current utilization
    ///
    /// # Arguments
    /// * `device_idx` - The device index
    /// * `user_current` - Current utilization percentage
    /// * `share` - Current share value
    ///
    /// # Returns
    /// The new share value after adjustment
    fn delta(&self, device_idx: u32, user_current: u32, share: i32) -> Result<i32, Error> {
        let device = self.get_device(device_idx)?;
        let up_limit = device.up_limit.load(Ordering::Acquire) as i32;

        // Calculate difference between target and current utilization
        // Use a minimum difference of 5 to avoid tiny adjustments
        let utilization_diff = if (up_limit - user_current as i32).abs() < 5 {
            5
        } else {
            (up_limit - user_current as i32).abs()
        };

        // Calculate the increment based on device characteristics and utilization difference
        let increment = self.calculate_increment(device, utilization_diff, up_limit);

        // Determine if we should increase or decrease the share
        if user_current <= up_limit as u32 {
            // Utilization is below limit, increase share (but don't exceed total cores)
            let total_cuda_cores = device.total_cuda_cores as i32;
            if share + increment > total_cuda_cores {
                Ok(total_cuda_cores)
            } else {
                Ok(share + increment)
            }
        } else {
            // Utilization is above limit, decrease share
            Ok(share - increment)
        }
    }

    /// Calculate the increment value for delta adjustment
    fn calculate_increment(
        &self,
        device: &DeviceInfo,
        utilization_diff: i32,
        up_limit: i32,
    ) -> i32 {
        let mut increment =
            device.sm_count as i32 * device.sm_count as i32 * device.max_thread_per_sm as i32 / 256
                * utilization_diff
                / 10;

        // Apply additional scaling when difference is large
        if utilization_diff > up_limit / 2 {
            increment = increment * utilization_diff * 2 / (up_limit + 1);
        }

        increment
    }

    /// Get the memory limit for a specific device
    /// Returns u64::MAX if no limit is set (mem_limit = 0)
    pub(crate) fn get_mem_limit(&self, device_idx: u32) -> Result<u64, Error> {
        let device = self.get_device(device_idx)?;
        let mem_limit = device.mem_limit.load(Ordering::Acquire);
        Ok(if mem_limit == 0 { u64::MAX } else { mem_limit })
    }

    /// Get the NVML device handle for a specific device
    pub(crate) fn device_handle(&self, device_idx: u32) -> Result<NvmlDeviceT, Error> {
        // Validate device index
        if device_idx as usize >= self.devices.len() {
            return Err(Error::InvalidDevice(device_idx));
        }

        // Get device handle
        self.nvml
            .device_by_index(device_idx)
            .map(|dev| unsafe { dev.handle() as NvmlDeviceT })
            .map_err(Error::from)
    }

    /// Get the block dimensions for a specific device
    pub(crate) fn get_block_dimensions(&self, device_idx: u32) -> Result<(u32, u32, u32), Error> {
        if device_idx as usize >= self.devices.len() {
            return Err(Error::InvalidDevice(device_idx));
        }

        let device = &self.devices[device_idx as usize];
        Ok((
            device.block_x.load(Ordering::Acquire),
            device.block_y.load(Ordering::Acquire),
            device.block_z.load(Ordering::Acquire),
        ))
    }

    /// Set the block dimensions for a specific device
    pub(crate) fn set_block_dimensions(
        &self,
        device_idx: u32,
        x: u32,
        y: u32,
        z: u32,
    ) -> Result<(), Error> {
        if device_idx as usize >= self.devices.len() {
            return Err(Error::InvalidDevice(device_idx));
        }

        let device = &self.devices[device_idx as usize];
        device.block_x.store(x, Ordering::Release);
        device.block_y.store(y, Ordering::Release);
        device.block_z.store(z, Ordering::Release);
        Ok(())
    }

    /// Get GPU utilization statistics for a specific device
    ///
    /// # Returns
    /// * `Ok(Some(Utilization))` - Utilization data is available
    /// * `Ok(None)` - No valid utilization data available
    /// * `Err(NvmlError)` - Error occurred while getting utilization data
    fn get_used_gpu_utilization(&self, device_idx: u32) -> Result<Option<Utilization>, NvmlError> {
        // Get the device
        let dev = self.nvml.device_by_index(device_idx)?;

        // Calculate timestamp for recent samples (last second)
        let last_seen_timestamp = unix_as_millis()
            .saturating_mul(1000) // Convert to microseconds
            .saturating_sub(1_000_000); // Look at last 1 second

        // Get process utilization samples
        let process_utilization_samples = dev.process_utilization_stats(last_seen_timestamp)?;

        // Initialize utilization counters
        let mut current = Utilization::default();
        let mut valid = false;

        // Get number of processes
        current.sys_process_num = dev.running_compute_processes_count()?;

        // Process each utilization sample
        for sample in process_utilization_samples {
            // Skip old samples
            if sample.timestamp < last_seen_timestamp {
                continue;
            }

            // Mark that we have valid data
            valid = true;

            // Calculate codec utilization
            let codec_util = codec_normalize(sample.enc_util + sample.dec_util);

            // Add to system-wide utilization
            current.sys_current += sample.sm_util;
            current.sys_current += codec_util;

            // Add to user process utilization if it's our process
            if sample.pid == self.pid {
                current.user_current += sample.sm_util;
                current.user_current += codec_util;
            }
        }

        // Return None if no valid data, otherwise return the utilization
        if !valid {
            Ok(None)
        } else {
            Ok(Some(current))
        }
    }

    /// Initialize a new Limiter with the given process ID and device configuration
    ///
    /// This is a compatibility method for the existing API in lib.rs
    pub(crate) fn init(
        pid: u32,
        device_idx: u32,
        up_limit: u32,
        mem_limit: u64,
    ) -> Result<Self, Error> {
        // Create a DeviceConfig from the parameters
        let config = DeviceConfig {
            device_idx,
            up_limit,
            mem_limit,
        };

        // Pass a slice with the single config
        Self::new(pid, &[config])
    }
}

const fn codec_normalize(x: u32) -> u32 {
    x * 85 / 100
}

pub(crate) fn unix_as_millis() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
