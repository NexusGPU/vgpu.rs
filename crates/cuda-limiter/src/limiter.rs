use std::sync::atomic::AtomicI32;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::thread::sleep;
use std::time::Duration;

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::CudaContext;
use cudarc::driver::DriverError;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;
use nvml_wrapper_sys::bindings::nvmlDevice_t;
use thiserror::Error;
use trap::TrapError;

use crate::detour::CUdevice;

// Configuration constant
const FACTOR: u32 = 32;
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

    #[error("Invalid CUDA device: {0}")]
    InvalidCuDevice(CUdevice),

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
    /// Total available memory in bytes
    total_mem: u64,

    cu_device: CUdevice,
    /// Block dimensions set by cuFuncSetBlockShape
    pub(crate) block_x: AtomicU32,
    pub(crate) block_y: AtomicU32,
    pub(crate) block_z: AtomicU32,

    /// Condition variable for signaling when CUDA cores become available
    pub(crate) cores_condvar: Arc<(Mutex<()>, Condvar)>,
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
#[derive(Debug)]
pub(crate) struct LimiterBuilder {
    pid: Option<u32>,
    device_configs: Vec<DeviceConfig>,
    nvml: Nvml,
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
    pub(crate) fn new(nvml: Nvml) -> Self {
        Self {
            pid: None,
            device_configs: Vec::new(),
            nvml,
        }
    }

    /// Set the process ID to monitor
    pub(crate) fn with_pid(mut self, pid: u32) -> Self {
        self.pid = Some(pid);
        self
    }

    /// Add a device configuration
    pub(crate) fn with_device_config(mut self, config: DeviceConfig) -> Self {
        self.device_configs.push(config);
        self
    }

    /// Build the Limiter
    pub(crate) fn build(self) -> Result<Limiter, Error> {
        let pid = self.pid.unwrap_or_else(std::process::id);

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

                let total_mem =
                    unsafe { cudarc::driver::result::device::total_mem(ctx.cu_device()) }? as u64;
                let total_cuda_cores = sm_count * max_thread_per_sm * FACTOR;
                self.nvml
                    .device_by_index(devices.len() as u32)
                    .map_err(Error::Nvml)?;
                tracing::info!(
                    "Device {}: sm_count: {}, max_thread_per_sm: {}, mem_limit: {} bytes, total_mem: {} bytes",
                    devices.len(),
                    sm_count,
                    max_thread_per_sm,
                    config.mem_limit,
                    total_mem
                );

                let device_info = DeviceInfo {
                    sm_count,
                    max_thread_per_sm,
                    total_cuda_cores,
                    available_cuda_cores: AtomicI32::new(0),
                    up_limit: AtomicU32::new(config.up_limit),
                    mem_limit: AtomicU64::new(config.mem_limit),
                    total_mem,
                    cu_device: ctx.cu_device(),
                    block_x: AtomicU32::new(0),
                    block_y: AtomicU32::new(0),
                    block_z: AtomicU32::new(0),
                    cores_condvar: Arc::new((Mutex::new(()), Condvar::new())),
                };

                devices.push(device_info);
            }
        }

        Ok(Limiter {
            nvml: self.nvml,
            pid,
            devices,
        })
    }
}

impl Limiter {
    /// Create a new Limiter with the given process ID and device configurations
    pub(crate) fn new(
        pid: u32,
        nvml: Nvml,
        device_configs: &[DeviceConfig],
    ) -> Result<Self, Error> {
        let mut builder = LimiterBuilder::new(nvml).with_pid(pid);

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

        // Wait for available CUDA cores using condition variable
        let &(ref lock, ref cvar) = &*device.cores_condvar;
        let mut guard = lock.lock().unwrap();

        // Check if cores are already available
        while device.available_cuda_cores.load(Ordering::Acquire) <= 0 {
            // Wait for notification that cores are available
            guard = cvar.wait(guard).unwrap();
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
        let dev = self.nvml.device_by_index(device_idx)?;
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

        let mut last_seen_timestamp = 0;
        loop {
            // Sleep for the specified duration
            sleep(watch_duration);

            // Get current utilization with retry logic
            let (util, newest_timestamp) =
                self.get_utilization_with_retry(device_idx, watch_duration, last_seen_timestamp)?;

            last_seen_timestamp = newest_timestamp;
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
        last_seen_timestamp: u64,
    ) -> Result<(Utilization, u64), Error> {
        let max_retries = 5;
        let mut retry_count = 0;
        loop {
            match self.get_used_gpu_utilization(device_idx, last_seen_timestamp) {
                Ok((Some(util), newest_timestamp_candidate)) => {
                    return Ok((util, newest_timestamp_candidate))
                }
                Ok((None, _)) => {
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
        let was_zero = available_cores <= 0;

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

        // If cores were previously unavailable (zero) and now they're available,
        // or if we're adding cores (positive adjustment), notify waiting threads
        if (was_zero && new_value > 0) || adjustment > 0 {
            // Notify all waiting threads that cores are available
            let &(ref _lock, ref cvar) = &*device.cores_condvar;
            cvar.notify_all();
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
        } else if (share - increment) < 0 {
            Ok(0)
        } else {
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
    pub(crate) fn get_mem_limit_cu(&self, cu_device: CUdevice) -> Result<u64, Error> {
        let device = self
            .devices
            .iter()
            .find(|d| d.cu_device == cu_device)
            .ok_or(Error::InvalidCuDevice(cu_device))?;
        let mem_limit = device.mem_limit.load(Ordering::Acquire);
        Ok(if mem_limit == 0 {
            device.total_mem
        } else {
            mem_limit
        })
    }

    /// Get the memory limit for a specific device
    pub(crate) fn get_mem_limit(&self, device_idx: u32) -> Result<u64, Error> {
        let device = self.get_device(device_idx)?;
        let mem_limit = device.mem_limit.load(Ordering::Acquire);
        Ok(if mem_limit == 0 {
            device.total_mem
        } else {
            mem_limit
        })
    }

    /// Get the NVML device handle for a specific device
    pub(crate) fn device_idx_by_handle(&self, device_handle: nvmlDevice_t) -> Option<u32> {
        for idx in 0..self.devices.len() {
            let dev = self.nvml.device_by_index(idx as u32);
            match dev {
                Ok(dev) => {
                    let handle = unsafe { dev.handle() };
                    if handle == device_handle {
                        return Some(idx as u32);
                    }
                }
                Err(e) => {
                    tracing::warn!("failed to get device by index {}: {}, skipped", idx, e);
                    continue;
                }
            }
        }
        None
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

    /// get GPU utilization statistics for a specific device
    ///
    /// # Parameters
    /// * `device_idx` - Device index
    /// * `last_seen_timestamp` - Last known timestamp (for incremental fetching)
    ///
    /// # Returns
    /// * `Ok((Some(Utilization), timestamp))` - Successfully retrieved utilization data and latest timestamp
    /// * `Ok((None, timestamp))` - No available utilization data, but returned latest timestamp
    /// * `Err(NvmlError)` - Error occurred while fetching utilization data
    fn get_used_gpu_utilization(
        &self,
        device_idx: u32,
        last_seen_timestamp: u64,
    ) -> Result<(Option<Utilization>, u64), NvmlError> {
        let mut newest_timestamp_candidate = last_seen_timestamp;
        // Get the device
        let dev = self.nvml.device_by_index(device_idx)?;
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

            // Collect the largest valid timestamp for this device to filter out
            // the samples during the next call to the function
            // nvmlDeviceGetProcessUtilization
            if sample.timestamp > newest_timestamp_candidate {
                newest_timestamp_candidate = sample.timestamp;
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
            Ok((None, newest_timestamp_candidate))
        } else {
            Ok((Some(current), newest_timestamp_candidate))
        }
    }
}

const fn codec_normalize(x: u32) -> u32 {
    x * 85 / 100
}

#[cfg(test)]
mod tests {
    use std::ffi;
    use std::path;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
    use std::time::SystemTime;

    use super::*;

    #[test]
    fn test_rate_limiter() {
        // Skip test if NVIDIA drivers are not available
        if !path::Path::new("/dev/nvidia0").exists() {
            println!("Skipping test_rate_limiter: NVIDIA device not found");
            return;
        }

        // Create a limiter instance
        let pid = std::process::id();
        let device_config = DeviceConfig {
            device_idx: 0,
            up_limit: 80,
            mem_limit: 1024 * 1024 * 1024, // 1GB
        };

        // Initialize NVML
        let nvml = match Nvml::init() {
            Ok(nvml) => nvml,
            Err(_) => Nvml::builder()
                .lib_path(ffi::OsStr::new("libnvidia-ml.so.1"))
                .init()
                .expect("Failed to initialize NVML"),
        };

        // Create a shared limiter that can be used across threads
        let limiter = Arc::new(Limiter::new(pid, nvml, &[device_config]).unwrap());

        // Get the device and set available cores to 0 initially
        let device = limiter.get_device(0).unwrap();
        device.available_cuda_cores.store(0, Ordering::Release);

        // Set up a flag to track if the rate_limiter call completed
        let completed = Arc::new(AtomicBool::new(false));

        // Start a thread that will call rate_limiter
        let handle = thread::spawn({
            let limiter = limiter.clone();
            let completed = completed.clone();
            move || {
                // This should block until cores are available
                limiter.rate_limiter(0, 10, 1);
                completed.store(true, Ordering::SeqCst);
            }
        });

        // Wait a short time to ensure the thread has started
        thread::sleep(Duration::from_millis(50));

        // At this point, the thread should be blocked in the rate_limiter loop
        // Check that it hasn't completed yet
        assert!(!completed.load(Ordering::SeqCst));

        // Now make cores available to unblock the thread
        device.available_cuda_cores.store(100, Ordering::Release);
        let &(ref _lock, ref cvar) = &*device.cores_condvar;
        cvar.notify_all();

        // Wait for the thread to complete or timeout
        let timeout = Duration::from_secs(2);
        let start = SystemTime::now();

        while !completed.load(Ordering::SeqCst) {
            if SystemTime::now().duration_since(start).unwrap() > timeout {
                panic!("Test timed out waiting for rate_limiter to complete");
            }
            thread::sleep(Duration::from_millis(10));
        }

        // If we got here, the rate_limiter completed
        handle.join().unwrap();

        // Check if cores were subtracted
        assert_eq!(device.available_cuda_cores.load(Ordering::Acquire), 90);
    }
}
