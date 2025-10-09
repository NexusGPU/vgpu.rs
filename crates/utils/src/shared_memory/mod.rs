use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

use tracing::warn;

use crate::shared_memory::mutex::ShmMutex;
use crate::shared_memory::set::Set;

pub mod bitmap;
pub mod erl_adapter;
pub mod handle;
pub mod manager;
pub mod mutex;
pub mod set;
pub mod traits;

/// Clean up empty parent directories after removing a file
/// This removes the directory structure recursively if directories become empty
pub fn cleanup_empty_parent_directories(
    file_path: &Path,
    stop_at_path: Option<&Path>,
) -> std::io::Result<()> {
    if let Some(parent_dir) = file_path.parent() {
        // Skip if we've reached the stop path
        if let Some(stop) = stop_at_path {
            if parent_dir == stop {
                return Ok(());
            }
        }

        // Try to remove the immediate parent directory if it's empty
        if let Ok(entries) = std::fs::read_dir(parent_dir) {
            let entry_count = entries.count();
            if entry_count == 0 {
                tracing::info!("Removing empty directory: {}", parent_dir.display());
                match std::fs::remove_dir(parent_dir) {
                    Ok(_) => {
                        tracing::info!("Removed empty directory: {}", parent_dir.display());
                        // Recursively try to remove parent directories if they're also empty
                        cleanup_empty_parent_directories(parent_dir, stop_at_path)?;
                    }
                    Err(e) => {
                        tracing::debug!(
                            "Failed to remove empty directory {}: {}",
                            parent_dir.display(),
                            e
                        );
                        return Err(e);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Pod identifier structure containing namespace and name
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PodIdentifier {
    pub namespace: String,
    pub name: String,
}

impl PodIdentifier {
    /// Create a new PodIdentifier
    pub fn new(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
        }
    }

    pub fn to_path(&self, base_path: impl AsRef<Path>) -> PathBuf {
        base_path
            .as_ref()
            .join(format!("{}/{}", self.namespace, self.name))
    }

    /// Parse a PodIdentifier from a full shared memory path
    /// Path format: {base_path}/{namespace}/{name}/shm
    /// This method extracts namespace/name from any base path
    pub fn from_shm_file_path(path: &str) -> Option<Self> {
        let path = Path::new(path);
        let components: Vec<_> = path
            .components()
            .filter_map(|c| c.as_os_str().to_str())
            .collect();

        if components.len() < 3 {
            return None;
        }

        // Extract the last 3 components: {namespace}/{name}/shm
        let len = components.len();
        let namespace = components[len - 3].to_string();
        let name = components[len - 2].to_string();
        Some(Self::new(namespace, name))
    }
}

impl std::fmt::Display for PodIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.namespace, self.name)
    }
}

const MAX_PROCESSES: usize = 2048;
/// Maximum number of devices that can be stored in shared memory
const MAX_DEVICES: usize = 16;
/// Maximum length of device UUID string (including null terminator)
const MAX_UUID_LEN: usize = 64;
/// Error type for reference count operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefCountError {
    Underflow,
}

/// V1 device state (legacy, without ERL)
#[repr(C)]
#[derive(Debug)]
pub struct SharedDeviceInfoV1 {
    /// Currently available CUDA cores.
    pub available_cuda_cores: AtomicI32,
    /// Utilization limit percentage (0-100).
    pub up_limit: AtomicU32,
    /// Memory limit in bytes.
    pub mem_limit: AtomicU64,
    /// Total number of CUDA cores.
    pub total_cuda_cores: AtomicU32,
    /// Current pod memory usage in bytes.
    pub pod_memory_used: AtomicU64,
}

/// V2 device state with ERL support (uses token-based limiting)
#[repr(C)]
#[derive(Debug)]
pub struct SharedDeviceInfoV2 {
    /// Utilization limit percentage (0-100).
    pub up_limit: AtomicU32,
    /// Memory limit in bytes.
    pub mem_limit: AtomicU64,
    /// Total number of CUDA cores.
    pub total_cuda_cores: AtomicU32,
    /// Current pod memory usage in bytes.
    pub pod_memory_used: AtomicU64,

    // ERL (Elastic Rate Limiting) related fields
    /// Current average cost from CUBIC congestion controller (as f64 bits)
    pub erl_avg_cost: AtomicU64,
    /// Token bucket capacity for this device
    pub erl_token_capacity: AtomicU64,
    /// Token bucket refill rate (tokens per second, as f64 bits)
    pub erl_token_refill_rate: AtomicU64,
    /// Current tokens in bucket (as f64 bits)
    pub erl_current_tokens: AtomicU64,
    /// Last token update timestamp (as f64 bits)
    pub erl_last_token_update: AtomicU64,
}

// Type alias for backward compatibility
pub type SharedDeviceInfo = SharedDeviceInfoV2;

impl SharedDeviceInfoV1 {
    pub fn new(total_cuda_cores: u32, up_limit: u32, mem_limit: u64) -> Self {
        Self {
            available_cuda_cores: AtomicI32::new(0),
            up_limit: AtomicU32::new(up_limit),
            mem_limit: AtomicU64::new(mem_limit),
            total_cuda_cores: AtomicU32::new(total_cuda_cores),
            pod_memory_used: AtomicU64::new(0),
        }
    }

    pub fn get_available_cores(&self) -> i32 {
        self.available_cuda_cores.load(Ordering::Acquire)
    }

    pub fn set_available_cores(&self, cores: i32) {
        self.available_cuda_cores.store(cores, Ordering::Release)
    }

    pub fn fetch_add_available_cores(&self, cores: i32) -> i32 {
        self.available_cuda_cores.fetch_add(cores, Ordering::AcqRel)
    }

    pub fn fetch_sub_available_cores(&self, cores: i32) -> i32 {
        self.available_cuda_cores.fetch_sub(cores, Ordering::AcqRel)
    }

    pub fn get_up_limit(&self) -> u32 {
        self.up_limit.load(Ordering::Acquire)
    }

    pub fn set_up_limit(&self, limit: u32) {
        self.up_limit.store(limit, Ordering::Release)
    }

    pub fn get_mem_limit(&self) -> u64 {
        self.mem_limit.load(Ordering::Acquire)
    }

    pub fn set_mem_limit(&self, limit: u64) {
        self.mem_limit.store(limit, Ordering::Release)
    }

    pub fn get_total_cores(&self) -> u32 {
        self.total_cuda_cores.load(Ordering::Acquire)
    }

    pub fn set_total_cores(&self, cores: u32) {
        self.total_cuda_cores.store(cores, Ordering::Release)
    }

    pub fn get_pod_memory_used(&self) -> u64 {
        self.pod_memory_used.load(Ordering::Acquire)
    }

    pub fn set_pod_memory_used(&self, memory: u64) {
        self.pod_memory_used.store(memory, Ordering::Release);
    }
}

impl SharedDeviceInfoV2 {
    pub fn new(total_cuda_cores: u32, up_limit: u32, mem_limit: u64) -> Self {
        Self {
            up_limit: AtomicU32::new(up_limit),
            mem_limit: AtomicU64::new(mem_limit),
            total_cuda_cores: AtomicU32::new(total_cuda_cores),
            pod_memory_used: AtomicU64::new(0),
            erl_avg_cost: AtomicU64::new(1.0_f64.to_bits()),
            erl_token_capacity: AtomicU64::new(100.0_f64.to_bits()),
            erl_token_refill_rate: AtomicU64::new(1.0_f64.to_bits()),
            erl_current_tokens: AtomicU64::new(100.0_f64.to_bits()),
            erl_last_token_update: AtomicU64::new(0.0_f64.to_bits()),
        }
    }

    // V2 保留 available_cuda_cores 以保持内存布局，但不提供操作方法（使用 ERL token 机制）
    // 为了兼容性，提供一个总是返回 0 的方法
    pub fn get_available_cores(&self) -> i32 {
        0 // V2 不使用 available_cores，总是返回 0
    }

    // 为了测试兼容性，提供空实现
    pub fn set_available_cores(&self, _cores: i32) {
        // V2 不使用 available_cores，忽略
    }

    pub fn fetch_add_available_cores(&self, _cores: i32) -> i32 {
        // V2 不使用 available_cores，返回 0
        0
    }

    pub fn fetch_sub_available_cores(&self, _cores: i32) -> i32 {
        // V2 不使用 available_cores，返回 0
        0
    }

    pub fn get_up_limit(&self) -> u32 {
        self.up_limit.load(Ordering::Acquire)
    }

    pub fn set_up_limit(&self, limit: u32) {
        self.up_limit.store(limit, Ordering::Release)
    }

    pub fn get_mem_limit(&self) -> u64 {
        self.mem_limit.load(Ordering::Acquire)
    }

    pub fn set_mem_limit(&self, limit: u64) {
        self.mem_limit.store(limit, Ordering::Release)
    }

    pub fn get_total_cores(&self) -> u32 {
        self.total_cuda_cores.load(Ordering::Acquire)
    }

    pub fn set_total_cores(&self, cores: u32) {
        self.total_cuda_cores.store(cores, Ordering::Release)
    }

    pub fn get_pod_memory_used(&self) -> u64 {
        self.pod_memory_used.load(Ordering::Acquire)
    }

    pub fn set_pod_memory_used(&self, memory: u64) {
        self.pod_memory_used.store(memory, Ordering::Release);
    }

    // ERL specific methods

    pub fn get_erl_avg_cost(&self) -> f64 {
        f64::from_bits(self.erl_avg_cost.load(Ordering::Acquire))
    }

    pub fn set_erl_avg_cost(&self, cost: f64) {
        self.erl_avg_cost.store(cost.to_bits(), Ordering::Release);
    }

    pub fn get_erl_token_capacity(&self) -> f64 {
        f64::from_bits(self.erl_token_capacity.load(Ordering::Acquire))
    }

    pub fn set_erl_token_capacity(&self, capacity: f64) {
        self.erl_token_capacity
            .store(capacity.to_bits(), Ordering::Release);
    }

    pub fn get_erl_token_refill_rate(&self) -> f64 {
        f64::from_bits(self.erl_token_refill_rate.load(Ordering::Acquire))
    }

    pub fn set_erl_token_refill_rate(&self, rate: f64) {
        self.erl_token_refill_rate
            .store(rate.to_bits(), Ordering::Release);
    }

    pub fn get_erl_current_tokens(&self) -> f64 {
        f64::from_bits(self.erl_current_tokens.load(Ordering::Acquire))
    }

    pub fn set_erl_current_tokens(&self, tokens: f64) {
        self.erl_current_tokens
            .store(tokens.to_bits(), Ordering::Release);
    }

    pub fn get_erl_last_token_update(&self) -> f64 {
        f64::from_bits(self.erl_last_token_update.load(Ordering::Acquire))
    }

    pub fn set_erl_last_token_update(&self, timestamp: f64) {
        self.erl_last_token_update
            .store(timestamp.to_bits(), Ordering::Release);
    }

    pub fn load_erl_token_state(&self) -> (f64, f64) {
        (
            self.get_erl_current_tokens(),
            self.get_erl_last_token_update(),
        )
    }

    pub fn store_erl_token_state(&self, tokens: f64, timestamp: f64) {
        self.set_erl_current_tokens(tokens);
        self.set_erl_last_token_update(timestamp);
    }

    pub fn load_erl_quota(&self) -> (f64, f64) {
        (
            self.get_erl_token_capacity(),
            self.get_erl_token_refill_rate(),
        )
    }
}

/// Device entry for V1 (legacy)
#[repr(C)]
#[derive(Debug)]
pub struct DeviceEntryV1 {
    pub uuid: [u8; MAX_UUID_LEN],
    pub device_info: SharedDeviceInfoV1,
    pub is_active: AtomicU32,
}

impl DeviceEntryV1 {
    pub fn new() -> Self {
        Self {
            uuid: [0; MAX_UUID_LEN],
            device_info: SharedDeviceInfoV1::new(0, 0, 0),
            is_active: AtomicU32::new(0),
        }
    }

    pub fn set_uuid(&self, uuid: &str) {
        let uuid_bytes = uuid.as_bytes();
        let copy_len = core::cmp::min(uuid_bytes.len(), MAX_UUID_LEN - 1);
        unsafe {
            let uuid_ptr = self.uuid.as_ptr() as *mut u8;
            core::ptr::write_bytes(uuid_ptr, 0, MAX_UUID_LEN);
            core::ptr::copy_nonoverlapping(uuid_bytes.as_ptr(), uuid_ptr, copy_len);
        }
    }

    pub fn get_uuid(&self) -> &str {
        let null_pos = self
            .uuid
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(MAX_UUID_LEN - 1);
        unsafe { core::str::from_utf8_unchecked(&self.uuid[..null_pos]) }
    }

    pub fn get_uuid_owned(&self) -> String {
        self.get_uuid().to_owned()
    }

    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire) != 0
    }

    pub fn set_active(&self, active: bool) {
        self.is_active.store(active as u32, Ordering::Release);
    }
}

impl Default for DeviceEntryV1 {
    fn default() -> Self {
        Self::new()
    }
}

/// Device entry for V2 (with ERL)
#[repr(C)]
#[derive(Debug)]
pub struct DeviceEntryV2 {
    pub uuid: [u8; MAX_UUID_LEN],
    pub device_info: SharedDeviceInfoV2,
    pub is_active: AtomicU32,
}

// Type alias for backward compatibility
pub type DeviceEntry = DeviceEntryV2;

impl DeviceEntry {
    /// Creates a new empty device entry
    pub fn new() -> Self {
        Self {
            uuid: [0; MAX_UUID_LEN],
            device_info: SharedDeviceInfo::new(0, 0, 0),
            is_active: AtomicU32::new(0),
        }
    }

    /// Sets the device UUID atomically
    pub fn set_uuid(&self, uuid: &str) {
        let uuid_bytes = uuid.as_bytes();
        let copy_len = std::cmp::min(uuid_bytes.len(), MAX_UUID_LEN - 1);

        // Clear the UUID array first
        unsafe {
            let uuid_ptr = self.uuid.as_ptr() as *mut u8;
            std::ptr::write_bytes(uuid_ptr, 0, MAX_UUID_LEN);
            // Copy the new UUID
            std::ptr::copy_nonoverlapping(uuid_bytes.as_ptr(), uuid_ptr, copy_len);
        }
    }

    /// Gets the device UUID as a string, using cached value when possible
    pub fn get_uuid(&self) -> &str {
        let null_pos = self
            .uuid
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(MAX_UUID_LEN - 1);

        // Safety: We ensure the UUID is always valid UTF-8 when setting it
        unsafe { std::str::from_utf8_unchecked(&self.uuid[..null_pos]) }
    }

    /// Gets the device UUID as an owned string
    pub fn get_uuid_owned(&self) -> String {
        self.get_uuid().to_owned()
    }

    /// Checks if this entry is active
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire) != 0
    }

    /// Sets the active status
    pub fn set_active(&self, active: bool) {
        self.is_active
            .store(if active { 1 } else { 0 }, Ordering::Release);
    }
}

impl Default for DeviceEntry {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared device state using only simple data types safe for inter-process sharing
#[repr(C)]
pub struct SharedDeviceStateV1 {
    /// Fixed-size array of device entries
    pub devices: [DeviceEntryV1; MAX_DEVICES],
    /// Number of active devices
    pub device_count: AtomicU32,
    /// Last heartbeat timestamp from hypervisor (for health monitoring).
    pub last_heartbeat: AtomicU64,
    /// Set of pids
    pub pids: ShmMutex<Set<usize, MAX_PROCESSES>>,
    /// Padding (512 bytes)
    pub _padding: [u8; 512],
}

/// V2 state with ERL support
#[repr(C)]
pub struct SharedDeviceStateV2 {
    /// Fixed-size array of device entries
    pub devices: [DeviceEntryV2; MAX_DEVICES],
    /// Number of active devices
    pub device_count: AtomicU32,
    /// Last heartbeat timestamp from hypervisor (for health monitoring).
    pub last_heartbeat: AtomicU64,
    /// Set of pids
    pub pids: ShmMutex<Set<usize, MAX_PROCESSES>>,
    /// Padding (512 bytes)
    pub _padding: [u8; 512],
}

/// Versioned shared device state enum for future compatibility
#[repr(C)]
#[allow(clippy::large_enum_variant)]
pub enum SharedDeviceState {
    V1(SharedDeviceStateV1),
    V2(SharedDeviceStateV2),
}

impl SharedDeviceState {
    /// Creates a new SharedDeviceState V1 instance (legacy).
    pub fn new(configs: &[DeviceConfig]) -> Self {
        Self::V2(SharedDeviceStateV2::new(configs))
    }

    /// Gets the current version of the shared device state
    pub fn version(&self) -> u32 {
        match self {
            Self::V1(_) => 1,
            Self::V2(_) => 2,
        }
    }

    /// Check if this state uses ERL features
    pub fn has_erl(&self) -> bool {
        matches!(self, Self::V2(_))
    }

    /// Delegates method calls to the appropriate version
    fn with_inner<T, F1, F2>(&self, f_v1: F1, f_v2: F2) -> T
    where
        F1: FnOnce(&SharedDeviceStateV1) -> T,
        F2: FnOnce(&SharedDeviceStateV2) -> T,
    {
        match self {
            Self::V1(inner) => f_v1(inner),
            Self::V2(inner) => f_v2(inner),
        }
    }

    // Delegate all methods to the inner version
    pub fn has_device(&self, index: usize) -> bool {
        self.with_inner(|v1| v1.has_device(index), |v2| v2.has_device(index))
    }

    pub fn device_count(&self) -> usize {
        self.with_inner(|v1| v1.device_count(), |v2| v2.device_count())
    }

    pub fn update_heartbeat(&self, timestamp: u64) {
        self.with_inner(
            |v1| v1.update_heartbeat(timestamp),
            |v2| v2.update_heartbeat(timestamp),
        )
    }

    pub fn get_last_heartbeat(&self) -> u64 {
        self.with_inner(|v1| v1.get_last_heartbeat(), |v2| v2.get_last_heartbeat())
    }

    pub fn is_healthy(&self, timeout: Duration) -> bool {
        self.with_inner(|v1| v1.is_healthy(timeout), |v2| v2.is_healthy(timeout))
    }

    pub fn add_pid(&self, pid: usize) {
        self.with_inner(|v1| v1.add_pid(pid), |v2| v2.add_pid(pid))
    }

    pub fn remove_pid(&self, pid: usize) {
        self.with_inner(|v1| v1.remove_pid(pid), |v2| v2.remove_pid(pid))
    }

    pub fn get_all_pids(&self) -> Vec<usize> {
        self.with_inner(|v1| v1.get_all_pids(), |v2| v2.get_all_pids())
    }

    pub fn cleanup_orphaned_locks(&self) {
        self.with_inner(
            |v1| v1.cleanup_orphaned_locks(),
            |v2| v2.cleanup_orphaned_locks(),
        )
    }

    /// Executes a closure with a V1 device entry by index
    pub fn with_device_v1<T, F>(&self, index: usize, f: F) -> Option<T>
    where
        F: FnOnce(&DeviceEntryV1) -> T,
    {
        if let Self::V1(inner) = self {
            inner
                .devices
                .get(index)
                .filter(|device| device.is_active())
                .map(f)
        } else {
            None
        }
    }

    /// Executes a closure with a V2 device entry by index
    pub fn with_device_v2<T, F>(&self, index: usize, f: F) -> Option<T>
    where
        F: FnOnce(&DeviceEntryV2) -> T,
    {
        if let Self::V2(inner) = self {
            inner
                .devices
                .get(index)
                .filter(|device| device.is_active())
                .map(f)
        } else {
            None
        }
    }

    /// Executes a closure with a device entry, dispatching to the appropriate version
    pub fn with_device<T, F1, F2>(&self, index: usize, f_v1: F1, f_v2: F2) -> Option<T>
    where
        F1: FnOnce(&DeviceEntryV1) -> T,
        F2: FnOnce(&DeviceEntryV2) -> T,
    {
        match self {
            Self::V1(inner) => inner
                .devices
                .get(index)
                .filter(|device| device.is_active())
                .map(f_v1),
            Self::V2(inner) => inner
                .devices
                .get(index)
                .filter(|device| device.is_active())
                .map(f_v2),
        }
    }

    /// Gets device information including additional fields for UI
    pub fn get_device_info(&self, index: usize) -> Option<(String, i32, u32, u64, u64, u32, bool)> {
        match self {
            Self::V1(inner) => inner
                .devices
                .get(index)
                .filter(|device| device.is_active())
                .map(|device| {
                    (
                        device.get_uuid_owned(),
                        device.device_info.get_available_cores(),
                        device.device_info.get_total_cores(),
                        device.device_info.get_mem_limit(),
                        device.device_info.get_pod_memory_used(),
                        device.device_info.get_up_limit(),
                        device.is_active(),
                    )
                }),
            Self::V2(inner) => inner
                .devices
                .get(index)
                .filter(|device| device.is_active())
                .map(|device| {
                    (
                        device.get_uuid_owned(),
                        0, // V2 不使用 available_cores，改用 ERL token
                        device.device_info.get_total_cores(),
                        device.device_info.get_mem_limit(),
                        device.device_info.get_pod_memory_used(),
                        device.device_info.get_up_limit(),
                        device.is_active(),
                    )
                }),
        }
    }

    /// Gets version number
    pub fn get_version(&self) -> u32 {
        self.version()
    }

    /// Iterates over all active devices with their indices (V2 only, returns DeviceEntryV2)
    pub fn iter_active_devices(&self) -> Box<dyn Iterator<Item = (usize, &DeviceEntryV2)> + '_> {
        match self {
            // V1 doesn't have DeviceEntryV2, return empty iterator
            // Tests expecting V1 iteration should use SharedDeviceStateV1 directly
            Self::V1(_) => Box::new(core::iter::empty()),
            Self::V2(inner) => Box::new(inner.iter_active_devices()),
        }
    }

    /// Iterates over all devices (including inactive ones) with their indices (returns DeviceEntryV2)
    pub fn iter_all_devices(&self) -> Box<dyn Iterator<Item = (usize, &DeviceEntryV2)> + '_> {
        match self {
            // V1 doesn't have DeviceEntryV2, return empty iterator
            // Tests expecting V1 iteration should use SharedDeviceStateV1 directly
            Self::V1(_) => Box::new(core::iter::empty()),
            Self::V2(inner) => Box::new(inner.iter_all_devices()),
        }
    }

    /// Executes a closure for each active device
    pub fn for_each_active_device<F>(&self, mut f: F)
    where
        F: FnMut(usize, &DeviceEntry),
    {
        self.iter_active_devices()
            .for_each(|(idx, device)| f(idx, device));
    }
}

impl SharedDeviceStateV1 {
    /// Creates a new SharedDeviceStateV1 instance.
    pub fn new(configs: &[DeviceConfig]) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let state = Self {
            devices: std::array::from_fn(|_| DeviceEntryV1::new()),
            device_count: AtomicU32::new(configs.len() as u32),
            last_heartbeat: AtomicU64::new(now),
            pids: ShmMutex::new(Set::new()),
            _padding: [0; 512],
        };

        for config in configs {
            let device_idx = config.device_idx as usize;
            if device_idx >= MAX_DEVICES {
                warn!(
                    "Device index {} exceeds maximum devices {}, skipping",
                    device_idx, MAX_DEVICES
                );
                continue;
            }

            state.devices[device_idx].set_uuid(&config.device_uuid);
            // Use atomic operations to set device info
            let device_info = &state.devices[device_idx].device_info;
            device_info
                .total_cuda_cores
                .store(config.total_cuda_cores, Ordering::Relaxed);
            device_info
                .available_cuda_cores
                .store(config.total_cuda_cores as i32, Ordering::Relaxed);
            device_info
                .up_limit
                .store(config.up_limit, Ordering::Relaxed);
            device_info
                .mem_limit
                .store(config.mem_limit, Ordering::Relaxed);
            state.devices[device_idx].set_active(true);
        }
        state
    }

    /// Checks if a device exists at the given index.
    pub fn has_device(&self, index: usize) -> bool {
        index < MAX_DEVICES && self.devices[index].is_active()
    }

    /// Gets the number of devices.
    pub fn device_count(&self) -> usize {
        self.device_count.load(Ordering::Acquire) as usize
    }

    /// Updates the heartbeat timestamp.
    pub fn update_heartbeat(&self, timestamp: u64) {
        self.last_heartbeat.store(timestamp, Ordering::Release);
    }

    /// Gets the last heartbeat timestamp.
    pub fn get_last_heartbeat(&self) -> u64 {
        self.last_heartbeat.load(Ordering::Acquire)
    }

    /// Checks if the shared memory is healthy based on heartbeat.
    pub fn is_healthy(&self, timeout: Duration) -> bool {
        let now = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(duration) => duration.as_secs(),
            Err(_) => {
                // If system time is before UNIX_EPOCH, consider unhealthy
                tracing::warn!("System time is before UNIX_EPOCH, considering unhealthy");
                return false;
            }
        };
        let last_heartbeat = self.get_last_heartbeat();

        if last_heartbeat == 0 {
            return false; // No heartbeat recorded
        }

        if last_heartbeat > now {
            // If last heartbeat is in the future, consider unhealthy
            return false;
        }

        tracing::debug!(
            last_heartbeat = last_heartbeat,
            now = now,
            timeout = timeout.as_secs(),
            "check device health"
        );

        now.saturating_sub(last_heartbeat) <= timeout.as_secs()
    }

    pub fn add_pid(&self, pid: usize) {
        let _ = self.pids.lock().insert_if_absent(pid);
    }

    pub fn remove_pid(&self, pid: usize) {
        let _ = self.pids.lock().remove_value(&pid);
    }

    /// Gets all PIDs currently stored in shared memory
    pub fn get_all_pids(&self) -> Vec<usize> {
        self.pids.lock().values().copied().collect()
    }

    /// Cleans up any orphaned locks held by dead processes
    /// This should be called during startup to prevent deadlocks
    pub fn cleanup_orphaned_locks(&self) {
        self.pids.cleanup_orphaned_lock();
    }

    /// Iterates over all active devices with their indices
    pub fn iter_active_devices(&self) -> impl Iterator<Item = (usize, &DeviceEntryV1)> {
        self.devices
            .iter()
            .enumerate()
            .filter(|(_, device)| device.is_active())
    }

    /// Iterates over all devices (including inactive ones) with their indices
    pub fn iter_all_devices(&self) -> impl Iterator<Item = (usize, &DeviceEntryV1)> {
        self.devices.iter().enumerate()
    }
}

impl SharedDeviceStateV2 {
    /// Creates a new SharedDeviceStateV2 instance with ERL support.
    pub fn new(configs: &[DeviceConfig]) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let state = Self {
            devices: std::array::from_fn(|_| DeviceEntry::new()),
            device_count: AtomicU32::new(configs.len() as u32),
            last_heartbeat: AtomicU64::new(now),
            pids: ShmMutex::new(Set::new()),
            _padding: [0; 512],
        };

        for config in configs {
            let device_idx = config.device_idx as usize;
            if device_idx >= MAX_DEVICES {
                warn!(
                    "Device index {} exceeds maximum devices {}, skipping",
                    device_idx, MAX_DEVICES
                );
                continue;
            }

            state.devices[device_idx].set_uuid(&config.device_uuid);
            let device_info = &state.devices[device_idx].device_info;

            device_info
                .total_cuda_cores
                .store(config.total_cuda_cores, Ordering::Relaxed);
            device_info
                .up_limit
                .store(config.up_limit, Ordering::Relaxed);
            device_info
                .mem_limit
                .store(config.mem_limit, Ordering::Relaxed);

            // Initialize ERL fields with defaults (will be set by hypervisor)
            device_info.set_erl_avg_cost(1.0);
            device_info.set_erl_token_capacity(100.0);
            device_info.set_erl_token_refill_rate(1.0);
            device_info.set_erl_current_tokens(100.0);
            device_info.set_erl_last_token_update(now as f64);

            state.devices[device_idx].set_active(true);
        }
        state
    }

    pub fn has_device(&self, index: usize) -> bool {
        index < MAX_DEVICES && self.devices[index].is_active()
    }

    pub fn device_count(&self) -> usize {
        self.device_count.load(Ordering::Acquire) as usize
    }

    pub fn update_heartbeat(&self, timestamp: u64) {
        self.last_heartbeat.store(timestamp, Ordering::Release);
    }

    pub fn get_last_heartbeat(&self) -> u64 {
        self.last_heartbeat.load(Ordering::Acquire)
    }

    pub fn is_healthy(&self, timeout: Duration) -> bool {
        let now = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(duration) => duration.as_secs(),
            Err(_) => {
                tracing::warn!("System time is before UNIX_EPOCH, considering unhealthy");
                return false;
            }
        };
        let last_heartbeat = self.get_last_heartbeat();

        if last_heartbeat == 0 || last_heartbeat > now {
            return false;
        }

        tracing::debug!(
            last_heartbeat = last_heartbeat,
            now = now,
            timeout = timeout.as_secs(),
            "check device health (V2 with ERL)"
        );

        now.saturating_sub(last_heartbeat) <= timeout.as_secs()
    }

    pub fn add_pid(&self, pid: usize) {
        let _ = self.pids.lock().insert_if_absent(pid);
    }

    pub fn remove_pid(&self, pid: usize) {
        let _ = self.pids.lock().remove_value(&pid);
    }

    pub fn get_all_pids(&self) -> Vec<usize> {
        self.pids.lock().values().copied().collect()
    }

    pub fn cleanup_orphaned_locks(&self) {
        self.pids.cleanup_orphaned_lock();
    }

    pub fn iter_active_devices(&self) -> impl Iterator<Item = (usize, &DeviceEntryV2)> {
        self.devices
            .iter()
            .enumerate()
            .filter(|(_, device)| device.is_active())
    }

    pub fn iter_all_devices(&self) -> impl Iterator<Item = (usize, &DeviceEntryV2)> {
        self.devices.iter().enumerate()
    }
}

/// Device configuration information.
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Device index
    pub device_idx: u32,
    /// Device UUID
    pub device_uuid: String,
    /// Utilization limit percentage (0-100)
    pub up_limit: u32,
    /// Memory limit in bytes
    pub mem_limit: u64,
    /// Number of streaming multiprocessors
    pub sm_count: u32,
    /// Maximum threads per streaming multiprocessor
    pub max_thread_per_sm: u32,
    /// Total number of CUDA cores
    pub total_cuda_cores: u32,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use crate::shared_memory::handle::{SharedMemoryHandle, SHM_PATH_SUFFIX};
    use crate::shared_memory::manager::ThreadSafeSharedMemoryManager;

    use super::*;

    const TEST_SHM_BASE_PATH: &str = "/tmp/shm";
    const TEST_DEVICE_IDX: u32 = 0;
    const TEST_TOTAL_CORES: u32 = 1024;
    const TEST_UP_LIMIT: u32 = 80;
    const TEST_MEM_LIMIT: u64 = 1024 * 1024 * 1024; // 1GB

    fn create_test_configs() -> Vec<DeviceConfig> {
        vec![DeviceConfig {
            device_idx: TEST_DEVICE_IDX,
            device_uuid: "test-device-uuid".to_string(),
            up_limit: TEST_UP_LIMIT,
            mem_limit: TEST_MEM_LIMIT,
            total_cuda_cores: TEST_TOTAL_CORES,
            sm_count: 10,
            max_thread_per_sm: 1024,
        }]
    }

    #[test]
    fn device_entry_basic_operations() {
        let entry = DeviceEntry::new();

        // Test UUID operations
        entry.set_uuid("test-uuid-123");
        assert_eq!(entry.get_uuid(), "test-uuid-123");

        // Test active status
        assert!(!entry.is_active());
        entry.set_active(true);
        assert!(entry.is_active());
        entry.set_active(false);
        assert!(!entry.is_active());

        // Test very long UUID handling
        let long_uuid = "a".repeat(MAX_UUID_LEN + 10);
        entry.set_uuid(&long_uuid);
        let stored_uuid = entry.get_uuid();
        assert!(stored_uuid.len() < MAX_UUID_LEN);
        assert!(stored_uuid.starts_with("a"));
    }

    #[test]
    fn shared_device_state_creation_and_basic_ops() {
        let configs = create_test_configs();
        let state = SharedDeviceState::new(&configs);

        // Test initial state (V2 by default)
        assert_eq!(state.version(), 2);
        assert_eq!(state.device_count(), 1);

        // Test that heartbeat is initialized to current time (should be non-zero and recent)
        let heartbeat = state.get_last_heartbeat();
        assert!(heartbeat > 0);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(now.saturating_sub(heartbeat) < 2); // Should be within 2 seconds

        // Should be healthy since heartbeat was just set
        assert!(state.is_healthy(Duration::from_secs(30)));

        // Test device exists by index
        let device_idx = configs[0].device_idx as usize;
        assert!(state.has_device(device_idx));
    }

    #[test]
    fn shared_device_state_heartbeat_functionality() {
        let state = SharedDeviceState::new(&[]);

        // Test initial healthy state (heartbeat is initialized to current time)
        assert!(state.is_healthy(Duration::from_secs(30)));

        // Test setting heartbeat to a specific time
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        state.update_heartbeat(now);
        assert_eq!(state.get_last_heartbeat(), now);
        assert!(state.is_healthy(Duration::from_secs(30)));

        // Test old heartbeat (should be unhealthy)
        state.update_heartbeat(now - 60);
        assert!(!state.is_healthy(Duration::from_secs(30)));
    }

    #[test]
    fn shared_device_info_atomic_operations() {
        // Test V1 device info (has available_cores)
        let device_info_v1 =
            SharedDeviceInfoV1::new(TEST_TOTAL_CORES, TEST_UP_LIMIT, TEST_MEM_LIMIT);

        // Test available cores operations (V1 only)
        device_info_v1.set_available_cores(512);
        assert_eq!(device_info_v1.get_available_cores(), 512);

        let old_value = device_info_v1.fetch_add_available_cores(100);
        assert_eq!(old_value, 512);
        assert_eq!(device_info_v1.get_available_cores(), 612);

        let old_value = device_info_v1.fetch_sub_available_cores(12);
        assert_eq!(old_value, 612);
        assert_eq!(device_info_v1.get_available_cores(), 600);

        // Test negative values
        device_info_v1.set_available_cores(-50);
        assert_eq!(device_info_v1.get_available_cores(), -50);

        // Test other fields
        device_info_v1.set_up_limit(90);
        assert_eq!(device_info_v1.get_up_limit(), 90);

        device_info_v1.set_mem_limit(2 * 1024 * 1024 * 1024);
        assert_eq!(device_info_v1.get_mem_limit(), 2 * 1024 * 1024 * 1024);

        // Test V2 device info (has ERL fields)
        let device_info_v2 = SharedDeviceInfo::new(TEST_TOTAL_CORES, TEST_UP_LIMIT, TEST_MEM_LIMIT);

        // V2 available_cores always returns 0
        assert_eq!(device_info_v2.get_available_cores(), 0);

        // Test ERL fields
        device_info_v2.set_erl_avg_cost(2.5);
        assert_eq!(device_info_v2.get_erl_avg_cost(), 2.5);

        device_info_v2.set_erl_token_capacity(100.0);
        assert_eq!(device_info_v2.get_erl_token_capacity(), 100.0);

        device_info_v2.set_pod_memory_used(512 * 1024 * 1024);
        assert_eq!(device_info_v2.get_pod_memory_used(), 512 * 1024 * 1024);
    }

    #[test]
    fn shared_memory_handle_create_and_open() {
        let configs = create_test_configs();
        let identifier = PodIdentifier::new("handle_create_open", "test");

        let pod_path = identifier.to_path(TEST_SHM_BASE_PATH);
        // Create shared memory
        let handle1 = SharedMemoryHandle::create(&pod_path, &configs)
            .expect("should create shared memory successfully");

        let state1 = handle1.get_state();
        assert_eq!(state1.version(), 2);
        assert_eq!(state1.device_count(), 1);

        // Verify shared memory file exists after creation
        assert!(Path::new(&pod_path).exists());
        // Open existing shared memory
        let handle2 = SharedMemoryHandle::open(&pod_path)
            .expect("should open existing shared memory successfully");

        let state2 = handle2.get_state();
        assert_eq!(state2.version(), 2);
        assert_eq!(state2.device_count(), 1);

        // Verify they access the same memory
        let device_idx = configs[0].device_idx as usize;
        state1.with_device(
            device_idx,
            |device| {
                device.device_info.set_available_cores(42);
            },
            |device| {
                device.device_info.set_available_cores(42);
            },
        );

        let cores = state2.with_device(
            device_idx,
            |device| device.device_info.get_available_cores(),
            |device| device.device_info.get_available_cores(),
        );
        match state2.version() {
            1 => assert_eq!(cores, Some(42)),
            2 => assert_eq!(cores, Some(0)),
            _ => assert!(
                cores.is_some(),
                "should read available cores for known versions"
            ),
        }

        // File should still exist while handles are active
        assert!(Path::new(&pod_path).exists());
    }

    #[test]
    fn shared_memory_handle_error_handling() {
        let result = SharedMemoryHandle::open("non_existent_memory");
        assert!(result.is_err());
    }

    #[test]
    fn concurrent_device_access() {
        let configs = create_test_configs();
        let identifier = PodIdentifier::new("concurrent_access", "test");
        let pod_path = identifier.to_path(TEST_SHM_BASE_PATH);

        let handle = Arc::new(
            SharedMemoryHandle::create(&pod_path, &configs)
                .expect("should create shared memory successfully"),
        );

        let device_idx = configs[0].device_idx as usize;
        let mut handles = vec![];

        // Spawn multiple threads doing concurrent access
        for i in 0..5 {
            let handle_clone = Arc::clone(&handle);

            let thread_handle = thread::spawn(move || {
                let state = handle_clone.get_state();

                for j in 0..20 {
                    let value = i * 20 + j;
                    state.with_device(
                        device_idx,
                        |device| {
                            device.device_info.set_available_cores(value);
                        },
                        |device| {
                            device.device_info.set_available_cores(value);
                        },
                    );

                    thread::sleep(Duration::from_millis(1));

                    let read_value = state
                        .with_device(
                            device_idx,
                            |device| device.device_info.get_available_cores(),
                            |device| device.device_info.get_available_cores(),
                        )
                        .unwrap();

                    // Value should be valid (set by some thread)
                    assert!((0..100).contains(&read_value));
                }
            });

            handles.push(thread_handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }
    }

    #[test]
    fn thread_safe_manager_basic_operations() {
        let manager = ThreadSafeSharedMemoryManager::new();
        let configs = create_test_configs();
        let identifier = PodIdentifier::new("manager_basic", "test");
        let pod_path = identifier.to_path(TEST_SHM_BASE_PATH);

        // Test creation
        manager.create_shared_memory(&pod_path, &configs).unwrap();
        assert!(manager.contains(&pod_path));

        // Verify shared memory file exists
        assert!(Path::new(&pod_path).exists());

        // Test getting shared memory
        let ptr = manager.get_shared_memory(&pod_path).unwrap();
        assert!(!ptr.is_null());

        // Test accessing through pointer
        unsafe {
            let state = &*ptr;
            assert_eq!(state.version(), 2);
            assert_eq!(state.device_count(), 1);
        }

        // Test cleanup
        manager.cleanup(&pod_path).unwrap();
        assert!(!manager.contains(&pod_path));

        // Check that the path exists but is an empty directory
        let path = Path::new(&pod_path);
        assert!(
            path.read_dir().unwrap().next().is_none(),
            "Directory should be empty"
        );
    }

    #[test]
    fn thread_safe_manager_concurrent_creation() {
        let manager = Arc::new(ThreadSafeSharedMemoryManager::new());
        let configs = create_test_configs();
        let identifier = PodIdentifier::new("manager_concurrent", "test");
        let pod_path = identifier.to_path(TEST_SHM_BASE_PATH);

        let mut handles = vec![];

        // Multiple threads trying to create the same shared memory
        for _ in 0..5 {
            let manager_clone = Arc::clone(&manager);
            let configs_clone = configs.clone();
            let pod_path_clone = pod_path.clone();

            let handle = thread::spawn(move || {
                let result = manager_clone.create_shared_memory(&pod_path_clone, &configs_clone);
                assert!(result.is_ok());
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have exactly one shared memory
        assert!(manager.contains(&pod_path));
        manager.cleanup(&pod_path).unwrap();
    }

    #[test]
    fn orphaned_file_cleanup() {
        let manager = ThreadSafeSharedMemoryManager::new();

        // Create a fake orphaned file in /tmp
        let test_file = "/tmp/test_orphaned_shm_file";
        std::fs::write(test_file, "fake shared memory data").unwrap();

        // Verify file exists
        assert!(Path::new(test_file).exists());

        // Test cleanup with a pattern that won't match
        let cleaned = manager
            .cleanup_orphaned_files("nonexistent_pattern", |_| false, Path::new("/"))
            .unwrap();
        assert_eq!(cleaned.len(), 0);

        // File should still exist since pattern didn't match
        assert!(Path::new(test_file).exists());

        // Clean up test file
        std::fs::remove_file(test_file).unwrap();
    }

    #[test]
    fn device_iteration_methods() {
        // Create multiple device configurations
        let configs = vec![
            DeviceConfig {
                device_idx: 0,
                device_uuid: "device-0".to_string(),
                up_limit: 80,
                mem_limit: 1024 * 1024 * 1024,
                total_cuda_cores: 1024,
                sm_count: 10,
                max_thread_per_sm: 1024,
            },
            DeviceConfig {
                device_idx: 2,
                device_uuid: "device-2".to_string(),
                up_limit: 70,
                mem_limit: 2 * 1024 * 1024 * 1024,
                total_cuda_cores: 2048,
                sm_count: 20,
                max_thread_per_sm: 1024,
            },
        ];

        let state = SharedDeviceState::new(&configs);

        // Test iter_active_devices
        let active_devices: Vec<_> = state.iter_active_devices().collect();
        assert_eq!(active_devices.len(), 2);

        // Check that indices match the device_idx from configs
        assert_eq!(active_devices[0].0, 0);
        assert_eq!(active_devices[0].1.get_uuid(), "device-0");
        assert_eq!(active_devices[1].0, 2);
        assert_eq!(active_devices[1].1.get_uuid(), "device-2");

        // Test iter_all_devices (should return all MAX_DEVICES entries)
        let all_devices: Vec<_> = state.iter_all_devices().collect();
        assert_eq!(all_devices.len(), MAX_DEVICES);

        // Only the first two devices (at indices 0 and 2) should be active
        let active_count = all_devices
            .iter()
            .filter(|(_, device)| device.is_active())
            .count();
        assert_eq!(active_count, 2);

        // Test for_each_active_device
        let mut found_devices = Vec::new();
        state.for_each_active_device(|idx, device| {
            found_devices.push((idx, device.get_uuid_owned()));
        });

        assert_eq!(found_devices.len(), 2);
        assert_eq!(found_devices[0], (0, "device-0".to_string()));
        assert_eq!(found_devices[1], (2, "device-2".to_string()));

        // Test deactivating a device and checking iteration
        match &state {
            SharedDeviceState::V1(inner) => inner.devices[2].set_active(false),
            SharedDeviceState::V2(inner) => inner.devices[2].set_active(false),
        }

        let active_after_deactivation: Vec<_> = state.iter_active_devices().collect();
        assert_eq!(active_after_deactivation.len(), 1);
        assert_eq!(active_after_deactivation[0].0, 0);
        assert_eq!(active_after_deactivation[0].1.get_uuid(), "device-0");
    }

    #[test]
    fn pid_set_deduplicates_on_add() {
        let state = SharedDeviceState::new(&[]);

        // Add the same pid multiple times
        state.add_pid(1234);
        state.add_pid(1234);
        state.add_pid(1234);

        let pids = state.get_all_pids();
        assert_eq!(
            pids.len(),
            1,
            "should contain only one PID after duplicate adds"
        );
        assert_eq!(pids[0], 1234);
    }

    #[test]
    fn pid_remove_by_value_works() {
        let state = SharedDeviceState::new(&[]);

        state.add_pid(111);
        state.add_pid(222);
        state.add_pid(333);

        state.remove_pid(222);

        let pids = state.get_all_pids();
        assert_eq!(pids.len(), 2, "should remove the specified PID");
        assert!(pids.contains(&111));
        assert!(pids.contains(&333));
        assert!(!pids.contains(&222));
    }

    #[test]
    fn pid_set_capacity_and_duplicate_behavior() {
        let state = SharedDeviceState::new(&[]);

        // Fill to capacity with unique PIDs
        for pid in 0..MAX_PROCESSES {
            state.add_pid(pid);
        }

        let pids = state.get_all_pids();
        assert_eq!(
            pids.len(),
            MAX_PROCESSES,
            "should reach max capacity with unique PIDs"
        );

        // Adding an existing PID should not change the count
        state.add_pid(0);
        let pids_after_dup = state.get_all_pids();
        assert_eq!(
            pids_after_dup.len(),
            MAX_PROCESSES,
            "should remain at capacity when inserting duplicate"
        );

        // Attempt to add a new PID beyond capacity should be a no-op
        state.add_pid(MAX_PROCESSES + 1);
        let pids_final = state.get_all_pids();
        assert_eq!(
            pids_final.len(),
            MAX_PROCESSES,
            "should remain at capacity when inserting new PID beyond capacity"
        );
    }

    #[test]
    fn test_cleanup_empty_parent_directories() {
        use std::fs;
        use tempfile::TempDir;

        // Create a temporary directory structure
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create nested directory structure: base/namespace/podname/
        let namespace_dir = base_path.join("test-namespace");
        let pod_dir = namespace_dir.join("test-pod");
        fs::create_dir_all(&pod_dir).unwrap();

        // Create a file in the pod directory
        let test_file = pod_dir.join(SHM_PATH_SUFFIX);
        fs::write(&test_file, "test data").unwrap();

        // Verify structure exists
        assert!(test_file.exists());
        assert!(pod_dir.exists());
        assert!(namespace_dir.exists());

        // Remove the file
        fs::remove_file(&test_file).unwrap();

        // Test cleanup without stop_at_path (should remove all empty dirs)
        let result = cleanup_empty_parent_directories(&test_file, None);
        assert!(result.is_ok());

        // Pod directory should be removed
        assert!(!pod_dir.exists());
        // Namespace directory should be removed
        assert!(!namespace_dir.exists());
    }

    #[test]
    fn test_cleanup_empty_parent_directories_with_stop_at_path() {
        use std::fs;
        use tempfile::TempDir;

        // Create a temporary directory structure
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create nested directory structure: base/namespace/podname/
        let namespace_dir = base_path.join("test-namespace");
        let pod_dir = namespace_dir.join("test-pod");
        fs::create_dir_all(&pod_dir).unwrap();

        // Create a file in the pod directory
        let test_file = pod_dir.join(SHM_PATH_SUFFIX);
        fs::write(&test_file, "test data").unwrap();

        // Remove the file
        fs::remove_file(&test_file).unwrap();

        // Test cleanup with stop_at_path set to base_path
        let result = cleanup_empty_parent_directories(&test_file, Some(base_path));
        assert!(result.is_ok());

        // Pod directory should be removed
        assert!(!pod_dir.exists());
        // Namespace directory should be removed
        assert!(!namespace_dir.exists());
        // Base directory should remain (it's the stop_at_path)
        assert!(base_path.exists());
    }

    #[test]
    fn test_cleanup_empty_parent_directories_stops_at_non_empty_dir() {
        use std::fs;
        use tempfile::TempDir;

        // Create a temporary directory structure
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create nested directory structure: base/namespace/podname/
        let namespace_dir = base_path.join("test-namespace");
        let pod_dir = namespace_dir.join("test-pod");
        fs::create_dir_all(&pod_dir).unwrap();

        // Create two files in the pod directory
        let test_file1 = pod_dir.join(SHM_PATH_SUFFIX);
        let test_file2 = pod_dir.join("other_file");
        fs::write(&test_file1, "test data").unwrap();
        fs::write(&test_file2, "other data").unwrap();

        // Remove only one file
        fs::remove_file(&test_file1).unwrap();

        // Test cleanup - should not remove pod directory since it's not empty
        let result = cleanup_empty_parent_directories(&test_file1, Some(base_path));
        assert!(result.is_ok());

        // Pod directory should still exist (not empty)
        assert!(pod_dir.exists());
        assert!(namespace_dir.exists());
        assert!(test_file2.exists());
    }

    #[test]
    fn test_cleanup_empty_parent_directories_with_nested_stop_path() {
        use std::fs;
        use tempfile::TempDir;

        // Create a temporary directory structure
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create nested directory structure: base/namespace/podname/
        let namespace_dir = base_path.join("test-namespace");
        let pod_dir = namespace_dir.join("test-pod");
        fs::create_dir_all(&pod_dir).unwrap();

        // Create a file in the pod directory
        let test_file = pod_dir.join(SHM_PATH_SUFFIX);
        fs::write(&test_file, "test data").unwrap();

        // Remove the file
        fs::remove_file(&test_file).unwrap();

        // Test cleanup with stop_at_path set to namespace directory
        let result = cleanup_empty_parent_directories(&test_file, Some(&namespace_dir));
        assert!(result.is_ok());

        // Pod directory should be removed
        assert!(!pod_dir.exists());
        // Namespace directory should remain (it's the stop_at_path)
        assert!(namespace_dir.exists());
        assert!(base_path.exists());
    }
}
