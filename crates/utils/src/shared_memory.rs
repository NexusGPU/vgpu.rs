//! Shared Memory Module
//! This module provides utilities for managing shared memory segments used for
//! GPU resource coordination between processes.

use std::collections::HashMap;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use anyhow::Context;
use anyhow::Result;
use shared_memory::Mode;
use shared_memory::Shmem;
use shared_memory::ShmemConf;
use shared_memory::ShmemError;
use spin::RwLock;
use tracing::info;
use tracing::warn;

/// Maximum number of devices that can be stored in shared memory
const MAX_DEVICES: usize = 16;
/// Maximum length of device UUID string (including null terminator)
const MAX_UUID_LEN: usize = 64;

/// Device state stored in shared memory for a single device.
#[repr(C)]
#[derive(Debug)]
pub struct SharedDeviceInfo {
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

impl SharedDeviceInfo {
    /// Creates a new SharedDeviceInfo instance.
    pub fn new(total_cuda_cores: u32, up_limit: u32, mem_limit: u64) -> Self {
        Self {
            available_cuda_cores: AtomicI32::new(0),
            up_limit: AtomicU32::new(up_limit),
            mem_limit: AtomicU64::new(mem_limit),
            total_cuda_cores: AtomicU32::new(total_cuda_cores),
            pod_memory_used: AtomicU64::new(0),
        }
    }

    /// Gets the number of available CUDA cores.
    pub fn get_available_cores(&self) -> i32 {
        self.available_cuda_cores.load(Ordering::Acquire)
    }

    /// Sets the number of available CUDA cores.
    pub fn set_available_cores(&self, cores: i32) {
        self.available_cuda_cores.store(cores, Ordering::Release)
    }

    /// Fetches and adds the number of available CUDA cores.
    pub fn fetch_add_available_cores(&self, cores: i32) -> i32 {
        self.available_cuda_cores.fetch_add(cores, Ordering::AcqRel)
    }

    /// Subtracts the number of available CUDA cores.
    pub fn fetch_sub_available_cores(&self, cores: i32) -> i32 {
        self.available_cuda_cores.fetch_sub(cores, Ordering::AcqRel)
    }

    /// Gets the utilization limit.
    pub fn get_up_limit(&self) -> u32 {
        self.up_limit.load(Ordering::Acquire)
    }

    /// Sets the utilization limit.
    pub fn set_up_limit(&self, limit: u32) {
        self.up_limit.store(limit, Ordering::Release)
    }

    /// Gets the memory limit.
    pub fn get_mem_limit(&self) -> u64 {
        self.mem_limit.load(Ordering::Acquire)
    }

    /// Sets the memory limit.
    pub fn set_mem_limit(&self, limit: u64) {
        self.mem_limit.store(limit, Ordering::Release)
    }

    /// Gets the total number of CUDA cores.
    pub fn get_total_cores(&self) -> u32 {
        self.total_cuda_cores.load(Ordering::Acquire)
    }

    /// Gets the pod memory usage.
    pub fn get_pod_memory_used(&self) -> u64 {
        self.pod_memory_used.load(Ordering::Acquire)
    }

    /// Updates the pod memory usage.
    pub fn set_pod_memory_used(&self, memory: u64) {
        self.pod_memory_used.store(memory, Ordering::Release);
    }
}

/// A device entry in the shared memory array
#[repr(C)]
#[derive(Debug)]
pub struct DeviceEntry {
    /// Device UUID as fixed-size null-terminated string
    pub uuid: [u8; MAX_UUID_LEN],
    /// Device information
    pub device_info: SharedDeviceInfo,
    /// Whether this entry is valid/active
    pub is_active: AtomicU32, // Using AtomicU32 as atomic bool
}

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

    /// Compares UUID efficiently without string allocation
    pub fn uuid_matches(&self, uuid: &str) -> bool {
        if !self.is_active() {
            return false;
        }
        let uuid_bytes = uuid.as_bytes();
        if uuid_bytes.len() >= MAX_UUID_LEN {
            return false;
        }

        // Check if lengths match by finding null terminator
        let stored_len = self
            .uuid
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(MAX_UUID_LEN);
        if stored_len != uuid_bytes.len() {
            return false;
        }

        // Compare bytes directly
        self.uuid[..stored_len] == *uuid_bytes
    }
}

impl Default for DeviceEntry {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared device state using only simple data types safe for inter-process sharing
#[repr(C)]
#[derive(Debug)]
pub struct SharedDeviceState {
    /// Fixed-size array of device entries
    pub devices: [DeviceEntry; MAX_DEVICES],
    /// Number of active devices
    pub device_count: AtomicU32,
    /// Last heartbeat timestamp from hypervisor (for health monitoring).
    pub last_heartbeat: AtomicU64,
    /// Reference count for tracking how many processes are using this shared memory
    pub reference_count: AtomicU32,
}

impl SharedDeviceState {
    /// Creates a new SharedDeviceState instance.
    pub fn new(configs: &[DeviceConfig]) -> Self {
        let state = Self {
            devices: std::array::from_fn(|_| DeviceEntry::new()),
            device_count: AtomicU32::new(0),
            last_heartbeat: AtomicU64::new(0),
            reference_count: AtomicU32::new(1),
        };

        // Add devices from configs
        let device_count = std::cmp::min(configs.len(), MAX_DEVICES);
        for (i, config) in configs.iter().take(device_count).enumerate() {
            state.devices[i].set_uuid(&config.device_uuid);
            // Use atomic operations to set device info
            let device_info = &state.devices[i].device_info;
            device_info
                .total_cuda_cores
                .store(config.total_cuda_cores, Ordering::Relaxed);
            device_info
                .up_limit
                .store(config.up_limit, Ordering::Relaxed);
            device_info
                .mem_limit
                .store(config.mem_limit, Ordering::Relaxed);
            state.devices[i].set_active(true);
        }

        if configs.len() > MAX_DEVICES {
            warn!(
                "Too many devices in config: {}, maximum {} supported",
                configs.len(),
                MAX_DEVICES
            );
        }

        state
            .device_count
            .store(device_count as u32, Ordering::Release);
        state
    }

    /// Finds device index by UUID efficiently
    fn find_device_index(&self, device_uuid: &str) -> Option<usize> {
        let current_count = self.device_count.load(Ordering::Acquire) as usize;

        (0..current_count).find(|&i| self.devices[i].uuid_matches(device_uuid))
    }

    /// Adds or updates a device in the state.
    pub fn add_device(&self, device_uuid: String, device_info: SharedDeviceInfo) -> bool {
        // First check if device already exists
        if let Some(i) = self.find_device_index(&device_uuid) {
            // Update existing device atomically
            let existing = &self.devices[i].device_info;
            existing
                .total_cuda_cores
                .store(device_info.get_total_cores(), Ordering::Relaxed);
            existing
                .up_limit
                .store(device_info.get_up_limit(), Ordering::Relaxed);
            existing
                .mem_limit
                .store(device_info.get_mem_limit(), Ordering::Relaxed);
            existing
                .available_cuda_cores
                .store(device_info.get_available_cores(), Ordering::Relaxed);
            existing
                .pod_memory_used
                .store(device_info.get_pod_memory_used(), Ordering::Relaxed);
            return true;
        }

        // Add new device if there's space
        let current_count = self.device_count.load(Ordering::Acquire) as usize;
        if current_count < MAX_DEVICES {
            // Try to atomically increment the count
            let new_count = current_count + 1;
            if self
                .device_count
                .compare_exchange_weak(
                    current_count as u32,
                    new_count as u32,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                // Successfully reserved the slot
                let entry = &self.devices[current_count];
                entry.set_uuid(&device_uuid);

                // Set device info atomically
                let info = &entry.device_info;
                info.total_cuda_cores
                    .store(device_info.get_total_cores(), Ordering::Relaxed);
                info.up_limit
                    .store(device_info.get_up_limit(), Ordering::Relaxed);
                info.mem_limit
                    .store(device_info.get_mem_limit(), Ordering::Relaxed);
                info.available_cuda_cores
                    .store(device_info.get_available_cores(), Ordering::Relaxed);
                info.pod_memory_used
                    .store(device_info.get_pod_memory_used(), Ordering::Relaxed);

                entry.set_active(true);
                return true;
            }
        }
        false
    }

    /// Removes a device from the state.
    pub fn remove_device(&self, device_uuid: &str) -> Option<SharedDeviceInfo> {
        if let Some(i) = self.find_device_index(device_uuid) {
            let entry = &self.devices[i];

            // Create a copy of the device info before removing
            let device_info = SharedDeviceInfo::new(
                entry.device_info.get_total_cores(),
                entry.device_info.get_up_limit(),
                entry.device_info.get_mem_limit(),
            );

            // Mark as inactive first
            entry.set_active(false);

            // Compact the array by moving last active device to this position
            let current_count = self.device_count.load(Ordering::Acquire) as usize;
            if i != current_count - 1 {
                // Move last device to removed position
                let last_entry = &self.devices[current_count - 1];
                if last_entry.is_active() {
                    // Copy last entry to current position
                    entry.set_uuid(last_entry.get_uuid());
                    let src_info = &last_entry.device_info;
                    let dst_info = &entry.device_info;
                    dst_info
                        .total_cuda_cores
                        .store(src_info.get_total_cores(), Ordering::Relaxed);
                    dst_info
                        .up_limit
                        .store(src_info.get_up_limit(), Ordering::Relaxed);
                    dst_info
                        .mem_limit
                        .store(src_info.get_mem_limit(), Ordering::Relaxed);
                    dst_info
                        .available_cuda_cores
                        .store(src_info.get_available_cores(), Ordering::Relaxed);
                    dst_info
                        .pod_memory_used
                        .store(src_info.get_pod_memory_used(), Ordering::Relaxed);
                    entry.set_active(true);

                    // Deactivate the last entry
                    last_entry.set_active(false);
                }
            }

            // Decrement count
            self.device_count
                .store((current_count - 1) as u32, Ordering::Release);
            return Some(device_info);
        }
        None
    }

    /// Checks if a device exists by UUID.
    pub fn has_device(&self, device_uuid: &str) -> bool {
        self.find_device_index(device_uuid).is_some()
    }

    /// Gets a list of all device UUIDs efficiently.
    pub fn get_device_uuids(&self) -> Vec<String> {
        let current_count = self.device_count.load(Ordering::Acquire) as usize;
        let mut uuids = Vec::with_capacity(current_count);

        for i in 0..current_count {
            if self.devices[i].is_active() {
                uuids.push(self.devices[i].get_uuid_owned());
            }
        }
        uuids
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
    pub fn is_healthy(&self, timeout_seconds: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let last_heartbeat = self.get_last_heartbeat();

        if last_heartbeat == 0 {
            return false; // No heartbeat recorded
        }

        now.saturating_sub(last_heartbeat) <= timeout_seconds
    }

    /// Executes a closure with a device by UUID efficiently.
    pub fn with_device_by_uuid<T, F>(&self, device_uuid: &str, f: F) -> Option<T>
    where F: FnOnce(&SharedDeviceInfo) -> T {
        self.find_device_index(device_uuid)
            .map(|i| f(&self.devices[i].device_info))
    }

    /// Executes a closure with a mutable device by UUID efficiently.
    pub fn with_device_by_uuid_mut<T, F>(&self, device_uuid: &str, f: F) -> Option<T>
    where F: FnOnce(&SharedDeviceInfo) -> T {
        // Note: Since SharedDeviceInfo uses atomic operations internally,
        // we don't actually need mutable access to modify its values
        self.with_device_by_uuid(device_uuid, f)
    }

    /// Increments the reference count and returns the new value.
    pub fn increment_ref_count(&self) -> u32 {
        self.reference_count.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Decrements the reference count and returns the new value.
    /// Returns 0 if the reference count would underflow.
    pub fn decrement_ref_count(&self) -> u32 {
        let current = self.reference_count.load(Ordering::Acquire);
        if current > 0 {
            self.reference_count.fetch_sub(1, Ordering::AcqRel) - 1
        } else {
            0
        }
    }

    /// Gets the current reference count.
    pub fn get_ref_count(&self) -> u32 {
        self.reference_count.load(Ordering::Acquire)
    }

    /// Safely decrements reference count using compare-and-swap to avoid underflow.
    pub fn try_decrement_ref_count(&self) -> Result<u32, ()> {
        loop {
            let current = self.reference_count.load(Ordering::Acquire);
            if current == 0 {
                return Err(());
            }
            let new_value = current - 1;
            if self
                .reference_count
                .compare_exchange_weak(current, new_value, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(new_value);
            }
        }
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

/// A thread-safe shared memory manager.
pub struct ThreadSafeSharedMemoryManager {
    /// Active shared memory segments: identifier -> Shmem
    active_memories: RwLock<HashMap<String, SharedMemoryHandle>>,
}

impl ThreadSafeSharedMemoryManager {
    /// Creates a new thread-safe shared memory manager.
    pub fn new() -> Self {
        Self {
            active_memories: RwLock::new(HashMap::new()),
        }
    }

    /// Creates or gets a shared memory segment.
    pub fn create_or_get_shared_memory(
        &self,
        identifier: &str,
        configs: &[DeviceConfig],
    ) -> Result<()> {
        let mut memories = self.active_memories.write();

        // Check if the segment already exists.
        if memories.contains_key(identifier) {
            return Ok(());
        }
        // Create a new shared memory segment.
        let shmem = SharedMemoryHandle::create(identifier, configs)?;

        // Store the Shmem object and configuration.
        memories.insert(identifier.to_string(), shmem);

        Ok(())
    }

    /// Gets a pointer to the shared memory by its identifier.
    pub fn get_shared_memory(&self, identifier: &str) -> Result<*mut SharedDeviceState> {
        let memories = self.active_memories.read();

        if let Some(shmem) = memories.get(identifier) {
            let ptr = shmem.get_ptr();
            Ok(ptr)
        } else {
            Err(anyhow::anyhow!("Shared memory not found: {}", identifier))
        }
    }

    /// Cleans up a shared memory segment.
    pub fn cleanup(&self, identifier: &str) -> Result<()> {
        let mut memories = self.active_memories.write();

        if let Some(shmem) = memories.remove(identifier) {
            // Drop the Shmem object to release the shared memory.
            drop(shmem);
            info!(identifier = %identifier, "Cleaned up shared memory segment");
        } else {
            warn!(identifier = %identifier, "Attempted to cleanup non-existent shared memory");
        }

        Ok(())
    }

    /// Force cleanup of a shared memory segment, ignoring reference count.
    /// This should only be used in emergency situations or during shutdown.
    pub fn force_cleanup(&self, identifier: &str) -> Result<()> {
        self.cleanup(identifier)
    }

    /// Checks if a shared memory segment should be cleaned up based on reference count.
    /// Returns true if the segment exists and has zero references.
    pub fn should_cleanup(&self, identifier: &str) -> bool {
        let memories = self.active_memories.read();
        if let Some(shmem) = memories.get(identifier) {
            let state = shmem.get_state();
            state.get_ref_count() == 0
        } else {
            false
        }
    }

    /// Attempt to cleanup shared memory segments with zero reference count.
    pub fn cleanup_unused(&self) -> Result<Vec<String>> {
        let mut cleaned_up = Vec::new();
        let identifiers: Vec<String> = {
            let memories = self.active_memories.read();
            memories.keys().cloned().collect()
        };

        for identifier in identifiers {
            if self.should_cleanup(&identifier) {
                match self.cleanup(&identifier) {
                    Ok(_) => {
                        cleaned_up.push(identifier.clone());
                        info!(identifier = %identifier, "Cleaned up unused shared memory segment");
                    }
                    Err(e) => {
                        warn!(identifier = %identifier, error = %e, "Failed to cleanup unused shared memory segment");
                    }
                }
            }
        }

        Ok(cleaned_up)
    }

    /// Checks if a shared memory segment exists.
    pub fn contains(&self, identifier: &str) -> bool {
        let memories = self.active_memories.read();
        memories.contains_key(identifier)
    }
}

impl Default for ThreadSafeSharedMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

// Ensure that ThreadSafeSharedMemoryManager is thread-safe.
unsafe impl Send for ThreadSafeSharedMemoryManager {}
unsafe impl Sync for ThreadSafeSharedMemoryManager {}

/// Safely access shared memory, automatically handling the segment's lifecycle.
pub struct SharedMemoryHandle {
    _shmem: Shmem,
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
            _shmem: shmem,
            ptr,
            identifier: identifier.to_string(),
        })
    }

    /// Gets a pointer to the shared device state.
    pub fn get_ptr(&self) -> *mut SharedDeviceState {
        self.ptr
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

// Implement Drop to handle reference counting cleanup
impl Drop for SharedMemoryHandle {
    fn drop(&mut self) {
        unsafe {
            let ref_count = (*self.ptr).try_decrement_ref_count().unwrap_or(0);
            if ref_count == 0 {
                // Last reference dropped, try to clean up the shared memory
                // Note: We can't guarantee cleanup here due to potential race conditions
                // with other processes. The OS will clean up when the last process exits.
                info!(
                    identifier = %self.identifier,
                    "Last reference to shared memory dropped, reference count: {}",
                    ref_count
                );
            } else {
                info!(
                    identifier = %self.identifier,
                    "Dropped shared memory reference, remaining count: {}",
                    ref_count
                );
            }
        }
    }
}

// Implement Send and Sync because SharedDeviceState uses atomic operations.
unsafe impl Send for SharedMemoryHandle {}
unsafe impl Sync for SharedMemoryHandle {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use super::*;

    const TEST_IDENTIFIER: &str = "test_shared_memory";
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

    fn create_unique_identifier(test_name: &str) -> String {
        format!("{}_{}_{}", TEST_IDENTIFIER, test_name, std::process::id())
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

        // Test initial state
        assert_eq!(state.device_count(), 1);
        assert_eq!(state.get_last_heartbeat(), 0);
        assert!(!state.is_healthy(30));

        // Test device exists
        let device_uuid = &configs[0].device_uuid;
        assert!(state.has_device(device_uuid));

        // Test device UUIDs retrieval
        let device_uuids = state.get_device_uuids();
        assert_eq!(device_uuids.len(), 1);
        assert_eq!(device_uuids[0], configs[0].device_uuid);
    }

    #[test]
    fn shared_device_state_heartbeat_functionality() {
        let state = SharedDeviceState::new(&[]);

        // Test initial unhealthy state
        assert!(!state.is_healthy(30));

        // Test setting heartbeat
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        state.update_heartbeat(now);
        assert_eq!(state.get_last_heartbeat(), now);
        assert!(state.is_healthy(30));

        // Test old heartbeat
        state.update_heartbeat(now - 60);
        assert!(!state.is_healthy(30));
    }

    #[test]
    fn shared_device_state_device_operations() {
        let state = SharedDeviceState::new(&[]);

        // Add a device
        let device_info = SharedDeviceInfo::new(TEST_TOTAL_CORES, TEST_UP_LIMIT, TEST_MEM_LIMIT);
        state.add_device("test-device".to_string(), device_info);

        assert_eq!(state.device_count(), 1);
        assert!(state.has_device("test-device"));

        // Test device access
        let cores = state.with_device_by_uuid("test-device", |device| device.get_total_cores());
        assert_eq!(cores, Some(TEST_TOTAL_CORES));

        // Test device modification
        state.with_device_by_uuid_mut("test-device", |device| {
            device.set_available_cores(512);
        });

        let available_cores =
            state.with_device_by_uuid("test-device", |device| device.get_available_cores());
        assert_eq!(available_cores, Some(512));

        // Test device removal
        let removed = state.remove_device("test-device");
        assert!(removed.is_some());
        assert_eq!(state.device_count(), 0);
        assert!(!state.has_device("test-device"));
    }

    #[test]
    fn shared_device_state_max_devices() {
        let state = SharedDeviceState::new(&[]);

        // Add MAX_DEVICES devices
        for i in 0..MAX_DEVICES {
            let device_info = SharedDeviceInfo::new(1024, 80, 1024 * 1024 * 1024);
            state.add_device(format!("device-{i}"), device_info);
        }

        assert_eq!(state.device_count(), MAX_DEVICES);

        // Try to add one more device (should be ignored)
        let device_info = SharedDeviceInfo::new(1024, 80, 1024 * 1024 * 1024);
        state.add_device("overflow-device".to_string(), device_info);

        assert_eq!(state.device_count(), MAX_DEVICES);
        assert!(!state.has_device("overflow-device"));
    }

    #[test]
    fn shared_device_info_atomic_operations() {
        let device_info = SharedDeviceInfo::new(TEST_TOTAL_CORES, TEST_UP_LIMIT, TEST_MEM_LIMIT);

        // Test available cores operations
        device_info.set_available_cores(512);
        assert_eq!(device_info.get_available_cores(), 512);

        let old_value = device_info.fetch_add_available_cores(100);
        assert_eq!(old_value, 512);
        assert_eq!(device_info.get_available_cores(), 612);

        let old_value = device_info.fetch_sub_available_cores(12);
        assert_eq!(old_value, 612);
        assert_eq!(device_info.get_available_cores(), 600);

        // Test negative values
        device_info.set_available_cores(-50);
        assert_eq!(device_info.get_available_cores(), -50);

        // Test other fields
        device_info.set_up_limit(90);
        assert_eq!(device_info.get_up_limit(), 90);

        device_info.set_mem_limit(2 * 1024 * 1024 * 1024);
        assert_eq!(device_info.get_mem_limit(), 2 * 1024 * 1024 * 1024);

        device_info.set_pod_memory_used(512 * 1024 * 1024);
        assert_eq!(device_info.get_pod_memory_used(), 512 * 1024 * 1024);
    }

    #[test]
    fn shared_memory_handle_create_and_open() {
        let configs = create_test_configs();
        let identifier = create_unique_identifier("handle_create_open");

        // Create shared memory
        let handle1 = SharedMemoryHandle::create(&identifier, &configs)
            .expect("should create shared memory successfully");

        let state1 = handle1.get_state();
        assert_eq!(state1.device_count(), 1);

        // Open existing shared memory
        let handle2 = SharedMemoryHandle::open(&identifier)
            .expect("should open existing shared memory successfully");

        let state2 = handle2.get_state();
        assert_eq!(state2.device_count(), 1);

        // Verify they access the same memory
        let device_uuid = &configs[0].device_uuid;
        state1.with_device_by_uuid_mut(device_uuid, |device| {
            device.set_available_cores(42);
        });

        let cores = state2.with_device_by_uuid(device_uuid, |device| device.get_available_cores());
        assert_eq!(cores, Some(42));
    }

    #[test]
    fn shared_memory_handle_error_handling() {
        let result = SharedMemoryHandle::open("non_existent_memory");
        assert!(result.is_err());
    }

    #[test]
    fn concurrent_device_access() {
        let configs = create_test_configs();
        let identifier = create_unique_identifier("concurrent_access");

        let handle = Arc::new(
            SharedMemoryHandle::create(&identifier, &configs)
                .expect("should create shared memory successfully"),
        );

        let device_uuid = configs[0].device_uuid.clone();
        let mut handles = vec![];

        // Spawn multiple threads doing concurrent access
        for i in 0..5 {
            let handle_clone = Arc::clone(&handle);
            let device_uuid_clone = device_uuid.clone();

            let thread_handle = thread::spawn(move || {
                let state = handle_clone.get_state();

                for j in 0..20 {
                    let value = i * 20 + j;
                    state.with_device_by_uuid_mut(&device_uuid_clone, |device| {
                        device.set_available_cores(value);
                    });

                    thread::sleep(Duration::from_millis(1));

                    let read_value = state
                        .with_device_by_uuid(&device_uuid_clone, |device| {
                            device.get_available_cores()
                        })
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
        let identifier = create_unique_identifier("manager_basic");

        // Test creation
        manager
            .create_or_get_shared_memory(&identifier, &configs)
            .unwrap();
        assert!(manager.contains(&identifier));

        // Test getting shared memory
        let ptr = manager.get_shared_memory(&identifier).unwrap();
        assert!(!ptr.is_null());

        // Test accessing through pointer
        unsafe {
            let state = &*ptr;
            assert_eq!(state.device_count(), 1);
        }

        // Test cleanup
        manager.cleanup(&identifier).unwrap();
        assert!(!manager.contains(&identifier));
    }

    #[test]
    fn thread_safe_manager_concurrent_creation() {
        let manager = Arc::new(ThreadSafeSharedMemoryManager::new());
        let configs = create_test_configs();
        let identifier = create_unique_identifier("manager_concurrent");

        let mut handles = vec![];

        // Multiple threads trying to create the same shared memory
        for _ in 0..5 {
            let manager_clone = Arc::clone(&manager);
            let configs_clone = configs.clone();
            let identifier_clone = identifier.clone();

            let handle = thread::spawn(move || {
                let result =
                    manager_clone.create_or_get_shared_memory(&identifier_clone, &configs_clone);
                assert!(result.is_ok());
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have exactly one shared memory
        assert!(manager.contains(&identifier));
        manager.cleanup(&identifier).unwrap();
    }

    #[test]
    fn stress_test_device_operations() {
        let state = SharedDeviceState::new(&[]);

        // Add multiple devices
        for i in 0..10 {
            let device_info = SharedDeviceInfo::new(
                1024 + (i as u32) * 100,
                80 + (i as u32),
                (1 + (i as u64)) * 1024 * 1024 * 1024,
            );
            state.add_device(format!("device-{i}"), device_info);
        }

        assert_eq!(state.device_count(), 10);

        // Remove every other device
        for i in (0..10).step_by(2) {
            let removed = state.remove_device(&format!("device-{i}"));
            assert!(removed.is_some());
        }

        assert_eq!(state.device_count(), 5);

        // Verify remaining devices
        for i in (1..10).step_by(2) {
            assert!(state.has_device(&format!("device-{i}")));
        }

        // Verify removed devices are gone
        for i in (0..10).step_by(2) {
            assert!(!state.has_device(&format!("device-{i}")));
        }
    }

    #[test]
    fn reference_counting_basic_operations() {
        let state = SharedDeviceState::new(&[]);

        // Initial reference count should be 1
        assert_eq!(state.get_ref_count(), 1);

        // Increment reference count
        let new_count = state.increment_ref_count();
        assert_eq!(new_count, 2);
        assert_eq!(state.get_ref_count(), 2);

        // Decrement reference count
        let new_count = state.decrement_ref_count();
        assert_eq!(new_count, 1);
        assert_eq!(state.get_ref_count(), 1);

        // Try safe decrement
        let result = state.try_decrement_ref_count();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
        assert_eq!(state.get_ref_count(), 0);

        // Try to decrement when already at 0
        let result = state.try_decrement_ref_count();
        assert!(result.is_err());
        assert_eq!(state.get_ref_count(), 0);
    }

    #[test]
    fn shared_memory_handle_reference_counting() {
        let configs = create_test_configs();
        let identifier = create_unique_identifier("ref_counting");

        // Create first handle
        let handle1 = SharedMemoryHandle::create(&identifier, &configs)
            .expect("should create shared memory successfully");

        let state1 = handle1.get_state();
        assert_eq!(state1.get_ref_count(), 1);

        // Open second handle to same memory
        let handle2 = SharedMemoryHandle::open(&identifier)
            .expect("should open existing shared memory successfully");

        let state2 = handle2.get_state();
        assert_eq!(state2.get_ref_count(), 2);

        // Verify they access the same memory
        assert_eq!(state1.get_ref_count(), 2);

        // Drop first handle
        drop(handle1);
        assert_eq!(state2.get_ref_count(), 1);

        // Drop second handle
        drop(handle2);
        // Note: Can't check state after drop since we no longer have access
    }

    #[test]
    fn thread_safe_manager_cleanup_operations() {
        let manager = ThreadSafeSharedMemoryManager::new();
        let configs = create_test_configs();
        let identifier = create_unique_identifier("cleanup_test");

        // Create shared memory
        manager
            .create_or_get_shared_memory(&identifier, &configs)
            .unwrap();
        assert!(manager.contains(&identifier));

        // Initially should not be ready for cleanup (has 1 reference)
        assert!(!manager.should_cleanup(&identifier));

        // Manually set reference count to 0 for testing
        {
            let ptr = manager.get_shared_memory(&identifier).unwrap();
            unsafe {
                (*ptr).try_decrement_ref_count().ok();
            }
        }

        // Now should be ready for cleanup
        assert!(manager.should_cleanup(&identifier));

        // Cleanup unused segments
        let cleaned = manager.cleanup_unused().unwrap();
        assert_eq!(cleaned.len(), 1);
        assert_eq!(cleaned[0], identifier);
        assert!(!manager.contains(&identifier));
    }

    #[test]
    fn concurrent_reference_counting() {
        let configs = create_test_configs();
        let identifier = create_unique_identifier("concurrent_ref_count");

        let handle = Arc::new(
            SharedMemoryHandle::create(&identifier, &configs)
                .expect("should create shared memory successfully"),
        );

        let mut handles = vec![];

        // Spawn multiple threads incrementing and decrementing reference count
        for i in 0..5 {
            let handle_clone = Arc::clone(&handle);

            let thread_handle = thread::spawn(move || {
                let state = handle_clone.get_state();

                for _ in 0..10 {
                    // Increment
                    state.increment_ref_count();
                    thread::sleep(Duration::from_millis(1));

                    // Try to decrement (may fail if already at 0)
                    let _ = state.try_decrement_ref_count();
                    thread::sleep(Duration::from_millis(1));
                }
            });

            handles.push(thread_handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }

        // Reference count should be >= 1 (the original reference)
        let final_count = handle.get_state().get_ref_count();
        assert!(final_count >= 1);
    }
}
