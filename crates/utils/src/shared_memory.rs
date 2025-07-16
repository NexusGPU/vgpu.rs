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
use shared_memory::Shmem;
use shared_memory::ShmemConf;
use shared_memory::ShmemError;
use spin::RwLock;
use tracing::info;
use tracing::warn;

/// Device state stored in shared memory for a single device.
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

impl SharedDeviceInfoV1 {
    /// Creates a new SharedDeviceStateV1 instance.
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
        self.available_cuda_cores
            .fetch_add(cores, Ordering::Release)
    }

    /// Subtracts the number of available CUDA cores.
    pub fn fetch_sub_available_cores(&self, cores: i32) -> i32 {
        self.available_cuda_cores
            .fetch_sub(cores, Ordering::Release)
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

    /// Updates the pod memory usage and timestamp.
    pub fn update_pod_memory_used(&self, used: u64) {
        self.pod_memory_used.store(used, Ordering::Release);
    }
}

#[derive(Debug)]
pub enum SharedDeviceInfo {
    V1(SharedDeviceInfoV1),
}

impl SharedDeviceInfo {
    /// Creates a new SharedDeviceInfo instance with V1 format.
    pub fn new(total_cuda_cores: u32, up_limit: u32, mem_limit: u64) -> Self {
        Self::V1(SharedDeviceInfoV1::new(
            total_cuda_cores,
            up_limit,
            mem_limit,
        ))
    }

    /// Gets the number of available CUDA cores.
    pub fn get_available_cores(&self) -> i32 {
        match self {
            Self::V1(info) => info.get_available_cores(),
        }
    }

    /// Sets the number of available CUDA cores.
    pub fn set_available_cores(&self, cores: i32) {
        match self {
            Self::V1(info) => info.set_available_cores(cores),
        }
    }

    /// Fetches and adds the number of available CUDA cores.
    pub fn fetch_add_available_cores(&self, cores: i32) -> i32 {
        match self {
            Self::V1(info) => info.fetch_add_available_cores(cores),
        }
    }

    /// Subtracts the number of available CUDA cores.
    pub fn fetch_sub_available_cores(&self, cores: i32) -> i32 {
        match self {
            Self::V1(info) => info.fetch_sub_available_cores(cores),
        }
    }

    /// Gets the utilization limit.
    pub fn get_up_limit(&self) -> u32 {
        match self {
            Self::V1(info) => info.get_up_limit(),
        }
    }

    /// Sets the utilization limit.
    pub fn set_up_limit(&self, limit: u32) {
        match self {
            Self::V1(info) => info.set_up_limit(limit),
        }
    }

    /// Gets the memory limit.
    pub fn get_mem_limit(&self) -> u64 {
        match self {
            Self::V1(info) => info.get_mem_limit(),
        }
    }

    /// Sets the memory limit.
    pub fn set_mem_limit(&self, limit: u64) {
        match self {
            Self::V1(info) => info.set_mem_limit(limit),
        }
    }

    /// Gets the total number of CUDA cores.
    pub fn get_total_cores(&self) -> u32 {
        match self {
            Self::V1(info) => info.get_total_cores(),
        }
    }

    /// Gets the pod memory usage.
    pub fn get_pod_memory_used(&self) -> u64 {
        match self {
            Self::V1(info) => info.get_pod_memory_used(),
        }
    }

    /// Updates the pod memory usage and timestamp.
    pub fn update_pod_memory_used(&self, used: u64) {
        match self {
            Self::V1(info) => info.update_pod_memory_used(used),
        }
    }
}

/// Hypervisor state containing multiple devices and health information.
#[repr(C)]
#[derive(Debug, Default)]
pub struct SharedDeviceState {
    /// Map of device UUID to device state.
    pub devices: RwLock<HashMap<String, SharedDeviceInfo>>,
    /// Last heartbeat timestamp from hypervisor (for health monitoring).
    pub last_heartbeat: AtomicU64,
}

impl SharedDeviceState {
    /// Creates a new SharedDeviceState instance.
    pub fn new(configs: &[DeviceConfig]) -> Self {
        let mut devices = HashMap::new();
        for config in configs {
            devices.insert(
                config.device_uuid.clone(),
                SharedDeviceInfo::new(config.total_cuda_cores, config.up_limit, config.mem_limit),
            );
        }
        Self {
            devices: RwLock::new(devices),
            last_heartbeat: AtomicU64::new(0),
        }
    }

    /// Gets the current CUDA device UUID.
    pub fn get_current_device_uuid() -> Result<String> {
        let mut device: i32 = 0;
        unsafe {
            // Call the CUDA API to get the current device
            let result = cudarc::driver::sys::cuCtxGetDevice(&mut device as *mut i32);
            if result != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(anyhow::anyhow!(
                    "Failed to get current CUDA device: error code {:?}",
                    result
                ));
            }

            // Get device UUID
            let mut uuid: cudarc::driver::sys::CUuuid_st =
                cudarc::driver::sys::CUuuid_st::default();
            let uuid_result = cudarc::driver::sys::cuDeviceGetUuid(
                &mut uuid as *mut cudarc::driver::sys::CUuuid_st,
                device,
            );
            if uuid_result != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(anyhow::anyhow!(
                    "Failed to get CUDA device UUID: error code {:?}",
                    uuid_result
                ));
            }

            // Convert UUID to string
            let uuid_str = format!(
                "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                uuid.bytes[0], uuid.bytes[1], uuid.bytes[2], uuid.bytes[3],
                uuid.bytes[4], uuid.bytes[5], uuid.bytes[6], uuid.bytes[7],
                uuid.bytes[8], uuid.bytes[9], uuid.bytes[10], uuid.bytes[11],
                uuid.bytes[12], uuid.bytes[13], uuid.bytes[14], uuid.bytes[15]
            );

            Ok(uuid_str)
        }
    }

    /// Adds or updates a device in the state.
    pub fn add_device(&self, device_uuid: String, device_info: SharedDeviceInfo) {
        let mut devices = self.devices.write();
        devices.insert(device_uuid, device_info);
    }

    /// Removes a device from the state.
    pub fn remove_device(&self, device_uuid: &str) -> Option<SharedDeviceInfo> {
        let mut devices = self.devices.write();
        devices.remove(device_uuid)
    }

    /// Checks if a device exists by UUID.
    pub fn has_device(&self, device_uuid: &str) -> bool {
        let devices = self.devices.read();
        devices.contains_key(device_uuid)
    }

    /// Gets a list of all device UUIDs.
    pub fn get_device_uuids(&self) -> Vec<String> {
        let devices = self.devices.read();
        devices.keys().cloned().collect()
    }

    /// Gets the number of devices.
    pub fn device_count(&self) -> usize {
        let devices = self.devices.read();
        devices.len()
    }

    /// Updates the heartbeat timestamp.
    pub fn update_heartbeat(&self, timestamp: u64) {
        self.last_heartbeat.store(timestamp, Ordering::Release);
    }

    /// Gets the last heartbeat timestamp.
    pub fn get_last_heartbeat(&self) -> u64 {
        self.last_heartbeat.load(Ordering::Acquire)
    }

    /// Checks if the hypervisor is healthy based on heartbeat timeout.
    /// Returns true if the last heartbeat is within the specified timeout.
    pub fn is_healthy(&self, timeout_seconds: u64) -> bool {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let last_heartbeat = self.get_last_heartbeat();

        if last_heartbeat == 0 {
            // No heartbeat recorded yet
            return false;
        }

        current_time - last_heartbeat <= timeout_seconds
    }

    /// Executes a function on a device by UUID if it exists.
    /// This provides thread-safe access to device operations.
    pub fn with_device_by_uuid<F, R>(&self, device_uuid: &str, f: F) -> Option<R>
    where F: FnOnce(&SharedDeviceInfo) -> R {
        let devices = self.devices.read();
        devices.get(device_uuid).map(f)
    }

    /// Executes a mutable function on a device by UUID if it exists.
    /// This provides thread-safe mutable access to device operations.
    pub fn with_device_by_uuid_mut<F, R>(&self, device_uuid: &str, f: F) -> Option<R>
    where F: FnOnce(&mut SharedDeviceInfo) -> R {
        let mut devices = self.devices.write();
        devices.get_mut(device_uuid).map(f)
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

        Ok(Self { _shmem: shmem, ptr })
    }

    /// Creates a new shared memory segment.
    pub fn create(identifier: &str, configs: &[DeviceConfig]) -> Result<Self> {
        let shmem = match ShmemConf::new()
            .size(std::mem::size_of::<SharedDeviceState>())
            .os_id(identifier)
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

        let ptr = shmem.as_ptr() as *mut SharedDeviceState;

        // Initialize the shared memory data.
        unsafe {
            ptr.write(SharedDeviceState::new(configs));
        }

        info!(
            identifier = %identifier,
            "Created shared memory segment"
        );

        Ok(Self { _shmem: shmem, ptr })
    }

    /// Gets a pointer to the shared device state.
    pub fn get_ptr(&self) -> *mut SharedDeviceState {
        self.ptr
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
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use similar_asserts::assert_eq;

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
    fn shared_device_state_creation() {
        let configs = create_test_configs();
        let state = SharedDeviceState::new(&configs);

        // Test initial state
        assert_eq!(state.device_count(), 1, "Should start with one device");
        assert_eq!(
            state.get_last_heartbeat(),
            0,
            "Should start with no heartbeat"
        );
        assert!(
            !state.is_healthy(30),
            "Should not be healthy without heartbeat"
        );

        // Add a device
        let device_uuid = &configs[0].device_uuid;
        assert!(state.has_device(device_uuid), "Should contain the device");

        // Test heartbeat
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        state.update_heartbeat(timestamp);
        assert_eq!(
            state.get_last_heartbeat(),
            timestamp,
            "Heartbeat should be updated"
        );
        assert!(
            state.is_healthy(30),
            "Should be healthy with recent heartbeat"
        );
    }

    #[test]
    fn shared_device_state_device_operations() {
        let state = SharedDeviceState::new(&create_test_configs());
        let device_uuid = "test-device-uuid".to_string();

        // Add device
        let device_state = SharedDeviceInfo::new(TEST_TOTAL_CORES, TEST_UP_LIMIT, TEST_MEM_LIMIT);
        state.add_device(device_uuid.clone(), device_state);

        // Test device operations using with_device_by_uuid
        let available_cores =
            state.with_device_by_uuid(&device_uuid, |device| device.get_available_cores());
        assert_eq!(
            available_cores,
            Some(0),
            "Available cores should be 0 initially"
        );

        // Test mutable operations using with_device_by_uuid_mut
        let result = state.with_device_by_uuid_mut(&device_uuid, |device| {
            device.set_available_cores(100);
            device.get_available_cores()
        });
        assert_eq!(
            result,
            Some(100),
            "Available cores should be updated to 100"
        );

        // Test device removal
        let removed_device = state.remove_device(&device_uuid);
        assert!(removed_device.is_some(), "Should return the removed device");
        assert_eq!(
            state.device_count(),
            0,
            "Should have no devices after removal"
        );
        assert!(
            !state.has_device(&device_uuid),
            "Should not contain the device after removal"
        );
    }

    #[test]
    fn shared_device_state_multiple_devices() {
        let state = SharedDeviceState::new(&[]);

        // Add multiple devices
        for i in 0..3 {
            let device_uuid = format!("device-{i}");
            let device_state = SharedDeviceInfo::new(
                1024 + i as u32 * 512,
                80 + i as u32 * 5,
                (1024 + i as u64 * 512) * 1024 * 1024,
            );
            state.add_device(device_uuid, device_state);
        }

        assert_eq!(state.device_count(), 3, "Should have 3 devices");

        let device_uuids = state.get_device_uuids();
        assert_eq!(device_uuids.len(), 3, "Should return 3 device UUIDs");

        // Test operations on specific devices
        let cores = state.with_device_by_uuid("device-1", |device| device.get_total_cores());
        assert_eq!(
            cores,
            Some(1024 + 512),
            "Device 1 should have correct total cores"
        );

        // Update available cores for device-2
        state.with_device_by_uuid_mut("device-2", |device| {
            device.set_available_cores(256);
        });

        let available_cores =
            state.with_device_by_uuid("device-2", |device| device.get_available_cores());
        assert_eq!(
            available_cores,
            Some(256),
            "Device 2 should have updated available cores"
        );
    }

    #[test]
    fn shared_device_state_atomic_operations() {
        let device_state = SharedDeviceInfo::new(TEST_TOTAL_CORES, TEST_UP_LIMIT, TEST_MEM_LIMIT);

        // Test available cores operations
        device_state.set_available_cores(512);
        assert_eq!(
            device_state.get_available_cores(),
            512,
            "Available cores should be updated correctly"
        );

        device_state.set_available_cores(-100);
        assert_eq!(
            device_state.get_available_cores(),
            -100,
            "Available cores should support negative values"
        );

        // Test up limit operations
        device_state.set_up_limit(90);
        assert_eq!(
            device_state.get_up_limit(),
            90,
            "Up limit should be updated correctly"
        );

        device_state.set_up_limit(0);
        assert_eq!(
            device_state.get_up_limit(),
            0,
            "Up limit should support zero value"
        );

        // Test memory limit operations
        let new_mem_limit = 2 * 1024 * 1024 * 1024; // 2GB
        device_state.set_mem_limit(new_mem_limit);
        assert_eq!(
            device_state.get_mem_limit(),
            new_mem_limit,
            "Memory limit should be updated correctly"
        );
    }

    #[test]
    fn shared_device_state_concurrent_access() {
        let state = Arc::new(SharedDeviceState::new(&create_test_configs()));
        let device_uuid = "test-device-uuid".to_string();

        // Add device to the state
        let device_state = SharedDeviceInfo::new(TEST_TOTAL_CORES, TEST_UP_LIMIT, TEST_MEM_LIMIT);
        state.add_device(device_uuid.clone(), device_state);

        let mut handles = vec![];

        // Spawn multiple threads to test atomic operations
        for i in 0..10 {
            let state_clone = Arc::clone(&state);
            let device_uuid_clone = device_uuid.clone();
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let value = i * 100 + j;
                    state_clone.with_device_by_uuid_mut(&device_uuid_clone, |device| {
                        device.set_available_cores(value);
                    });

                    if let Some(read_value) = state_clone
                        .with_device_by_uuid(&device_uuid_clone, |device| {
                            device.get_available_cores()
                        })
                    {
                        // Since we are accessing concurrently, the read value may not be the one just set,
                        // but it should be a valid value set by one of the threads.
                        assert!(
                            (0..1000).contains(&read_value),
                            "Read value should be within expected range"
                        );
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }

        // Final state should be one of the values set by the threads
        if let Some(final_value) =
            state.with_device_by_uuid(&device_uuid, |device| device.get_available_cores())
        {
            assert!(
                (0..1000).contains(&final_value),
                "Final value should be within expected range"
            );
        }
    }

    #[test]
    fn shared_memory_handle_creation() {
        let configs = create_test_configs();
        let identifier = create_unique_identifier("handle_create");

        // Create shared memory handle
        let handle = SharedMemoryHandle::create(&identifier, &configs)
            .expect("should create shared memory handle successfully");

        let state = handle.get_state();
        let device_uuid = &configs[0].device_uuid;
        state
            .with_device_by_uuid(device_uuid, |device| {
                assert_eq!(
                    device.get_total_cores(),
                    TEST_TOTAL_CORES,
                    "Total cores should match configuration"
                );
                assert_eq!(
                    device.get_up_limit(),
                    TEST_UP_LIMIT,
                    "Up limit should match configuration"
                );
                assert_eq!(
                    device.get_mem_limit(),
                    TEST_MEM_LIMIT,
                    "Memory limit should match configuration"
                );
            })
            .expect("device should exist");
    }

    #[test]
    fn shared_memory_handle_open_existing() {
        let configs = create_test_configs();
        let identifier = create_unique_identifier("handle_open");

        // Create shared memory handle first
        let _handle1 = SharedMemoryHandle::create(&identifier, &configs)
            .expect("should create shared memory handle successfully");

        // Open existing shared memory
        let handle2 = SharedMemoryHandle::open(&identifier)
            .expect("should open existing shared memory successfully");

        let state = handle2.get_state();
        let device_uuid = &configs[0].device_uuid;
        assert_eq!(
            state
                .with_device_by_uuid(device_uuid, |d| d.get_total_cores())
                .unwrap(),
            TEST_TOTAL_CORES,
            "Total cores should match original configuration"
        );
    }

    #[test]
    fn shared_memory_handle_concurrent_access() {
        let configs = create_test_configs();
        let identifier = create_unique_identifier("handle_concurrent");
        let device_uuid = configs[0].device_uuid.clone();

        // Create shared memory handle
        let handle = Arc::new(
            SharedMemoryHandle::create(&identifier, &configs)
                .expect("should create shared memory handle successfully"),
        );

        let mut handles = vec![];

        // Spawn multiple threads to test concurrent access
        for i in 0..5 {
            let handle_clone = Arc::clone(&handle);
            let device_uuid_clone = device_uuid.clone();
            let thread_handle = thread::spawn(move || {
                for j in 0..20 {
                    let value = i * 20 + j;
                    let state = handle_clone.get_state();
                    state
                        .with_device_by_uuid_mut(&device_uuid_clone, |d| {
                            d.set_available_cores(value)
                        })
                        .unwrap();

                    // Small delay to increase chance of interleaving
                    thread::sleep(Duration::from_millis(1));

                    let read_value = state
                        .with_device_by_uuid(&device_uuid_clone, |d| d.get_available_cores())
                        .unwrap();
                    // Due to concurrent access, the read value may not be the one we just set
                    assert!(read_value >= 0, "Read value should be non-negative");
                }
            });
            handles.push(thread_handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }

        // Verify final state is accessible
        let final_value = handle
            .get_state()
            .with_device_by_uuid(&device_uuid, |d| d.get_available_cores())
            .unwrap();
        assert!(final_value >= 0, "Final value should be non-negative");
    }

    #[test]
    fn shared_memory_handle_error_handling() {
        let non_existent_identifier = "non_existent_handle";

        // Try to open non-existent shared memory
        let result = SharedMemoryHandle::open(non_existent_identifier);
        assert!(
            result.is_err(),
            "Should return error when opening non-existent shared memory"
        );
    }

    #[test]
    fn boundary_conditions() {
        // Test with maximum values
        let max_uuid = "max-uuid".to_string();
        let max_configs = vec![DeviceConfig {
            device_idx: u32::MAX,
            device_uuid: max_uuid.clone(),
            up_limit: u32::MAX,
            mem_limit: u64::MAX,
            sm_count: u32::MAX,
            max_thread_per_sm: u32::MAX,
            total_cuda_cores: u32::MAX,
        }];

        let state = SharedDeviceState::new(&max_configs);

        state
            .with_device_by_uuid(&max_uuid, |device| {
                assert_eq!(
                    device.get_total_cores(),
                    u32::MAX,
                    "Should handle maximum u32 value for cores"
                );
                assert_eq!(
                    device.get_up_limit(),
                    u32::MAX,
                    "Should handle maximum u32 value for limit"
                );
                assert_eq!(
                    device.get_mem_limit(),
                    u64::MAX,
                    "Should handle maximum u64 value for memory"
                );
            })
            .unwrap();

        // Test with minimum values
        let min_uuid = "min-uuid".to_string();
        let min_configs = vec![DeviceConfig {
            device_idx: 0,
            device_uuid: min_uuid.clone(),
            up_limit: 0,
            mem_limit: 0,
            total_cuda_cores: 0,
            sm_count: 0,
            max_thread_per_sm: 0,
        }];

        let state = SharedDeviceState::new(&min_configs);

        state
            .with_device_by_uuid(&min_uuid, |device| {
                assert_eq!(device.get_total_cores(), 0, "Should handle zero cores");
                assert_eq!(device.get_up_limit(), 0, "Should handle zero limit");
                assert_eq!(device.get_mem_limit(), 0, "Should handle zero memory");
            })
            .unwrap();

        // Test available cores with extreme values
        state
            .with_device_by_uuid_mut(&min_uuid, |device| {
                device.set_available_cores(i32::MAX);
                assert_eq!(
                    device.get_available_cores(),
                    i32::MAX,
                    "Should handle maximum i32 value"
                );

                device.set_available_cores(i32::MIN);
                assert_eq!(
                    device.get_available_cores(),
                    i32::MIN,
                    "Should handle minimum i32 value"
                );
            })
            .unwrap();
    }

    #[test]
    fn cross_process_simulation() {
        let configs = create_test_configs();
        let identifier = create_unique_identifier("cross_process");

        // Simulate process 1: Create shared memory
        let handle1 = SharedMemoryHandle::create(&identifier, &configs)
            .expect("should create shared memory successfully");

        // Modify state in "process 1"
        let device_uuid = &configs[0].device_uuid;
        handle1
            .get_state()
            .with_device_by_uuid_mut(device_uuid, |d| {
                d.set_available_cores(200);
                d.set_up_limit(85);
            })
            .unwrap();

        // Simulate process 2: Open existing shared memory
        let handle2 = SharedMemoryHandle::open(&identifier)
            .expect("should open existing shared memory successfully");

        // Verify state is shared between "processes"
        handle2
            .get_state()
            .with_device_by_uuid(device_uuid, |d| {
                assert_eq!(
                    d.get_available_cores(),
                    200,
                    "Available cores should be shared"
                );
                assert_eq!(d.get_up_limit(), 85, "Up limit should be shared");
            })
            .unwrap();

        // Modify state in "process 2"
        handle2
            .get_state()
            .with_device_by_uuid_mut(device_uuid, |d| {
                d.set_available_cores(300);
                d.set_mem_limit(2 * 1024 * 1024 * 1024);
            })
            .unwrap();

        // Verify changes are visible in "process 1"
        handle1
            .get_state()
            .with_device_by_uuid(device_uuid, |d| {
                assert_eq!(
                    d.get_available_cores(),
                    300,
                    "Changes should be visible across processes"
                );
                assert_eq!(
                    d.get_mem_limit(),
                    2 * 1024 * 1024 * 1024,
                    "Memory limit changes should be visible"
                );
            })
            .unwrap();
    }

    #[test]
    fn shared_memory_handle_send_sync() {
        let configs = create_test_configs();
        let identifier = create_unique_identifier("send_sync");

        let handle = SharedMemoryHandle::create(&identifier, &configs)
            .expect("should create shared memory successfully");

        // Test that handle can be sent across threads
        let handle_arc = Arc::new(handle);
        let handle_clone = Arc::clone(&handle_arc);
        let device_uuid = configs[0].device_uuid.clone();

        let thread_handle = thread::spawn(move || {
            let state = handle_clone.get_state();
            state
                .with_device_by_uuid_mut(&device_uuid, |d| {
                    d.set_available_cores(42);
                })
                .unwrap();
            state
                .with_device_by_uuid(&device_uuid, |d| d.get_available_cores())
                .unwrap()
        });

        let result = thread_handle
            .join()
            .expect("Thread should complete successfully");
        assert_eq!(
            result, 42,
            "Should be able to access shared memory from different thread"
        );

        // Verify the change is visible in the original thread
        assert_eq!(
            handle_arc
                .get_state()
                .with_device_by_uuid(&configs[0].device_uuid, |d| d.get_available_cores())
                .unwrap(),
            42,
            "Changes should be visible across threads"
        );
    }

    // Tests for ThreadSafeSharedMemoryManager
    #[test]
    fn thread_safe_shared_memory_manager_creation() {
        let manager = ThreadSafeSharedMemoryManager::new();
        let memories = manager.active_memories.read();
        assert_eq!(memories.len(), 0);
    }

    #[test]
    fn thread_safe_shared_memory_manager_basic_operations() {
        let manager = ThreadSafeSharedMemoryManager::new();
        let configs = create_test_configs();
        let identifier = create_unique_identifier("thread_safe_basic");

        // Test creation
        manager
            .create_or_get_shared_memory(&identifier, &configs)
            .unwrap();
        assert!(manager.contains(&identifier));

        // Test get shared memory
        let ptr = manager.get_shared_memory(&identifier).unwrap();
        assert!(!ptr.is_null());

        // Test cleanup
        manager.cleanup(&identifier).unwrap();
        assert!(!manager.contains(&identifier));
    }

    #[test]
    fn thread_safe_shared_memory_manager_concurrent_access() {
        let manager = Arc::new(ThreadSafeSharedMemoryManager::new());
        let configs = create_test_configs();
        let identifier = create_unique_identifier("thread_safe_concurrent");

        // Create shared memory first
        manager
            .create_or_get_shared_memory(&identifier, &configs)
            .unwrap();

        let mut handles = vec![];

        // Spawn multiple threads to access the shared memory
        for i in 0..5 {
            let manager_clone = manager.clone();
            let identifier_clone = identifier.clone();

            let handle = thread::spawn(move || {
                // Each thread tries to get the shared memory multiple times
                for _ in 0..10 {
                    let result = manager_clone.get_shared_memory(&identifier_clone);
                    assert!(result.is_ok());

                    let ptr = result.unwrap();
                    assert!(!ptr.is_null());

                    // Small delay to increase chance of race conditions
                    thread::sleep(Duration::from_millis(1));
                }

                format!("Thread {i} completed")
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.contains("completed"));
        }

        // Clean up
        manager.cleanup(&identifier).unwrap();
    }

    #[test]
    fn thread_safe_shared_memory_manager_multiple_memories() {
        let manager = ThreadSafeSharedMemoryManager::new();
        let config1 = vec![DeviceConfig {
            device_idx: 0,
            device_uuid: "multi-id-1".to_string(),
            up_limit: 80,
            mem_limit: 1024 * 1024 * 1024,
            total_cuda_cores: 2048,
            sm_count: 10,
            max_thread_per_sm: 1024,
        }];
        let config2 = vec![DeviceConfig {
            device_idx: 1,
            device_uuid: "multi-id-2".to_string(),
            up_limit: 90,
            mem_limit: 2048 * 1024 * 1024,
            total_cuda_cores: 4096,
            sm_count: 10,
            max_thread_per_sm: 1024,
        }];

        let id1 = create_unique_identifier("thread_safe_multi_1");
        let id2 = create_unique_identifier("thread_safe_multi_2");

        // Create two different shared memories
        manager.create_or_get_shared_memory(&id1, &config1).unwrap();
        manager.create_or_get_shared_memory(&id2, &config2).unwrap();

        // Verify both exist
        assert!(manager.contains(&id1));
        assert!(manager.contains(&id2));

        // Verify configs are different
        let ptr1 = manager.get_shared_memory(&id1).unwrap();
        let ptr2 = manager.get_shared_memory(&id2).unwrap();
        unsafe {
            (*ptr1)
                .with_device_by_uuid("multi-id-1", |d| {
                    assert_eq!(d.get_up_limit(), 80);
                })
                .unwrap();
            (*ptr2)
                .with_device_by_uuid("multi-id-2", |d| {
                    assert_eq!(d.get_up_limit(), 90);
                })
                .unwrap();
        }

        // Get identifiers
        let identifiers = manager
            .active_memories
            .read()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(identifiers.len(), 2);
        assert!(identifiers.contains(&id1));
        assert!(identifiers.contains(&id2));

        // Clean up
        manager.cleanup(&id1).unwrap();
        manager.cleanup(&id2).unwrap();

        assert!(!manager.contains(&id1));
        assert!(!manager.contains(&id2));
    }

    #[test]
    fn thread_safe_shared_memory_manager_error_handling() {
        let manager = ThreadSafeSharedMemoryManager::new();
        let non_existent_id = "non_existent_memory";

        // Test getting non-existent shared memory
        let result = manager.get_shared_memory(non_existent_id);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Shared memory not found"));

        // Test getting config for non-existent memory
        {
            let memories = manager.active_memories.read();
            assert!(!memories.contains_key(non_existent_id));
        } // Read lock is dropped here

        // Test cleanup of non-existent memory (should not panic)
        let cleanup_result = manager.cleanup(non_existent_id);
        assert!(cleanup_result.is_ok());
    }

    #[test]
    fn thread_safe_shared_memory_manager_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<ThreadSafeSharedMemoryManager>();
        assert_sync::<ThreadSafeSharedMemoryManager>();
    }

    #[test]
    fn test_concurrent_creation_race_condition() {
        let manager = Arc::new(ThreadSafeSharedMemoryManager::new());
        let configs = create_test_configs();
        let identifier = create_unique_identifier("concurrent_creation_race");
        let num_threads = 10;
        let mut handles = vec![];

        for _ in 0..num_threads {
            let manager_clone = Arc::clone(&manager);
            let configs_clone = configs.clone();
            let identifier_clone = identifier.clone();
            let handle = thread::spawn(move || {
                let result =
                    manager_clone.create_or_get_shared_memory(&identifier_clone, &configs_clone);
                assert!(
                    result.is_ok(),
                    "create_or_get_shared_memory should not fail"
                );
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify that the shared memory was created and is accessible
        assert!(manager.contains(&identifier));
        let ptr = manager.get_shared_memory(&identifier).unwrap();
        unsafe {
            let state = &*ptr;
            state
                .with_device_by_uuid(&configs[0].device_uuid, |d| {
                    assert_eq!(d.get_mem_limit(), TEST_MEM_LIMIT);
                })
                .unwrap();
        }

        // Cleanup
        manager.cleanup(&identifier).unwrap();
        assert!(!manager.contains(&identifier));
    }

    #[test]
    fn test_cleanup_concurrency() {
        let manager = Arc::new(ThreadSafeSharedMemoryManager::new());
        let configs = create_test_configs();
        let identifier = create_unique_identifier("cleanup_concurrency");

        // Create the shared memory segment initially
        manager
            .create_or_get_shared_memory(&identifier, &configs)
            .unwrap();

        let (tx, rx) = std::sync::mpsc::channel();

        let manager_clone1 = Arc::clone(&manager);
        let identifier_clone1 = identifier.clone();
        let cleanup_handle = thread::spawn(move || {
            // Wait for the signal from the access thread
            rx.recv().unwrap();
            manager_clone1.cleanup(&identifier_clone1).unwrap();
        });

        let manager_clone2 = Arc::clone(&manager);
        let identifier_clone2 = identifier.clone();
        let access_handle = thread::spawn(move || {
            // Signal the cleanup thread to start
            tx.send(()).unwrap();
            // Repeatedly try to access the shared memory
            for _ in 0..100 {
                if manager_clone2
                    .get_shared_memory(&identifier_clone2)
                    .is_err()
                {
                    // Success: cleanup happened and we can no longer access the memory
                    return true;
                }
                thread::sleep(Duration::from_millis(10));
            }
            // Failure: we were always able to access the memory
            false
        });

        cleanup_handle.join().unwrap();
        let access_result = access_handle.join().unwrap();

        assert!(access_result, "Access should fail after cleanup");

        // Final check to ensure it's gone
        assert!(!manager.contains(&identifier));
    }

    #[test]
    fn test_multi_threaded_read_write_integration() {
        let manager = Arc::new(ThreadSafeSharedMemoryManager::new());
        let configs = create_test_configs();
        let identifier = create_unique_identifier("multi_thread_integration");

        // Create the shared memory segment
        manager
            .create_or_get_shared_memory(&identifier, &configs)
            .unwrap();

        let num_threads = 5;
        let mut handles = vec![];
        let device_uuid = configs[0].device_uuid.clone();

        for i in 0..num_threads {
            let identifier_clone = identifier.clone();
            let device_uuid_clone = device_uuid.clone();
            let handle = thread::spawn(move || {
                let handle = SharedMemoryHandle::open(&identifier_clone).unwrap();
                let device_info = handle.get_state();

                // Write a unique value from this thread
                device_info
                    .with_device_by_uuid_mut(&device_uuid_clone, |d| d.set_available_cores(i))
                    .unwrap();
                thread::sleep(Duration::from_millis(20)); // Allow time for other threads to see it

                // Read the value and see if it has been changed by another thread
                let value = device_info
                    .with_device_by_uuid(&device_uuid_clone, |d| d.get_available_cores())
                    .unwrap();
                (i, value)
            });
            handles.push(handle);
        }

        let mut results = vec![];
        for handle in handles {
            results.push(handle.join().unwrap());
        }

        // Check that the final value is one of the values written by the threads
        let final_value = SharedMemoryHandle::open(&identifier)
            .unwrap()
            .get_state()
            .with_device_by_uuid(&device_uuid, |d| d.get_available_cores())
            .unwrap();
        assert!((0..num_threads).contains(&final_value));

        // Cleanup
        manager.cleanup(&identifier).unwrap();
    }

    #[test]
    fn test_new_shared_device_state_structure() {
        let state = SharedDeviceState::new(&[]);

        // Test initial state
        assert_eq!(state.device_count(), 0);
        assert_eq!(state.get_last_heartbeat(), 0);
        assert!(state.get_device_uuids().is_empty());

        // Add devices with different UUIDs
        let device1 = SharedDeviceInfo::new(1024, 80, 1024 * 1024 * 1024);
        let device2 = SharedDeviceInfo::new(2048, 90, 2048 * 1024 * 1024);

        state.add_device("gpu-0".to_string(), device1);
        state.add_device("gpu-1".to_string(), device2);

        // Test device management
        assert_eq!(state.device_count(), 2);
        assert!(state.has_device("gpu-0"));
        assert!(state.has_device("gpu-1"));
        assert!(!state.has_device("gpu-2"));

        let device_uuids = state.get_device_uuids();
        assert_eq!(device_uuids.len(), 2);
        assert!(device_uuids.contains(&"gpu-0".to_string()));
        assert!(device_uuids.contains(&"gpu-1".to_string()));

        // Test device operations
        state.with_device_by_uuid_mut("gpu-0", |device| {
            device.set_available_cores(100);
        });

        let cores = state.with_device_by_uuid("gpu-0", |device| device.get_available_cores());
        assert_eq!(cores, Some(100));

        // Test heartbeat functionality
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        state.update_heartbeat(now);
        assert_eq!(state.get_last_heartbeat(), now);
        assert!(state.is_healthy(30)); // Should be healthy within 30 seconds

        // Test unhealthy state with old heartbeat
        state.update_heartbeat(now - 60); // 60 seconds ago
        assert!(!state.is_healthy(30)); // Should be unhealthy

        // Test device removal
        let removed = state.remove_device("gpu-0");
        assert!(removed.is_some());
        assert_eq!(state.device_count(), 1);
        assert!(!state.has_device("gpu-0"));
        assert!(state.has_device("gpu-1"));

        // Test with_device_by_uuid on removed device
        let result = state.with_device_by_uuid("gpu-0", |device| device.get_available_cores());
        assert_eq!(result, None);
    }
}
