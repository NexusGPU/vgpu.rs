//! Shared Memory Module
//! This module provides utilities for managing shared memory segments used for
//! GPU resource coordination between processes.

use std::collections::HashMap;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::RwLock;

use anyhow::Context;
use anyhow::Result;
use shared_memory::Shmem;
use shared_memory::ShmemConf;
use shared_memory::ShmemError;
use tracing::info;
use tracing::warn;

/// Device state stored in shared memory.
#[repr(C)]
#[derive(Debug)]
pub struct SharedDeviceState {
    /// Currently available CUDA cores.
    pub available_cuda_cores: AtomicI32,
    /// Utilization limit percentage (0-100).
    pub up_limit: AtomicU32,
    /// Memory limit in bytes.
    pub mem_limit: AtomicU64,
    /// Total number of CUDA cores.
    pub total_cuda_cores: AtomicU32,
    /// Device index.
    pub device_idx: AtomicU32,
    /// Current pod memory usage in bytes.
    pub pod_memory_used: AtomicU64,
    /// Last memory update timestamp.
    pub last_memory_update: AtomicU64,
}

impl SharedDeviceState {
    /// Creates a new `SharedDeviceState`.
    pub fn new(device_idx: u32, total_cuda_cores: u32, up_limit: u32, mem_limit: u64) -> Self {
        Self {
            available_cuda_cores: AtomicI32::new(0),
            up_limit: AtomicU32::new(up_limit),
            mem_limit: AtomicU64::new(mem_limit),
            total_cuda_cores: AtomicU32::new(total_cuda_cores),
            device_idx: AtomicU32::new(device_idx),
            pod_memory_used: AtomicU64::new(0),
            last_memory_update: AtomicU64::new(0),
        }
    }

    /// Gets the number of available CUDA cores.
    pub fn get_available_cores(&self) -> i32 {
        self.available_cuda_cores.load(Ordering::Acquire)
    }

    /// Sets the number of available CUDA cores.
    pub fn set_available_cores(&self, cores: i32) {
        self.available_cuda_cores.store(cores, Ordering::Release);
    }

    /// Gets the utilization limit.
    pub fn get_up_limit(&self) -> u32 {
        self.up_limit.load(Ordering::Acquire)
    }

    /// Sets the utilization limit.
    pub fn set_up_limit(&self, limit: u32) {
        self.up_limit.store(limit, Ordering::Release);
    }

    /// Gets the memory limit.
    pub fn get_mem_limit(&self) -> u64 {
        self.mem_limit.load(Ordering::Acquire)
    }

    /// Sets the memory limit.
    pub fn set_mem_limit(&self, limit: u64) {
        self.mem_limit.store(limit, Ordering::Release);
    }

    /// Gets the total number of CUDA cores.
    pub fn get_total_cores(&self) -> u32 {
        self.total_cuda_cores.load(Ordering::Acquire)
    }

    /// Gets the device index.
    pub fn get_device_idx(&self) -> u32 {
        self.device_idx.load(Ordering::Acquire)
    }

    /// Gets the pod memory usage.
    pub fn get_pod_memory_used(&self) -> u64 {
        self.pod_memory_used.load(Ordering::Acquire)
    }

    /// Updates the pod memory usage and timestamp.
    pub fn update_pod_memory_used(&self, used: u64, timestamp: u64) {
        self.pod_memory_used.store(used, Ordering::Release);
        self.last_memory_update.store(timestamp, Ordering::Release);
    }

    /// Gets the last memory update timestamp.
    pub fn get_last_memory_update(&self) -> u64 {
        self.last_memory_update.load(Ordering::Acquire)
    }
}

/// Device configuration information.
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub device_idx: u32,
    pub up_limit: u32,
    pub mem_limit: u64,
    pub total_cuda_cores: u32,
}

/// Manages shared memory segments.
pub struct SharedMemoryManager {
    /// Shared memory map: identifier -> Shmem
    shared_memories: RwLock<HashMap<String, Shmem>>,
}

/// A thread-safe shared memory manager.
pub struct ThreadSafeSharedMemoryManager {
    /// Active shared memory segments: identifier -> (Shmem, DeviceConfig)
    active_memories: RwLock<HashMap<String, (Shmem, DeviceConfig)>>,
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
        config: &DeviceConfig,
    ) -> Result<()> {
        let mut memories = self.active_memories.write().unwrap();

        // Check if the segment already exists.
        if memories.contains_key(identifier) {
            return Ok(());
        }

        // Create a new shared memory segment.
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

        // Initialize the shared memory data.
        let ptr = shmem.as_ptr() as *mut SharedDeviceState;
        unsafe {
            ptr.write(SharedDeviceState::new(
                config.device_idx,
                config.total_cuda_cores,
                config.up_limit,
                config.mem_limit,
            ));
        }

        // Store the Shmem object and configuration.
        memories.insert(identifier.to_string(), (shmem, config.clone()));

        info!(
            identifier = %identifier,
            device_idx = config.device_idx,
            "Created shared memory segment"
        );

        Ok(())
    }

    /// Gets a pointer to the shared memory by its identifier.
    pub fn get_shared_memory(&self, identifier: &str) -> Result<*mut SharedDeviceState> {
        let memories = self.active_memories.read().unwrap();

        if let Some((shmem, _)) = memories.get(identifier) {
            let ptr = shmem.as_ptr() as *mut SharedDeviceState;
            Ok(ptr)
        } else {
            Err(anyhow::anyhow!("Shared memory not found: {}", identifier))
        }
    }

    /// Cleans up a shared memory segment.
    pub fn cleanup(&self, identifier: &str) -> Result<()> {
        let mut memories = self.active_memories.write().unwrap();

        if let Some((shmem, _)) = memories.remove(identifier) {
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
        let memories = self.active_memories.read().unwrap();
        memories.contains_key(identifier)
    }

    /// Gets all shared memory identifiers.
    pub fn get_all_identifiers(&self) -> Vec<String> {
        let memories = self.active_memories.read().unwrap();
        memories.keys().cloned().collect()
    }

    /// Gets the device configuration for a shared memory segment.
    pub fn get_device_config(&self, identifier: &str) -> Option<DeviceConfig> {
        let memories = self.active_memories.read().unwrap();
        memories.get(identifier).map(|(_, config)| config.clone())
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

impl SharedMemoryManager {
    /// Creates a new shared memory manager.
    pub fn new() -> Self {
        Self {
            shared_memories: RwLock::new(HashMap::new()),
        }
    }

    /// Creates or gets a shared memory segment.
    pub fn create_or_get_shared_memory(
        &self,
        identifier: &str,
        config: &DeviceConfig,
    ) -> Result<()> {
        let mut memories = self.shared_memories.write().unwrap();

        // Check if the segment already exists.
        if memories.contains_key(identifier) {
            return Ok(());
        }

        // Create a new shared memory segment.
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

        // Initialize the shared memory data.
        let ptr = shmem.as_ptr() as *mut SharedDeviceState;
        unsafe {
            ptr.write(SharedDeviceState::new(
                config.device_idx,
                config.total_cuda_cores,
                config.up_limit,
                config.mem_limit,
            ));
        }

        memories.insert(identifier.to_string(), shmem);

        info!(
            identifier = %identifier,
            device_idx = config.device_idx,
            "Created shared memory segment"
        );

        Ok(())
    }

    /// Gets a pointer to the shared memory by its identifier.
    pub fn get_shared_memory(&self, identifier: &str) -> Result<*mut SharedDeviceState> {
        let memories = self.shared_memories.read().unwrap();

        if let Some(shmem) = memories.get(identifier) {
            let ptr = shmem.as_ptr() as *mut SharedDeviceState;
            Ok(ptr)
        } else {
            Err(anyhow::anyhow!("Shared memory not found: {}", identifier))
        }
    }

    /// Cleans up a shared memory segment.
    pub fn cleanup(&self, identifier: &str) -> Result<()> {
        let mut memories = self.shared_memories.write().unwrap();

        if let Some(shmem) = memories.remove(identifier) {
            drop(shmem);
            info!(identifier = %identifier, "Cleaned up shared memory segment");
        } else {
            warn!(identifier = %identifier, "Attempted to cleanup non-existent shared memory");
        }

        Ok(())
    }

    /// Checks if a shared memory segment exists.
    pub fn contains(&self, identifier: &str) -> bool {
        let memories = self.shared_memories.read().unwrap();
        memories.contains_key(identifier)
    }

    /// Gets all shared memory identifiers.
    pub fn get_all_identifiers(&self) -> Vec<String> {
        let memories = self.shared_memories.read().unwrap();
        memories.keys().cloned().collect()
    }
}

impl Default for SharedMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

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
    pub fn create(identifier: &str, config: &DeviceConfig) -> Result<Self> {
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
            ptr.write(SharedDeviceState::new(
                config.device_idx,
                config.total_cuda_cores,
                config.up_limit,
                config.mem_limit,
            ));
        }

        info!(
            identifier = %identifier,
            device_idx = config.device_idx,
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

    fn create_test_config() -> DeviceConfig {
        DeviceConfig {
            device_idx: TEST_DEVICE_IDX,
            up_limit: TEST_UP_LIMIT,
            mem_limit: TEST_MEM_LIMIT,
            total_cuda_cores: TEST_TOTAL_CORES,
        }
    }

    fn create_unique_identifier(test_name: &str) -> String {
        format!("{TEST_IDENTIFIER}_{test_name}_{}", std::process::id())
    }

    #[test]
    fn shared_device_state_creation() {
        let state = SharedDeviceState::new(
            TEST_DEVICE_IDX,
            TEST_TOTAL_CORES,
            TEST_UP_LIMIT,
            TEST_MEM_LIMIT,
        );

        assert_eq!(
            state.get_device_idx(),
            TEST_DEVICE_IDX,
            "Device index should match initialization value"
        );
        assert_eq!(
            state.get_total_cores(),
            TEST_TOTAL_CORES,
            "Total cores should match initialization value"
        );
        assert_eq!(
            state.get_up_limit(),
            TEST_UP_LIMIT,
            "Up limit should match initialization value"
        );
        assert_eq!(
            state.get_mem_limit(),
            TEST_MEM_LIMIT,
            "Memory limit should match initialization value"
        );
        assert_eq!(
            state.get_available_cores(),
            0,
            "Available cores should be initialized to 0"
        );
    }

    #[test]
    fn shared_device_state_atomic_operations() {
        let state = SharedDeviceState::new(
            TEST_DEVICE_IDX,
            TEST_TOTAL_CORES,
            TEST_UP_LIMIT,
            TEST_MEM_LIMIT,
        );

        // Test available cores operations
        state.set_available_cores(512);
        assert_eq!(
            state.get_available_cores(),
            512,
            "Available cores should be updated correctly"
        );

        state.set_available_cores(-100);
        assert_eq!(
            state.get_available_cores(),
            -100,
            "Available cores should support negative values"
        );

        // Test up limit operations
        state.set_up_limit(90);
        assert_eq!(
            state.get_up_limit(),
            90,
            "Up limit should be updated correctly"
        );

        state.set_up_limit(0);
        assert_eq!(
            state.get_up_limit(),
            0,
            "Up limit should support zero value"
        );

        // Test memory limit operations
        let new_mem_limit = 2 * 1024 * 1024 * 1024; // 2GB
        state.set_mem_limit(new_mem_limit);
        assert_eq!(
            state.get_mem_limit(),
            new_mem_limit,
            "Memory limit should be updated correctly"
        );
    }

    #[test]
    fn shared_device_state_concurrent_access() {
        let state = Arc::new(SharedDeviceState::new(
            TEST_DEVICE_IDX,
            TEST_TOTAL_CORES,
            TEST_UP_LIMIT,
            TEST_MEM_LIMIT,
        ));
        let mut handles = vec![];

        // Spawn multiple threads to test atomic operations
        for i in 0..10 {
            let state_clone = Arc::clone(&state);
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let value = i * 100 + j;
                    state_clone.set_available_cores(value);
                    let read_value = state_clone.get_available_cores();
                    // Since we are accessing concurrently, the read value may not be the one just set,
                    // but it should be a valid value set by one of the threads.
                    assert!(
                        (0..1000).contains(&read_value),
                        "Read value should be within expected range"
                    );
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }

        // Final state should be one of the values set by the threads
        let final_value = state.get_available_cores();
        assert!(
            (0..1000).contains(&final_value),
            "Final value should be within expected range"
        );
    }

    #[test]
    fn shared_memory_manager_creation() {
        let manager = SharedMemoryManager::new();
        assert_eq!(
            manager.get_all_identifiers().len(),
            0,
            "New manager should have no shared memories"
        );
        assert!(
            !manager.contains("non_existent"),
            "Manager should not contain non-existent identifier"
        );
    }

    #[test]
    fn shared_memory_manager_create_and_access() {
        let manager = SharedMemoryManager::new();
        let config = create_test_config();
        let identifier = create_unique_identifier("create_access");

        // Create shared memory
        manager
            .create_or_get_shared_memory(&identifier, &config)
            .expect("should create shared memory successfully");

        assert!(
            manager.contains(&identifier),
            "Manager should contain created identifier"
        );
        assert_eq!(
            manager.get_all_identifiers().len(),
            1,
            "Manager should have one shared memory"
        );

        // Access shared memory
        let ptr = manager
            .get_shared_memory(&identifier)
            .expect("should retrieve shared memory pointer");

        unsafe {
            let state = &*ptr;
            assert_eq!(
                state.get_device_idx(),
                TEST_DEVICE_IDX,
                "Device index should match configuration"
            );
            assert_eq!(
                state.get_total_cores(),
                TEST_TOTAL_CORES,
                "Total cores should match configuration"
            );
        }

        // Cleanup
        manager
            .cleanup(&identifier)
            .expect("should cleanup shared memory successfully");

        assert!(
            !manager.contains(&identifier),
            "Manager should not contain cleaned up identifier"
        );
    }

    #[test]
    fn shared_memory_manager_duplicate_creation() {
        let manager = SharedMemoryManager::new();
        let config = create_test_config();
        let identifier = create_unique_identifier("duplicate");

        // Create shared memory twice
        manager
            .create_or_get_shared_memory(&identifier, &config)
            .expect("should create shared memory successfully");

        manager
            .create_or_get_shared_memory(&identifier, &config)
            .expect("should handle duplicate creation gracefully");

        assert_eq!(
            manager.get_all_identifiers().len(),
            1,
            "Manager should still have only one shared memory"
        );

        // Cleanup
        manager
            .cleanup(&identifier)
            .expect("should cleanup shared memory successfully");
    }

    #[test]
    fn shared_memory_manager_multiple_memories() {
        let manager = SharedMemoryManager::new();
        let config = create_test_config();
        let identifiers = vec![
            create_unique_identifier("multi_1"),
            create_unique_identifier("multi_2"),
            create_unique_identifier("multi_3"),
        ];

        // Create multiple shared memories
        for identifier in &identifiers {
            manager
                .create_or_get_shared_memory(identifier, &config)
                .expect("should create shared memory successfully");
        }

        assert_eq!(
            manager.get_all_identifiers().len(),
            3,
            "Manager should have three shared memories"
        );

        // Verify each shared memory
        for identifier in &identifiers {
            assert!(
                manager.contains(identifier),
                "Manager should contain all created identifiers"
            );
            let ptr = manager
                .get_shared_memory(identifier)
                .expect("should retrieve shared memory pointer");

            unsafe {
                let state = &*ptr;
                assert_eq!(
                    state.get_device_idx(),
                    TEST_DEVICE_IDX,
                    "Device index should match configuration"
                );
            }
        }

        // Cleanup all
        for identifier in &identifiers {
            manager
                .cleanup(identifier)
                .expect("should cleanup shared memory successfully");
        }

        assert_eq!(
            manager.get_all_identifiers().len(),
            0,
            "Manager should have no shared memories after cleanup"
        );
    }

    #[test]
    fn shared_memory_manager_error_handling() {
        let manager = SharedMemoryManager::new();

        // Try to access non-existent shared memory
        let result = manager.get_shared_memory("non_existent");
        assert!(
            result.is_err(),
            "Should return error for non-existent shared memory"
        );

        // Try to cleanup non-existent shared memory
        let result = manager.cleanup("non_existent");
        assert!(
            result.is_ok(),
            "Should handle cleanup of non-existent shared memory gracefully"
        );
    }

    #[test]
    fn shared_memory_handle_creation() {
        let config = create_test_config();
        let identifier = create_unique_identifier("handle_create");

        // Create shared memory handle
        let handle = SharedMemoryHandle::create(&identifier, &config)
            .expect("should create shared memory handle successfully");

        let state = handle.get_state();
        assert_eq!(
            state.get_device_idx(),
            TEST_DEVICE_IDX,
            "Device index should match configuration"
        );
        assert_eq!(
            state.get_total_cores(),
            TEST_TOTAL_CORES,
            "Total cores should match configuration"
        );
        assert_eq!(
            state.get_up_limit(),
            TEST_UP_LIMIT,
            "Up limit should match configuration"
        );
        assert_eq!(
            state.get_mem_limit(),
            TEST_MEM_LIMIT,
            "Memory limit should match configuration"
        );

        // Test pointer access
        let ptr = handle.get_ptr();
        unsafe {
            let state_from_ptr = &*ptr;
            assert_eq!(
                state_from_ptr.get_device_idx(),
                TEST_DEVICE_IDX,
                "Device index from pointer should match"
            );
        }
    }

    #[test]
    fn shared_memory_handle_open_existing() {
        let config = create_test_config();
        let identifier = create_unique_identifier("handle_open");

        // Create shared memory handle first
        let _handle1 = SharedMemoryHandle::create(&identifier, &config)
            .expect("should create shared memory handle successfully");

        // Open existing shared memory
        let handle2 = SharedMemoryHandle::open(&identifier)
            .expect("should open existing shared memory successfully");

        let state = handle2.get_state();
        assert_eq!(
            state.get_device_idx(),
            TEST_DEVICE_IDX,
            "Device index should match original configuration"
        );
        assert_eq!(
            state.get_total_cores(),
            TEST_TOTAL_CORES,
            "Total cores should match original configuration"
        );
    }

    #[test]
    fn shared_memory_handle_concurrent_access() {
        let config = create_test_config();
        let identifier = create_unique_identifier("handle_concurrent");

        // Create shared memory handle
        let handle = Arc::new(
            SharedMemoryHandle::create(&identifier, &config)
                .expect("should create shared memory handle successfully"),
        );

        let mut handles = vec![];

        // Spawn multiple threads to test concurrent access
        for i in 0..5 {
            let handle_clone = Arc::clone(&handle);
            let thread_handle = thread::spawn(move || {
                for j in 0..20 {
                    let value = i * 20 + j;
                    let state = handle_clone.get_state();
                    state.set_available_cores(value);

                    // Small delay to increase chance of interleaving
                    thread::sleep(Duration::from_millis(1));

                    let read_value = state.get_available_cores();
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
        let final_value = handle.get_state().get_available_cores();
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
        let max_config = DeviceConfig {
            device_idx: u32::MAX,
            up_limit: u32::MAX,
            mem_limit: u64::MAX,
            total_cuda_cores: u32::MAX,
        };

        let state = SharedDeviceState::new(
            max_config.device_idx,
            max_config.total_cuda_cores,
            max_config.up_limit,
            max_config.mem_limit,
        );

        assert_eq!(
            state.get_device_idx(),
            u32::MAX,
            "Should handle maximum u32 value"
        );
        assert_eq!(
            state.get_total_cores(),
            u32::MAX,
            "Should handle maximum u32 value for cores"
        );
        assert_eq!(
            state.get_up_limit(),
            u32::MAX,
            "Should handle maximum u32 value for limit"
        );
        assert_eq!(
            state.get_mem_limit(),
            u64::MAX,
            "Should handle maximum u64 value for memory"
        );

        // Test with minimum values
        let min_config = DeviceConfig {
            device_idx: 0,
            up_limit: 0,
            mem_limit: 0,
            total_cuda_cores: 0,
        };

        let state = SharedDeviceState::new(
            min_config.device_idx,
            min_config.total_cuda_cores,
            min_config.up_limit,
            min_config.mem_limit,
        );

        assert_eq!(state.get_device_idx(), 0, "Should handle minimum value");
        assert_eq!(state.get_total_cores(), 0, "Should handle zero cores");
        assert_eq!(state.get_up_limit(), 0, "Should handle zero limit");
        assert_eq!(state.get_mem_limit(), 0, "Should handle zero memory");

        // Test available cores with extreme values
        state.set_available_cores(i32::MAX);
        assert_eq!(
            state.get_available_cores(),
            i32::MAX,
            "Should handle maximum i32 value"
        );

        state.set_available_cores(i32::MIN);
        assert_eq!(
            state.get_available_cores(),
            i32::MIN,
            "Should handle minimum i32 value"
        );
    }

    #[test]
    fn cross_process_simulation() {
        let config = create_test_config();
        let identifier = create_unique_identifier("cross_process");

        // Simulate process 1: Create shared memory
        let handle1 = SharedMemoryHandle::create(&identifier, &config)
            .expect("should create shared memory successfully");

        // Modify state in "process 1"
        handle1.get_state().set_available_cores(200);
        handle1.get_state().set_up_limit(85);

        // Simulate process 2: Open existing shared memory
        let handle2 = SharedMemoryHandle::open(&identifier)
            .expect("should open existing shared memory successfully");

        // Verify state is shared between "processes"
        assert_eq!(
            handle2.get_state().get_available_cores(),
            200,
            "Available cores should be shared"
        );
        assert_eq!(
            handle2.get_state().get_up_limit(),
            85,
            "Up limit should be shared"
        );

        // Modify state in "process 2"
        handle2.get_state().set_available_cores(300);
        handle2.get_state().set_mem_limit(2 * 1024 * 1024 * 1024);

        // Verify changes are visible in "process 1"
        assert_eq!(
            handle1.get_state().get_available_cores(),
            300,
            "Changes should be visible across processes"
        );
        assert_eq!(
            handle1.get_state().get_mem_limit(),
            2 * 1024 * 1024 * 1024,
            "Memory limit changes should be visible"
        );
    }

    #[test]
    fn shared_memory_handle_send_sync() {
        let config = create_test_config();
        let identifier = create_unique_identifier("send_sync");

        let handle = SharedMemoryHandle::create(&identifier, &config)
            .expect("should create shared memory successfully");

        // Test that handle can be sent across threads
        let handle_arc = Arc::new(handle);
        let handle_clone = Arc::clone(&handle_arc);

        let thread_handle = thread::spawn(move || {
            let state = handle_clone.get_state();
            state.set_available_cores(42);
            state.get_available_cores()
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
            handle_arc.get_state().get_available_cores(),
            42,
            "Changes should be visible across threads"
        );
    }

    // Tests for ThreadSafeSharedMemoryManager
    #[test]
    fn thread_safe_shared_memory_manager_creation() {
        let manager = ThreadSafeSharedMemoryManager::new();
        assert_eq!(manager.get_all_identifiers().len(), 0);
    }

    #[test]
    fn thread_safe_shared_memory_manager_basic_operations() {
        let manager = ThreadSafeSharedMemoryManager::new();
        let config = create_test_config();
        let identifier = create_unique_identifier("thread_safe_basic");

        // Test creation
        manager
            .create_or_get_shared_memory(&identifier, &config)
            .unwrap();
        assert!(manager.contains(&identifier));

        // Test get config
        let retrieved_config = manager.get_device_config(&identifier).unwrap();
        assert_eq!(retrieved_config.device_idx, config.device_idx);
        assert_eq!(retrieved_config.up_limit, config.up_limit);

        // Test get shared memory
        let ptr = manager.get_shared_memory(&identifier).unwrap();
        assert!(!ptr.is_null());

        // Test cleanup
        manager.cleanup(&identifier).unwrap();
        assert!(!manager.contains(&identifier));
    }

    #[test]
    fn thread_safe_shared_memory_manager_concurrent_access() {
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        let manager = Arc::new(ThreadSafeSharedMemoryManager::new());
        let config = create_test_config();
        let identifier = create_unique_identifier("thread_safe_concurrent");

        // Create shared memory first
        manager
            .create_or_get_shared_memory(&identifier, &config)
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

                    // Verify we can read the device config
                    let config = manager_clone.get_device_config(&identifier_clone).unwrap();
                    assert_eq!(config.device_idx, TEST_DEVICE_IDX);

                    // Small delay to increase chance of race conditions
                    thread::sleep(Duration::from_millis(1));
                }

                format!("Thread {} completed", i)
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
        let config1 = DeviceConfig {
            device_idx: 0,
            up_limit: 80,
            mem_limit: 1024 * 1024 * 1024,
            total_cuda_cores: 2048,
        };
        let config2 = DeviceConfig {
            device_idx: 1,
            up_limit: 90,
            mem_limit: 2048 * 1024 * 1024,
            total_cuda_cores: 4096,
        };

        let id1 = create_unique_identifier("thread_safe_multi_1");
        let id2 = create_unique_identifier("thread_safe_multi_2");

        // Create two different shared memories
        manager.create_or_get_shared_memory(&id1, &config1).unwrap();
        manager.create_or_get_shared_memory(&id2, &config2).unwrap();

        // Verify both exist
        assert!(manager.contains(&id1));
        assert!(manager.contains(&id2));

        // Verify configs are different
        let retrieved_config1 = manager.get_device_config(&id1).unwrap();
        let retrieved_config2 = manager.get_device_config(&id2).unwrap();

        assert_eq!(retrieved_config1.device_idx, 0);
        assert_eq!(retrieved_config2.device_idx, 1);
        assert_eq!(retrieved_config1.up_limit, 80);
        assert_eq!(retrieved_config2.up_limit, 90);

        // Get identifiers
        let identifiers = manager.get_all_identifiers();
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
        let config = manager.get_device_config(non_existent_id);
        assert!(config.is_none());

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
}
