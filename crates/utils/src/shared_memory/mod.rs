use std::sync::atomic::AtomicI32;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

use tracing::warn;

use crate::shared_memory::mutex::ShmMutex;
use crate::shared_memory::set::Set;

pub mod bitmap;
pub mod handle;
pub mod manager;
pub mod mutex;
pub mod set;

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
    pub devices: [DeviceEntry; MAX_DEVICES],
    /// Number of active devices
    pub device_count: AtomicU32,
    /// Last heartbeat timestamp from hypervisor (for health monitoring).
    pub last_heartbeat: AtomicU64,
    /// Set of pids
    pub pids: ShmMutex<Set<usize, MAX_PROCESSES>>,
}

/// Versioned shared device state enum for future compatibility
#[repr(C)]
pub enum SharedDeviceState {
    V1(SharedDeviceStateV1),
}

impl SharedDeviceState {
    /// Creates a new SharedDeviceState instance (currently V1).
    pub fn new(configs: &[DeviceConfig]) -> Self {
        Self::V1(SharedDeviceStateV1::new(configs))
    }

    /// Gets the current version of the shared device state
    pub fn version(&self) -> u32 {
        match self {
            Self::V1(_) => 1,
        }
    }

    /// Delegates method calls to the appropriate version
    fn with_inner<T, F>(&self, f: F) -> T
    where
        F: FnOnce(&SharedDeviceStateV1) -> T,
    {
        match self {
            Self::V1(inner) => f(inner),
        }
    }

    // Delegate all methods to the inner version
    pub fn has_device(&self, index: usize) -> bool {
        self.with_inner(|inner| inner.has_device(index))
    }

    pub fn device_count(&self) -> usize {
        self.with_inner(|inner| inner.device_count())
    }

    pub fn update_heartbeat(&self, timestamp: u64) {
        self.with_inner(|inner| inner.update_heartbeat(timestamp))
    }

    pub fn get_last_heartbeat(&self) -> u64 {
        self.with_inner(|inner| inner.get_last_heartbeat())
    }

    pub fn is_healthy(&self, timeout: Duration) -> bool {
        self.with_inner(|inner| inner.is_healthy(timeout))
    }

    pub fn add_pid(&self, pid: usize) {
        self.with_inner(|inner| inner.add_pid(pid))
    }

    pub fn remove_pid(&self, pid: usize) {
        self.with_inner(|inner| inner.remove_pid(pid))
    }

    pub fn get_all_pids(&self) -> Vec<usize> {
        self.with_inner(|inner| inner.get_all_pids())
    }

    pub fn cleanup_orphaned_locks(&self) {
        self.with_inner(|inner| inner.cleanup_orphaned_locks())
    }

    /// Executes a closure with a device entry by index
    pub fn with_device<T, F>(&self, index: usize, f: F) -> Option<T>
    where
        F: FnOnce(&DeviceEntry) -> T,
    {
        match self {
            Self::V1(inner) => {
                inner.devices.get(index).filter(|device| device.is_active()).map(f)
            }
        }
    }

    /// Gets device information including additional fields for UI
    pub fn get_device_info(&self, index: usize) -> Option<(String, i32, u32, u64, u64, u32, bool)> {
        self.with_device(index, |device| {
            (
                device.get_uuid_owned(),
                device.device_info.get_available_cores(),
                device.device_info.get_total_cores(),
                device.device_info.get_mem_limit(),
                device.device_info.get_pod_memory_used(),
                device.device_info.get_up_limit(),
                device.is_active(),
            )
        })
    }

    /// Gets all detailed state information for TUI display
    pub fn get_detailed_state_info(&self) -> (u64, Vec<usize>, u32) {
        match self {
            Self::V1(inner) => (
                inner.get_last_heartbeat(),
                inner.get_all_pids(),
                1, // V1 version
            ),
        }
    }

    /// Gets version number
    pub fn get_version(&self) -> u32 {
        self.version()
    }

    /// Iterates over all active devices with their indices
    pub fn iter_active_devices(&self) -> impl Iterator<Item = (usize, &DeviceEntry)> {
        match self {
            Self::V1(inner) => inner.iter_active_devices(),
        }
    }

    /// Iterates over all devices (including inactive ones) with their indices
    pub fn iter_all_devices(&self) -> impl Iterator<Item = (usize, &DeviceEntry)> {
        match self {
            Self::V1(inner) => inner.iter_all_devices(),
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
            devices: std::array::from_fn(|_| DeviceEntry::new()),
            device_count: AtomicU32::new(configs.len() as u32),
            last_heartbeat: AtomicU64::new(now),
            pids: ShmMutex::new(Set::new()),
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
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let last_heartbeat = self.get_last_heartbeat();

        if last_heartbeat == 0 {
            return false; // No heartbeat recorded
        }

        now.saturating_sub(last_heartbeat) <= timeout.as_secs()
    }

    pub fn add_pid(&self, pid: usize) {
        self.pids.lock().insert(pid);
    }

    pub fn remove_pid(&self, pid: usize) {
        self.pids.lock().remove(pid);
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
    pub fn iter_active_devices(&self) -> impl Iterator<Item = (usize, &DeviceEntry)> {
        self.devices
            .iter()
            .enumerate()
            .filter(|(_, device)| device.is_active())
    }

    /// Iterates over all devices (including inactive ones) with their indices
    pub fn iter_all_devices(&self) -> impl Iterator<Item = (usize, &DeviceEntry)> {
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

    use crate::shared_memory::handle::SharedMemoryHandle;
    use crate::shared_memory::manager::ThreadSafeSharedMemoryManager;

    use super::*;

    const TEST_IDENTIFIER: &str = "test_shared_memory";
    const TEST_DEVICE_IDX: u32 = 0;
    const TEST_TOTAL_CORES: u32 = 1024;
    const TEST_UP_LIMIT: u32 = 80;
    const TEST_MEM_LIMIT: u64 = 1024 * 1024 * 1024; // 1GB

    /// Helper function to check if shared memory file exists in /dev/shm
    fn shared_memory_file_exists(identifier: &str) -> bool {
        let path = format!("/dev/shm/{identifier}");
        std::path::Path::new(&path).exists()
    }

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
        assert_eq!(state.version(), 1);
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
        assert_eq!(state1.version(), 1);
        assert_eq!(state1.device_count(), 1);

        // Verify shared memory file exists after creation
        assert!(shared_memory_file_exists(&identifier));

        // Open existing shared memory
        let handle2 = SharedMemoryHandle::open(&identifier)
            .expect("should open existing shared memory successfully");

        let state2 = handle2.get_state();
        assert_eq!(state2.version(), 1);
        assert_eq!(state2.device_count(), 1);

        // Verify they access the same memory
        let device_idx = configs[0].device_idx as usize;
        state1.with_device(device_idx, |device| {
            device.device_info.set_available_cores(42);
        });

        let cores = state2.with_device(device_idx, |device| {
            device.device_info.get_available_cores()
        });
        assert_eq!(cores, Some(42));

        // File should still exist while handles are active
        assert!(shared_memory_file_exists(&identifier));
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

        let device_idx = configs[0].device_idx as usize;
        let mut handles = vec![];

        // Spawn multiple threads doing concurrent access
        for i in 0..5 {
            let handle_clone = Arc::clone(&handle);

            let thread_handle = thread::spawn(move || {
                let state = handle_clone.get_state();

                for j in 0..20 {
                    let value = i * 20 + j;
                    state.with_device(device_idx, |device| {
                        device.device_info.set_available_cores(value);
                    });

                    thread::sleep(Duration::from_millis(1));

                    let read_value = state
                        .with_device(device_idx, |device| {
                            device.device_info.get_available_cores()
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

        // Verify shared memory file exists
        assert!(shared_memory_file_exists(&identifier));

        // Test getting shared memory
        let ptr = manager.get_shared_memory(&identifier).unwrap();
        assert!(!ptr.is_null());

        // Test accessing through pointer
        unsafe {
            let state = &*ptr;
            assert_eq!(state.version(), 1);
            assert_eq!(state.device_count(), 1);
        }

        // Test cleanup
        manager.cleanup(&identifier).unwrap();
        assert!(!manager.contains(&identifier));
        assert!(!shared_memory_file_exists(&identifier));
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
    fn orphaned_file_cleanup() {
        let manager = ThreadSafeSharedMemoryManager::new();

        // Create a fake orphaned file in /tmp (since we can't write to /dev/shm in tests)
        let test_file = "/tmp/test_orphaned_shm_file";
        std::fs::write(test_file, "fake shared memory data").unwrap();

        // Verify file exists
        assert!(std::path::Path::new(test_file).exists());

        // Test cleanup with a pattern that won't match
        let cleaned = manager
            .cleanup_orphaned_files("nonexistent_pattern")
            .unwrap();
        assert_eq!(cleaned.len(), 0);

        // File should still exist since pattern didn't match
        assert!(std::path::Path::new(test_file).exists());

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
        let SharedDeviceState::V1(inner) = &state;
        inner.devices[2].set_active(false);

        let active_after_deactivation: Vec<_> = state.iter_active_devices().collect();
        assert_eq!(active_after_deactivation.len(), 1);
        assert_eq!(active_after_deactivation[0].0, 0);
        assert_eq!(active_after_deactivation[0].1.get_uuid(), "device-0");
    }
}
