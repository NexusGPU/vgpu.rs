//! Test adapters implementing the dependency injection traits
//!
//! This module provides mock/test implementations of the DI traits for use
//! in testing environments and integration tests.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use utils::shared_memory::DeviceConfig;
use utils::shared_memory::PodIdentifier;

use super::pod_state_store::PodStateStore;
use super::traits::{DeviceSnapshotProvider, TimeSource};
use super::utilization::{DeviceSnapshot, ProcessUtilization};
use utils::shared_memory::traits::SharedMemoryAccess;

/// Mock device snapshot provider for testing
pub struct MockDeviceSnapshotProvider {
    snapshots: Arc<Mutex<HashMap<u32, DeviceSnapshot>>>,
    next_timestamp: Arc<Mutex<u64>>,
    error_mode: Arc<Mutex<bool>>,
}

impl MockDeviceSnapshotProvider {
    pub fn new() -> Self {
        Self {
            snapshots: Arc::new(Mutex::new(HashMap::new())),
            next_timestamp: Arc::new(Mutex::new(1000)),
            error_mode: Arc::new(Mutex::new(false)),
        }
    }

    /// Enable or disable error mode for testing error handling
    pub fn set_error_mode(&self, enabled: bool) {
        let mut error_mode = self.error_mode.lock().unwrap();
        *error_mode = enabled;
    }

    /// Helper method to set up device snapshot with PIDs and memories for testing
    pub fn set_device_snapshot(
        &self,
        device_idx: u32,
        timestamp: u64,
        pids: Vec<u32>,
        memories: Vec<u64>,
    ) {
        let mut process_utilizations = HashMap::new();
        let mut process_memories = HashMap::new();

        for (i, pid) in pids.iter().enumerate() {
            process_utilizations.insert(
                *pid,
                ProcessUtilization {
                    sm_util: 30 + (i as u32 * 10), // Varying utilization for each process
                    codec_util: 10 + (i as u32 * 5),
                },
            );

            if i < memories.len() {
                process_memories.insert(*pid, memories[i]);
            }
        }

        let snapshot = DeviceSnapshot {
            process_utilizations,
            process_memories,
            timestamp,
        };

        let mut snapshots = self.snapshots.lock().unwrap();
        snapshots.insert(device_idx, snapshot);
    }

    /// Set a mock snapshot directly for a device
    pub fn set_device_snapshot_direct(&self, device_idx: u32, snapshot: DeviceSnapshot) {
        let mut snapshots = self.snapshots.lock().unwrap();
        snapshots.insert(device_idx, snapshot);
    }

    /// Create a snapshot with synthetic utilization data for testing
    pub fn create_synthetic_snapshot(
        &self,
        device_idx: u32,
        process_utils: HashMap<u32, (u32, u32)>, // pid -> (sm_util, codec_util)
        process_memories: HashMap<u32, u64>,
    ) {
        let mut process_utilizations = HashMap::new();
        for (pid, (sm_util, codec_util)) in process_utils {
            process_utilizations.insert(
                pid,
                ProcessUtilization {
                    sm_util,
                    codec_util,
                },
            );
        }

        let snapshot = DeviceSnapshot {
            process_utilizations,
            process_memories,
            timestamp: self.get_next_timestamp(),
        };

        self.set_device_snapshot_direct(device_idx, snapshot);
    }

    fn get_next_timestamp(&self) -> u64 {
        let mut timestamp = self.next_timestamp.lock().unwrap();
        *timestamp += 1;
        *timestamp
    }
}

impl Default for MockDeviceSnapshotProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceSnapshotProvider for MockDeviceSnapshotProvider {
    type Error = String;

    fn get_device_snapshot(
        &self,
        device_idx: u32,
        _last_seen_ts: u64,
    ) -> Result<DeviceSnapshot, Self::Error> {
        // Check if error mode is enabled
        {
            let error_mode = self.error_mode.lock().unwrap();
            if *error_mode {
                return Err("Mock error mode enabled".to_string());
            }
        }

        let snapshots = self.snapshots.lock().unwrap();
        match snapshots.get(&device_idx) {
            Some(snapshot) => {
                // Preserve the explicitly set timestamp for deterministic tests
                Ok(snapshot.clone())
            }
            None => {
                // Return empty snapshot with a monotonically increasing timestamp
                Ok(DeviceSnapshot {
                    process_utilizations: HashMap::new(),
                    process_memories: HashMap::new(),
                    timestamp: self.get_next_timestamp(),
                })
            }
        }
    }
}

/// Mock time source for testing
pub struct MockTime {
    current_time: Arc<Mutex<u64>>,
}

impl MockTime {
    pub fn new(initial_time: u64) -> Self {
        Self {
            current_time: Arc::new(Mutex::new(initial_time)),
        }
    }

    /// Advance time by the given number of seconds
    pub fn advance(&self, seconds: u64) {
        let mut time = self.current_time.lock().unwrap();
        *time += seconds;
    }

    /// Set the current time
    pub fn set_time(&self, time: u64) {
        let mut current_time = self.current_time.lock().unwrap();
        *current_time = time;
    }
}

impl Default for MockTime {
    fn default() -> Self {
        Self::new(1_000_000) // Default to some reasonable timestamp
    }
}

impl TimeSource for MockTime {
    fn now_unix_secs(&self) -> u64 {
        *self.current_time.lock().unwrap()
    }
}

/// Mock shared memory access for testing
pub struct MockSharedMemoryAccess {
    // For simplicity, we'll just store that operations were called
    // Real tests would probably want to track the actual shared memory state
    operations: Arc<Mutex<Vec<String>>>,
}

impl MockSharedMemoryAccess {
    pub fn new() -> Self {
        Self {
            operations: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get list of operations that were performed
    pub fn get_operations(&self) -> Vec<String> {
        self.operations.lock().unwrap().clone()
    }

    fn log_operation(&self, operation: String) {
        self.operations.lock().unwrap().push(operation);
    }
}

impl Default for MockSharedMemoryAccess {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedMemoryAccess for MockSharedMemoryAccess {
    type Error = anyhow::Error;

    fn find_shared_memory_files(&self, glob: &str) -> Result<Vec<PathBuf>, Self::Error> {
        self.log_operation(format!("find_shared_memory_files({glob})"));
        Ok(vec![]) // Return empty list for testing
    }

    fn extract_identifier_from_path(
        &self,
        base_path: impl AsRef<Path>,
        path: impl AsRef<Path>,
    ) -> Result<PodIdentifier, Self::Error> {
        let path = path.as_ref();
        let base_path = base_path.as_ref();
        self.log_operation(format!(
            "extract_identifier_from_path({base_path:?}, {path:?})"
        ));

        // Try to extract relative path, fallback to filename if path doesn't start with base_path
        let identifier = if let Ok(relative_path) = path.strip_prefix(base_path) {
            relative_path.to_string_lossy().to_string()
        } else {
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown")
                .to_string()
        };

        PodIdentifier::from_path(&identifier).ok_or_else(|| {
            anyhow::anyhow!("Failed to parse PodIdentifier from path: {}", identifier)
        })
    }

    fn create_shared_memory(
        &self,
        pod_path: impl AsRef<Path>,
        _cfgs: &[DeviceConfig],
    ) -> Result<(), Self::Error> {
        self.log_operation(format!(
            "create_shared_memory({})",
            pod_path.as_ref().display()
        ));
        Ok(())
    }

    fn get_shared_memory(
        &self,
        pod_path: impl AsRef<Path>,
    ) -> Result<*const utils::shared_memory::SharedDeviceState, Self::Error> {
        self.log_operation(format!(
            "get_shared_memory({})",
            pod_path.as_ref().display()
        ));
        // Return a null pointer for testing - real tests would need proper shared memory
        Ok(std::ptr::null())
    }

    fn add_pid(&self, pod_path: impl AsRef<Path>, host_pid: usize) -> Result<(), Self::Error> {
        self.log_operation(format!(
            "add_pid({}, {host_pid})",
            pod_path.as_ref().display()
        ));
        Ok(())
    }

    fn remove_pid(&self, pod_path: impl AsRef<Path>, host_pid: usize) -> Result<(), Self::Error> {
        self.log_operation(format!(
            "remove_pid({}, {host_pid})",
            pod_path.as_ref().display()
        ));
        Ok(())
    }

    fn cleanup_orphaned_files<F, P>(
        &self,
        glob: &str,
        _should_remove: F,
        base_path: P,
    ) -> Result<Vec<PodIdentifier>, Self::Error>
    where
        F: Fn(&PodIdentifier) -> bool,
        P: AsRef<Path>,
    {
        self.log_operation(format!(
            "cleanup_orphaned_files({glob}, {:?})",
            base_path.as_ref()
        ));
        Ok(vec![]) // Return empty list for testing
    }

    fn cleanup_unused<F, P>(
        &self,
        _should_keep: F,
        base_path: P,
    ) -> Result<Vec<PodIdentifier>, Self::Error>
    where
        F: Fn(&PodIdentifier) -> bool,
        P: AsRef<Path>,
    {
        self.log_operation(format!("cleanup_unused({:?})", base_path.as_ref()));
        Ok(vec![]) // Return empty list for testing
    }
}

/// Test type alias for the coordinator with mock dependencies
pub type TestLimiterCoordinator = super::coordinator::LimiterCoordinator<
    MockSharedMemoryAccess,
    PodStateStore,
    MockDeviceSnapshotProvider,
    MockTime,
>;

impl TestLimiterCoordinator {
    /// Create a new test coordinator with mock dependencies
    pub fn new_test(
        watch_interval: std::time::Duration,
        device_count: u32,
        shared_memory_glob_pattern: String,
    ) -> (
        Self,
        Arc<MockSharedMemoryAccess>,
        Arc<PodStateStore>,
        Arc<MockDeviceSnapshotProvider>,
        Arc<MockTime>,
    ) {
        let shared_memory = Arc::new(MockSharedMemoryAccess::new());
        let base_path = PathBuf::from("/tmp/test_shm");
        let pod_state = Arc::new(PodStateStore::new(base_path.clone()));
        let snapshot = Arc::new(MockDeviceSnapshotProvider::new());
        let time = Arc::new(MockTime::default());

        let config = super::coordinator::CoordinatorConfig {
            watch_interval,
            device_count,
            shared_memory_glob_pattern,
            base_path,
        };

        let coordinator = Self::new(
            config,
            shared_memory.clone(),
            pod_state.clone(),
            snapshot.clone(),
            time.clone(),
        );

        (coordinator, shared_memory, pod_state, snapshot, time)
    }
}
