//! ERL (Elastic Rate Limiting) adapter
//!
//! Implement ERL SharedStorage trait for SharedMemoryHandle

use std::sync::Arc;

use erl::{ErlError, SharedStorage};
use error_stack::{Report, Result};

use super::handle::SharedMemoryHandle;

/// ERL adapter for SharedMemoryHandle
///
/// Bridge shared memory access to ERL's SharedStorage trait
pub struct ErlSharedMemoryAdapter {
    handle: Arc<SharedMemoryHandle>,
}

impl ErlSharedMemoryAdapter {
    /// Create a new ERL adapter
    pub fn new(handle: Arc<SharedMemoryHandle>) -> Self {
        Self { handle }
    }
}

impl SharedStorage<usize> for ErlSharedMemoryAdapter {
    /// Load token bucket state (limiter and hypervisor need)
    fn load_token_state(&self, key: &usize) -> Result<(f64, f64), ErlError> {
        self.handle
            .get_state()
            .with_device_v2(*key, |device| device.device_info.load_erl_token_state())
            .ok_or_else(|| {
                Report::new(ErlError::MonitoringFailed {
                    reason: format!("Device {key} not found or not using V2"),
                })
            })
    }

    /// Save token bucket state (mainly used by limiter)
    fn save_token_state(&self, key: &usize, tokens: f64, timestamp: f64) -> Result<(), ErlError> {
        self.handle
            .get_state()
            .with_device_v2(*key, |device| {
                device.device_info.store_erl_token_state(tokens, timestamp);
            })
            .ok_or_else(|| {
                Report::new(ErlError::MonitoringFailed {
                    reason: format!("Device {key} not found or not using V2"),
                })
            })
    }

    /// Load quota information (limiter read, hypervisor set)
    fn load_quota(&self, key: &usize) -> Result<(f64, f64), ErlError> {
        self.handle
            .get_state()
            .with_device_v2(*key, |device| device.device_info.load_erl_quota())
            .ok_or_else(|| {
                Report::new(ErlError::MonitoringFailed {
                    reason: format!("Device {key} not found or not using V2"),
                })
            })
    }

    /// Set quota information (only used by hypervisor)
    fn set_quota(&self, key: &usize, capacity: f64, refill_rate: f64) -> Result<(), ErlError> {
        self.handle
            .get_state()
            .with_device_v2(*key, |device| {
                device.device_info.set_erl_token_capacity(capacity);
                device.device_info.set_erl_token_refill_rate(refill_rate);
            })
            .ok_or_else(|| {
                Report::new(ErlError::InvalidConfiguration {
                    reason: format!("Device {key} not found or not using V2"),
                })
            })
    }

    /// Load current average cost (limiter read, hypervisor update)
    fn load_avg_cost(&self, key: &usize) -> Result<f64, ErlError> {
        self.handle
            .get_state()
            .with_device_v2(*key, |device| device.device_info.get_erl_avg_cost())
            .ok_or_else(|| {
                Report::new(ErlError::MonitoringFailed {
                    reason: format!("Device {key} not found or not using V2"),
                })
            })
    }

    /// Save current average cost (only used by hypervisor)
    fn save_avg_cost(&self, key: &usize, avg_cost: f64) -> Result<(), ErlError> {
        self.handle
            .get_state()
            .with_device_v2(*key, |device| {
                device.device_info.set_erl_avg_cost(avg_cost);
            })
            .ok_or_else(|| {
                Report::new(ErlError::CongestionControlFailed {
                    reason: format!("Device {key} not found or not using V2"),
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared_memory::handle::SharedMemoryHandle;

    #[test]
    fn test_erl_adapter_basic_operations() {
        // Create test shared memory with V2 (ERL support)
        let handle =
            SharedMemoryHandle::mock("/tmp/test_erl_basic", vec![(0, "test-device".to_string())]);
        let adapter = ErlSharedMemoryAdapter::new(Arc::new(handle));

        // Test quota setting and reading
        adapter.set_quota(&0, 100.0, 1.0).expect("should set quota");
        let (capacity, refill_rate) = adapter.load_quota(&0).expect("should load quota");
        assert_eq!(capacity, 100.0);
        assert_eq!(refill_rate, 1.0);

        // Test average cost setting and reading
        adapter
            .save_avg_cost(&0, 2.5)
            .expect("should save avg cost");
        let avg_cost = adapter.load_avg_cost(&0).expect("should load avg cost");
        assert_eq!(avg_cost, 2.5);

        // Test token state setting and reading
        adapter
            .save_token_state(&0, 50.0, 1234567890.0)
            .expect("should save token state");
        let (tokens, timestamp) = adapter
            .load_token_state(&0)
            .expect("should load token state");
        assert_eq!(tokens, 50.0);
        assert_eq!(timestamp, 1234567890.0);
    }

    #[test]
    fn test_erl_adapter_device_not_found() {
        let handle = SharedMemoryHandle::mock(
            "/tmp/test_erl_notfound",
            vec![(0, "test-device".to_string())],
        );
        let adapter = ErlSharedMemoryAdapter::new(Arc::new(handle));

        // Test access to non-existent device
        let result = adapter.load_quota(&999);
        assert!(result.is_err());
    }
}
