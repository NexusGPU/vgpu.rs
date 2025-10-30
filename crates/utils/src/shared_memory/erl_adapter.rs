//! ERL (Elastic Rate Limiting) adapter
//!
//! Implement ERL `DeviceBackend` trait for `SharedMemoryHandle`

use std::ops::Deref;
use std::sync::Arc;

use erl::{DeviceBackend, DeviceQuota, ErlError, TokenState};
use error_stack::Report;
pub type Result<T, C> = core::result::Result<T, Report<C>>;

use super::handle::SharedMemoryHandle;

/// Trait to abstract access to SharedMemoryHandle
pub trait SharedMemoryAccess: Send + Sync {
    fn get_handle(&self) -> &SharedMemoryHandle;
}

impl SharedMemoryAccess for Arc<SharedMemoryHandle> {
    fn get_handle(&self) -> &SharedMemoryHandle {
        self.deref()
    }
}

impl SharedMemoryAccess for &SharedMemoryHandle {
    fn get_handle(&self) -> &SharedMemoryHandle {
        self
    }
}

/// ERL adapter for SharedMemoryHandle
///
/// Bridge shared memory access to ERL's `DeviceBackend` trait.
pub struct ErlSharedMemoryAdapter<H: SharedMemoryAccess> {
    handle: H,
}

impl<H: SharedMemoryAccess> ErlSharedMemoryAdapter<H> {
    /// Create a new ERL adapter
    pub fn new(handle: H) -> Self {
        Self { handle }
    }
}

impl<H: SharedMemoryAccess> DeviceBackend for ErlSharedMemoryAdapter<H> {
    fn read_token_state(&self, device: usize) -> Result<TokenState, ErlError> {
        self.handle
            .get_handle()
            .get_state()
            .with_device_v2(device, |dev| {
                let (tokens, ts) = dev.device_info.load_erl_token_state();
                TokenState::new(tokens, ts)
            })
            .ok_or_else(|| {
                Report::new(ErlError::storage(format!(
                    "Device {device} not found or not using V2"
                )))
            })
    }

    fn write_token_state(&self, device: usize, state: TokenState) -> Result<(), ErlError> {
        self.handle
            .get_handle()
            .get_state()
            .with_device_v2(device, |dev| {
                dev.device_info
                    .store_erl_token_state(state.tokens, state.last_update);
            })
            .ok_or_else(|| {
                Report::new(ErlError::storage(format!(
                    "Device {device} not found or not using V2"
                )))
            })
    }

    fn read_quota(&self, device: usize) -> Result<DeviceQuota, ErlError> {
        self.handle
            .get_handle()
            .get_state()
            .with_device_v2(device, |dev| {
                let (capacity, rate) = dev.device_info.load_erl_quota();
                DeviceQuota::new(capacity, rate)
            })
            .ok_or_else(|| {
                Report::new(ErlError::storage(format!(
                    "Device {device} not found or not using V2"
                )))
            })
    }

    fn write_refill_rate(&self, device: usize, refill_rate: f64) -> Result<(), ErlError> {
        self.handle
            .get_handle()
            .get_state()
            .with_device_v2(device, |dev| {
                dev.device_info.set_erl_token_refill_rate(refill_rate);
            })
            .ok_or_else(|| {
                Report::new(ErlError::invalid_config(format!(
                    "Device {device} not found or not using V2"
                )))
            })
    }

    fn write_capacity(&self, device: usize, capacity: f64) -> Result<(), ErlError> {
        self.handle
            .get_handle()
            .get_state()
            .with_device_v2(device, |dev| {
                dev.device_info.set_erl_token_capacity(capacity);
            })
            .ok_or_else(|| {
                Report::new(ErlError::invalid_config(format!(
                    "Device {device} not found or not using V2"
                )))
            })
    }

    fn fetch_sub_tokens(&self, device: usize, cost: f64) -> Result<f64, ErlError> {
        self.handle
            .get_handle()
            .get_state()
            .with_device_v2(device, |dev| dev.device_info.fetch_sub_erl_tokens(cost))
            .ok_or_else(|| {
                Report::new(ErlError::storage(format!(
                    "Device {device} not found or not using V2"
                )))
            })
    }

    fn fetch_add_tokens(&self, device: usize, amount: f64) -> Result<f64, ErlError> {
        self.handle
            .get_handle()
            .get_state()
            .with_device_v2(device, |dev| dev.device_info.fetch_add_erl_tokens(amount))
            .ok_or_else(|| {
                Report::new(ErlError::storage(format!(
                    "Device {device} not found or not using V2"
                )))
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
        let adapter = ErlSharedMemoryAdapter::new(&handle);

        // Test quota setting and reading
        adapter
            .write_capacity(0, 100.0)
            .expect("should set capacity");
        adapter
            .write_refill_rate(0, 1.0)
            .expect("should set refill rate");
        let quota = adapter.read_quota(0).expect("should load quota");
        assert_eq!(quota.capacity, 100.0);
        assert_eq!(quota.refill_rate, 1.0);

        adapter.write_refill_rate(0, 5.0).expect("set new rate");
        let quota = adapter.read_quota(0).expect("reload quota");
        assert_eq!(quota.refill_rate, 5.0);

        // Test token state setting and reading
        adapter
            .write_token_state(0, TokenState::new(50.0, 1234567890.0))
            .expect("should save token state");
        let state = adapter
            .read_token_state(0)
            .expect("should load token state");
        assert_eq!(state.tokens, 50.0);
        assert_eq!(state.last_update, 1234567890.0);
    }

    #[test]
    fn test_erl_adapter_device_not_found() {
        let handle = SharedMemoryHandle::mock(
            "/tmp/test_erl_notfound",
            vec![(0, "test-device".to_string())],
        );
        let adapter = ErlSharedMemoryAdapter::new(&handle);

        // Test access to non-existent device
        let result = adapter.read_quota(999);
        assert!(result.is_err());
    }
}
