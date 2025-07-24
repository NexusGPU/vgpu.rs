use std::collections::HashMap;
use std::sync::OnceLock;

pub(crate) mod gpu;
pub(crate) mod mem;
pub(crate) mod nvml;

pub static GLOBAL_DEVICE_UUIDS: OnceLock<HashMap<i32, String>> = OnceLock::new();

/// macro: get current device UUID and execute expression
#[macro_export]
macro_rules! with_device {
    ($expr:expr) => {{
        let mut device: i32 = 0;
        unsafe {
            // Call the CUDA API to get the current device
            let result = ($crate::culib::culib().cuCtxGetDevice)(&mut device as *mut i32);
            if result != ::cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                panic!("Failed to get current CUDA device: error code {:?}", result);
            }

            let device_uuid = $crate::detour::GLOBAL_DEVICE_UUIDS
                .get()
                .unwrap()
                .get(&device);
            if device_uuid.is_none() {
                panic!("Device UUID not found for device {}", device);
            }

            $expr(device, device_uuid.unwrap())
        }
    }};
}

#[inline]
pub(crate) fn round_up(n: usize, base: usize) -> usize {
    if !n.is_multiple_of(base) {
        n + base - (n % base)
    } else {
        n
    }
}
