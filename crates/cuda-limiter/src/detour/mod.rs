pub(crate) mod gpu;
pub(crate) mod mem;
pub(crate) mod nvml;

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

            let limiter: &crate::limiter::Limiter = $crate::GLOBAL_LIMITER
                .get()
                .expect("Limiter not initialized");

            let device_index = limiter.device_index_by_cu_device(device);

            if let Err(e) = device_index {
                panic!("Failed to get device index: {}", e);
            } else {
                $expr(limiter, device_index.unwrap())
            }
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
