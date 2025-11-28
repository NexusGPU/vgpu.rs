pub(crate) mod gpu;
pub(crate) mod mem;
pub(crate) mod nvml;

/// macro: get current device UUID and execute expression
/// Returns Result<T, Error> where T is the return type of the expression
#[macro_export]
macro_rules! with_device {
    ($expr:expr) => {{
        let mut device: i32 = 0;
        let cuda_result = unsafe {
            // Call the CUDA API to get the current device
            ($crate::culib::culib().cuCtxGetDevice)(&mut device as *mut i32)
        };

        if cuda_result != ::cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            Err($crate::limiter::Error::Cuda(cuda_result))
        } else {
            match $crate::GLOBAL_LIMITER.get() {
                Some(limiter) => match limiter.device_raw_index_by_cu_device(device) {
                    Ok(device_index) => Ok($expr(limiter, device_index)),
                    Err(e) => Err(e),
                },
                None => {
                    $crate::report_limiter_not_initialized();
                    Err($crate::limiter::Error::LimiterNotInitialized)
                }
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
