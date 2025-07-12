use std::os::raw::c_uint;

pub(crate) mod gpu;
pub(crate) mod mem;
pub(crate) mod nvml;

// 添加缺少的类型定义
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct CUdeviceptrV1(pub c_uint);

// 定义 CudaArrayDescriptor 别名
pub(crate) use cudarc::driver::sys::CUDA_ARRAY_DESCRIPTOR as CudaArrayDescriptor;

/// macro: get current device and execute expression
#[macro_export]
macro_rules! with_device {
    ($expr:expr) => {{
        let mut device: i32 = 0;
        unsafe {
            // Call the CUDA API to get the current device
            let result = ::cudarc::driver::sys::cuCtxGetDevice(&mut device as *mut i32);

            if result != ::cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                panic!("Failed to get current CUDA device: error code {:?}", result);
            }
        }
        $expr(device as u32)
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
