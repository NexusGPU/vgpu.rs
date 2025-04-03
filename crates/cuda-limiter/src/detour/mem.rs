use std::ffi::c_uint;
use std::thread::sleep;
use std::time::Duration;

use tf_macro::hook_fn;
use utils::{hooks::HookManager, replace_symbol};

use crate::{
    detour::{round_up, NVML_ERROR_UNKNOWN},
    GLOBAL_LIMITER,
};

use super::{
    CUarray, CUdevice, CUdeviceptr, CUdeviceptrV1, CUmipmappedArray, CUresult, CuarrayFormatEnum,
    CudaArray3dDescriptor, CudaArrayDescriptor,
};

const CUDA_SUCCESS: CUresult = 0;
const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
// Maximum number of retry attempts for CUDA functions
const MAX_RETRY_ATTEMPTS: u32 = 10;
// Delay between retry attempts in milliseconds
const RETRY_DELAY_MS: u64 = 300;

/// Helper function to retry CUDA operations with exponential backoff
///
/// # Arguments
/// * `f` - Function to retry
///
/// # Returns
/// * The result from the function or the last error after all retries
unsafe fn retry_cuda_op<F>(mut f: F) -> CUresult
where
    F: FnMut() -> CUresult,
{
    let mut result = f();

    // Only retry if the operation failed
    if result != CUDA_SUCCESS {
        for attempt in 1..=MAX_RETRY_ATTEMPTS {
            // Exponential backoff: delay increases with each retry
            let delay = RETRY_DELAY_MS * (1 << (attempt - 1));
            sleep(Duration::from_millis(delay));

            result = f();

            // If successful, break the retry loop
            if result == CUDA_SUCCESS {
                break;
            }
        }
    }

    result
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_v2_detour(dptr: *mut CUdeviceptr, bytesize: u64) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            let request_size = bytesize;
            if used + request_size > limiter.get_mem_limit() {
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                retry_cuda_op(|| FN_CU_MEM_ALLOC_V2(dptr, bytesize))
            }
        }
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            NVML_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_detour(dptr: *mut CUdeviceptrV1, bytesize: u64) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            let request_size = bytesize;
            if used + request_size > limiter.get_mem_limit() {
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                retry_cuda_op(|| FN_CU_MEM_ALLOC(dptr, bytesize))
            }
        }
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            NVML_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_managed_detour(
    dptr: *mut CUdeviceptr,
    bytesize: u64,
    flags: c_uint,
) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            let request_size = bytesize;
            if used + request_size > limiter.get_mem_limit() {
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                retry_cuda_op(|| FN_CU_MEM_ALLOC_MANAGED(dptr, bytesize, flags))
            }
        }
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            NVML_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_pitch_v2_detour(
    dptr: *mut CUdeviceptr,
    p_pitch: *mut usize,
    width_in_bytes: usize,
    height: usize,
    element_size_bytes: usize,
) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            let request_size = round_up(width_in_bytes * height, element_size_bytes) as u64;
            if used + request_size > limiter.get_mem_limit() {
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                retry_cuda_op(|| {
                    FN_CU_MEM_ALLOC_PITCH_V2(
                        dptr,
                        p_pitch,
                        width_in_bytes,
                        height,
                        element_size_bytes,
                    )
                })
            }
        }
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            NVML_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_pitch_detour(
    dptr: *mut CUdeviceptrV1,
    p_pitch: *mut usize,
    width_in_bytes: usize,
    height: usize,
    element_size_bytes: usize,
) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            let request_size = (height * width_in_bytes) as u64;
            if used + request_size > limiter.get_mem_limit() {
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                retry_cuda_op(|| {
                    FN_CU_MEM_ALLOC_PITCH(dptr, p_pitch, width_in_bytes, height, element_size_bytes)
                })
            }
        }
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            NVML_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_array_create_v2_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CudaArrayDescriptor,
) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            let request_size = allocate_array_request_size(p_allocate_array);
            if used + request_size > limiter.get_mem_limit() {
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                retry_cuda_op(|| FN_CU_ARRAY_CREATE_V2(p_handle, p_allocate_array))
            }
        }
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            NVML_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_array_create_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CudaArrayDescriptor,
) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            let request_size = allocate_array_request_size(p_allocate_array);
            if used + request_size > limiter.get_mem_limit() {
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                retry_cuda_op(|| FN_CU_ARRAY_CREATE(p_handle, p_allocate_array))
            }
        }
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            NVML_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_array_3d_create_v2_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CudaArray3dDescriptor,
) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            let request_size = allocate_array_3d_request_size(p_allocate_array);
            if used + request_size > limiter.get_mem_limit() {
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                retry_cuda_op(|| FN_CU_ARRAY_3D_CREATE_V2(p_handle, p_allocate_array))
            }
        }
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            NVML_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_array_3d_create_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CudaArray3dDescriptor,
) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            let request_size = allocate_array_3d_request_size(p_allocate_array);
            if used + request_size > limiter.get_mem_limit() {
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                retry_cuda_op(|| FN_CU_ARRAY_3D_CREATE(p_handle, p_allocate_array))
            }
        }
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            NVML_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mipmapped_array_create_detour(
    p_handle: *mut CUmipmappedArray,
    p_mipmapped_array_desc: *const CudaArray3dDescriptor,
    num_mipmap_levels: c_uint,
) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            let p_mipmapped_array_desc = &*p_mipmapped_array_desc;
            let base_size = get_array_base_size(p_mipmapped_array_desc.format as _);
            let request_size = base_size
                * p_mipmapped_array_desc.num_channels
                * p_mipmapped_array_desc.height
                * p_mipmapped_array_desc.width
                * p_mipmapped_array_desc.depth;

            if used + request_size > limiter.get_mem_limit() {
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                retry_cuda_op(|| {
                    FN_CU_MIPMAPPED_ARRAY_CREATE(
                        p_handle,
                        p_mipmapped_array_desc,
                        num_mipmap_levels,
                    )
                })
            }
        }
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            NVML_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_device_total_mem_v2_detour(bytes: *mut u64, _dev: CUdevice) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    *bytes = limiter.get_mem_limit();
    CUDA_SUCCESS
}

#[hook_fn]
pub(crate) unsafe fn cu_device_total_mem_detour(bytes: *mut u64, _dev: CUdevice) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    *bytes = limiter.get_mem_limit();
    CUDA_SUCCESS
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_get_info_v2_detour(free: *mut u64, total: *mut u64) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    let mem_limit = limiter.get_mem_limit();

    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            *free = mem_limit.saturating_sub(used);
            *total = mem_limit;
            CUDA_SUCCESS
        }
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            NVML_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_get_info_detour(free: *mut u64, total: *mut u64) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    let mem_limit = limiter.get_mem_limit();

    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            *free = mem_limit.saturating_sub(used);
            *total = mem_limit;
            CUDA_SUCCESS
        }
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            NVML_ERROR_UNKNOWN
        }
    }
}

/// Enables hooks for CUDA memory management functions.
pub(crate) unsafe fn enable_hooks(hook_manager: &mut HookManager) {
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemAlloc_v2",
        cu_mem_alloc_v2_detour,
        FnCu_mem_alloc_v2,
        FN_CU_MEM_ALLOC_V2
    );
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemAlloc",
        cu_mem_alloc_detour,
        FnCu_mem_alloc,
        FN_CU_MEM_ALLOC
    );
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemAllocManaged",
        cu_mem_alloc_managed_detour,
        FnCu_mem_alloc_managed,
        FN_CU_MEM_ALLOC_MANAGED
    );
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemAllocPitch_v2",
        cu_mem_alloc_pitch_v2_detour,
        FnCu_mem_alloc_pitch_v2,
        FN_CU_MEM_ALLOC_PITCH_V2
    );
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemAllocPitch",
        cu_mem_alloc_pitch_detour,
        FnCu_mem_alloc_pitch,
        FN_CU_MEM_ALLOC_PITCH
    );
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuArrayCreate_v2",
        cu_array_create_v2_detour,
        FnCu_array_create_v2,
        FN_CU_ARRAY_CREATE_V2
    );
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuArrayCreate",
        cu_array_create_detour,
        FnCu_array_create,
        FN_CU_ARRAY_CREATE
    );
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuArray3DCreate_v2",
        cu_array_3d_create_v2_detour,
        FnCu_array_3d_create_v2,
        FN_CU_ARRAY_3D_CREATE_V2
    );
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuArray3DCreate",
        cu_array_3d_create_detour,
        FnCu_array_3d_create,
        FN_CU_ARRAY_3D_CREATE
    );
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMipmappedArrayCreate",
        cu_mipmapped_array_create_detour,
        FnCu_mipmapped_array_create,
        FN_CU_MIPMAPPED_ARRAY_CREATE
    );
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuDeviceTotalMem_v2",
        cu_device_total_mem_v2_detour,
        FnCu_device_total_mem_v2,
        FN_CU_DEVICE_TOTAL_MEM_V2
    );
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuDeviceTotalMem",
        cu_device_total_mem_detour,
        FnCu_device_total_mem,
        FN_CU_DEVICE_TOTAL_MEM
    );
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemGetInfo_v2",
        cu_mem_get_info_v2_detour,
        FnCu_mem_get_info_v2,
        FN_CU_MEM_GET_INFO_V2
    );
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemGetInfo",
        cu_mem_get_info_detour,
        FnCu_mem_get_info,
        FN_CU_MEM_GET_INFO
    );
}

#[inline]
fn get_array_base_size(format: CuarrayFormatEnum) -> u64 {
    match format {
        CuarrayFormatEnum::CuAdFormatUnsignedInt8 | CuarrayFormatEnum::CuAdFormatSignedInt8 => 8,
        CuarrayFormatEnum::CuAdFormatUnsignedInt16
        | CuarrayFormatEnum::CuAdFormatSignedInt16
        | CuarrayFormatEnum::CuAdFormatHalf => 16,
        CuarrayFormatEnum::CuAdFormatUnsignedInt32
        | CuarrayFormatEnum::CuAdFormatSignedInt32
        | CuarrayFormatEnum::CuAdFormatFloat => 32,
        _ => 32,
    }
}

#[inline]
fn allocate_array_request_size(p_allocate_array: *const CudaArrayDescriptor) -> u64 {
    let p_allocate_array = unsafe { &*p_allocate_array };
    let base_size = get_array_base_size(p_allocate_array.format);
    base_size * p_allocate_array.num_channels * p_allocate_array.height * p_allocate_array.width
}

#[inline]
fn allocate_array_3d_request_size(p_allocate_array: *const CudaArray3dDescriptor) -> u64 {
    let p_allocate_array = unsafe { &*p_allocate_array };
    let base_size = get_array_base_size(p_allocate_array.format);
    base_size
        * p_allocate_array.num_channels
        * p_allocate_array.height
        * p_allocate_array.width
        * p_allocate_array.depth
}
