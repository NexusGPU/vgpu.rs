use std::ffi::{c_uint, c_ulonglong};

use tf_macro::hook_fn;
use trap::{Trap, TrapFrame};
use utils::{hooks::HookManager, replace_symbol};

use crate::{
    detour::{round_up, NVML_ERROR_UNKNOWN},
    limiter::Limiter,
    GLOBAL_LIMITER,
};

use super::{
    CUarray, CUdevice, CUdeviceptr, CUdeviceptrV1, CUmipmappedArray, CUresult, CuarrayFormatEnum,
    CudaArray3dDescriptor, CudaArrayDescriptor,
};
const CUDA_SUCCESS: CUresult = 0;
const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;

// Helper function for allocation with retry logic
unsafe fn cuda_alloc_with_retry<T: Trap, F>(
    limiter: &Limiter<T>,
    request_size: u64,
    alloc_fn: F,
) -> CUresult
where
    F: Fn() -> CUresult,
{
    loop {
        let result = alloc_fn();
        match result {
            CUDA_SUCCESS => {
                // Assuming limiter state is tracked elsewhere or doesn't need update here
                return result;
            }
            CUDA_ERROR_OUT_OF_MEMORY => {
                // OOM: enter trap and wait
                match limiter.trap.enter_trap_and_wait(TrapFrame::OutOfMemory {
                    requested_bytes: request_size,
                }) {
                    Ok(_) => {
                        // Wait succeeded, loop to retry allocation
                        tracing::debug!(
                            "OOM trap wait succeeded for request size {}, retrying allocation.",
                            request_size
                        );
                        continue;
                    }
                    Err(_) => {
                        // Wait failed or interrupted
                        tracing::warn!(
                            "OOM trap wait failed or interrupted for request size {}.",
                            request_size
                        );
                        return CUDA_ERROR_OUT_OF_MEMORY as CUresult;
                    }
                }
            }
            _ => {
                // Other CUDA error
                return result;
            }
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_v2_detour(dptr: *mut CUdeviceptr, bytesize: u64) -> CUresult {
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter
        .get()
        .expect("Limiter not initialized during cuMemAlloc_v2");
    let request_size = bytesize;

    // Check against the memory limit *before* attempting allocation
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            if used.saturating_add(request_size) > limiter.get_mem_limit() {
                tracing::warn!(
                    "Allocation denied by limiter: used ({}) + request ({}) > limit ({})",
                    used,
                    request_size,
                    limiter.get_mem_limit()
                );
                CUDA_ERROR_OUT_OF_MEMORY as CUresult // Return OOM if limit exceeded
            } else {
                // Proceed with allocation attempt only if within limit
                cuda_alloc_with_retry(limiter, request_size, || FN_CU_MEM_ALLOC_V2(dptr, bytesize))
            }
        }
        Err(e) => {
            // Handle error fetching used memory (e.g., log and maybe deny allocation)
            tracing::error!(
                "Failed to get used GPU memory: {:?}. Denying allocation.",
                e
            );
            // Decide on behavior: return OOM or attempt allocation anyway?
            // Returning OOM for safety.
            CUDA_ERROR_OUT_OF_MEMORY as CUresult
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_detour(dptr: *mut CUdeviceptrV1, bytesize: u64) -> CUresult {
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter
        .get()
        .expect("Limiter not initialized during cuMemAlloc");
    let request_size = bytesize;
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            if used.saturating_add(request_size) > limiter.get_mem_limit() {
                tracing::warn!(
                    "Allocation denied by limiter (cuMemAlloc): used ({}) + request ({}) > limit ({})",
                    used,
                    request_size,
                    limiter.get_mem_limit()
                );
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                cuda_alloc_with_retry(limiter, request_size, || FN_CU_MEM_ALLOC(dptr, bytesize))
            }
        }
        Err(e) => {
            tracing::error!(
                "Failed to get used GPU memory (cuMemAlloc): {:?}. Denying allocation.",
                e
            );
            CUDA_ERROR_OUT_OF_MEMORY as CUresult
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_managed_detour(
    dptr: *mut CUdeviceptr,
    bytesize: u64,
    flags: c_uint,
) -> CUresult {
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter
        .get()
        .expect("Limiter not initialized during cuMemAllocManaged");
    let request_size = bytesize;
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            if used.saturating_add(request_size) > limiter.get_mem_limit() {
                tracing::warn!(
                    "Allocation denied by limiter (cuMemAllocManaged): used ({}) + request ({}) > limit ({})",
                    used,
                    request_size,
                    limiter.get_mem_limit()
                );
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                cuda_alloc_with_retry(limiter, request_size, || {
                    FN_CU_MEM_ALLOC_MANAGED(dptr, bytesize, flags)
                })
            }
        }
        Err(e) => {
            tracing::error!(
                "Failed to get used GPU memory (cuMemAllocManaged): {:?}. Denying allocation.",
                e
            );
            CUDA_ERROR_OUT_OF_MEMORY as CUresult
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
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter
        .get()
        .expect("Limiter not initialized during cuMemAllocPitch_v2");
    let request_size = round_up(width_in_bytes * height, element_size_bytes) as u64;
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            if used.saturating_add(request_size) > limiter.get_mem_limit() {
                tracing::warn!(
                    "Allocation denied by limiter (cuMemAllocPitch_v2): used ({}) + request ({}) > limit ({})",
                    used,
                    request_size,
                    limiter.get_mem_limit()
                );
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                cuda_alloc_with_retry(limiter, request_size, || {
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
            tracing::error!(
                "Failed to get used GPU memory (cuMemAllocPitch_v2): {:?}. Denying allocation.",
                e
            );
            CUDA_ERROR_OUT_OF_MEMORY as CUresult
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
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter
        .get()
        .expect("Limiter not initialized during cuMemAllocPitch");
    let request_size = (width_in_bytes * height) as u64;
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            if used.saturating_add(request_size) > limiter.get_mem_limit() {
                tracing::warn!(
                    "Allocation denied by limiter (cuMemAllocPitch): used ({}) + request ({}) > limit ({})",
                    used,
                    request_size,
                    limiter.get_mem_limit()
                );
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                cuda_alloc_with_retry(limiter, request_size, || {
                    FN_CU_MEM_ALLOC_PITCH(dptr, p_pitch, width_in_bytes, height, element_size_bytes)
                })
            }
        }
        Err(e) => {
            tracing::error!(
                "Failed to get used GPU memory (cuMemAllocPitch): {:?}. Denying allocation.",
                e
            );
            CUDA_ERROR_OUT_OF_MEMORY as CUresult
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_array_create_v2_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CudaArrayDescriptor,
) -> CUresult {
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter
        .get()
        .expect("Limiter not initialized during cuArrayCreate_v2");
    let request_size = allocate_array_request_size(p_allocate_array);
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            if used.saturating_add(request_size) > limiter.get_mem_limit() {
                tracing::warn!(
                    "Allocation denied by limiter (cuArrayCreate_v2): used ({}) + request ({}) > limit ({})",
                    used,
                    request_size,
                    limiter.get_mem_limit()
                );
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                cuda_alloc_with_retry(limiter, request_size, || {
                    FN_CU_ARRAY_CREATE_V2(p_handle, p_allocate_array)
                })
            }
        }
        Err(e) => {
            tracing::error!(
                "Failed to get used GPU memory (cuArrayCreate_v2): {:?}. Denying allocation.",
                e
            );
            CUDA_ERROR_OUT_OF_MEMORY as CUresult
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_array_create_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CudaArrayDescriptor,
) -> CUresult {
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter
        .get()
        .expect("Limiter not initialized during cuArrayCreate");
    let request_size = allocate_array_request_size(p_allocate_array);
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            if used.saturating_add(request_size) > limiter.get_mem_limit() {
                tracing::warn!(
                    "Allocation denied by limiter (cuArrayCreate): used ({}) + request ({}) > limit ({})",
                    used,
                    request_size,
                    limiter.get_mem_limit()
                );
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                cuda_alloc_with_retry(limiter, request_size, || {
                    FN_CU_ARRAY_CREATE(p_handle, p_allocate_array)
                })
            }
        }
        Err(e) => {
            tracing::error!(
                "Failed to get used GPU memory (cuArrayCreate): {:?}. Denying allocation.",
                e
            );
            CUDA_ERROR_OUT_OF_MEMORY as CUresult
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_array_3d_create_v2_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CudaArray3dDescriptor,
) -> CUresult {
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter
        .get()
        .expect("Limiter not initialized during cuArray3DCreate_v2");
    let request_size = allocate_array_3d_request_size(p_allocate_array);
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            if used.saturating_add(request_size) > limiter.get_mem_limit() {
                tracing::warn!(
                    "Allocation denied by limiter (cuArray3DCreate_v2): used ({}) + request ({}) > limit ({})",
                    used,
                    request_size,
                    limiter.get_mem_limit()
                );
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                cuda_alloc_with_retry(limiter, request_size, || {
                    FN_CU_ARRAY_3D_CREATE_V2(p_handle, p_allocate_array)
                })
            }
        }
        Err(e) => {
            tracing::error!(
                "Failed to get used GPU memory (cuArray3DCreate_v2): {:?}. Denying allocation.",
                e
            );
            CUDA_ERROR_OUT_OF_MEMORY as CUresult
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_array_3d_create_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CudaArray3dDescriptor,
) -> CUresult {
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter
        .get()
        .expect("Limiter not initialized during cuArray3DCreate");
    let request_size = allocate_array_3d_request_size(p_allocate_array);
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            if used.saturating_add(request_size) > limiter.get_mem_limit() {
                tracing::warn!(
                    "Allocation denied by limiter (cuArray3DCreate): used ({}) + request ({}) > limit ({})",
                    used,
                    request_size,
                    limiter.get_mem_limit()
                );
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                cuda_alloc_with_retry(limiter, request_size, || {
                    FN_CU_ARRAY_3D_CREATE(p_handle, p_allocate_array)
                })
            }
        }
        Err(e) => {
            tracing::error!(
                "Failed to get used GPU memory (cuArray3DCreate): {:?}. Denying allocation.",
                e
            );
            CUDA_ERROR_OUT_OF_MEMORY as CUresult
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mipmapped_array_create_detour(
    p_handle: *mut CUmipmappedArray,
    p_mipmapped_array_desc: *const CudaArray3dDescriptor,
    num_mipmap_levels: c_uint,
) -> CUresult {
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter
        .get()
        .expect("Limiter not initialized during cuMipmappedArrayCreate");
    let desc_ref = &*p_mipmapped_array_desc;
    let base_size = get_array_base_size(desc_ref.format as _);
    let request_size =
        base_size * desc_ref.num_channels * desc_ref.height * desc_ref.width * desc_ref.depth;
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            if used.saturating_add(request_size) > limiter.get_mem_limit() {
                tracing::warn!(
                    "Allocation denied by limiter (cuMipmappedArrayCreate): used ({}) + request ({}) > limit ({})",
                    used,
                    request_size,
                    limiter.get_mem_limit()
                );
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                cuda_alloc_with_retry(limiter, request_size, || {
                    FN_CU_MIPMAPPED_ARRAY_CREATE(
                        p_handle,
                        p_mipmapped_array_desc,
                        num_mipmap_levels,
                    )
                })
            }
        }
        Err(e) => {
            tracing::error!(
                "Failed to get used GPU memory (cuMipmappedArrayCreate): {:?}. Denying allocation.",
                e
            );
            CUDA_ERROR_OUT_OF_MEMORY as CUresult
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_create_detour(
    handle: *mut c_ulonglong,
    size: u64,
    prop: *const u64,
    flags: c_uint,
) -> CUresult {
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter
        .get()
        .expect("Limiter not initialized during cuMemCreate");
    let request_size = size;
    match limiter.get_used_gpu_memory() {
        Ok(used) => {
            if used.saturating_add(request_size) > limiter.get_mem_limit() {
                tracing::warn!(
                    "Allocation denied by limiter (cuMemCreate): used ({}) + request ({}) > limit ({})",
                    used,
                    request_size,
                    limiter.get_mem_limit()
                );
                CUDA_ERROR_OUT_OF_MEMORY as CUresult
            } else {
                cuda_alloc_with_retry(limiter, request_size, || {
                    FN_CU_MEM_CREATE(handle, size, prop, flags)
                })
            }
        }
        Err(e) => {
            tracing::error!(
                "Failed to get used GPU memory (cuMemCreate): {:?}. Denying allocation.",
                e
            );
            CUDA_ERROR_OUT_OF_MEMORY as CUresult
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_device_total_mem_v2_detour(bytes: *mut u64, _dev: CUdevice) -> CUresult {
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter.get().expect("get limiter");

    *bytes = limiter.get_mem_limit();
    CUDA_SUCCESS
}

#[hook_fn]
pub(crate) unsafe fn cu_device_total_mem_detour(bytes: *mut u64, _dev: CUdevice) -> CUresult {
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter.get().expect("get limiter");

    *bytes = limiter.get_mem_limit();
    CUDA_SUCCESS
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_get_info_v2_detour(free: *mut u64, total: *mut u64) -> CUresult {
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter.get().expect("get limiter");

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
    let limiter = GLOBAL_LIMITER;
    let limiter = limiter.get().expect("get limiter");

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
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemCreate",
        cu_mem_create_detour,
        FnCu_mem_create,
        FN_CU_MEM_CREATE
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
