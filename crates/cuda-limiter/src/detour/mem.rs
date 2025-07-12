use std::ffi::c_uint;
use std::ffi::c_ulonglong;

use cudarc::driver::sys::CUarray;
use cudarc::driver::sys::CUarray_format;
use cudarc::driver::sys::CUdevice;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::sys::CUmipmappedArray;
use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUDA_ARRAY3D_DESCRIPTOR;
use cudarc::driver::sys::CUDA_ARRAY_DESCRIPTOR;
use tf_macro::hook_fn;
use trap::Trap;
use trap::TrapFrame;
use utils::hooks::HookManager;
use utils::replace_symbol;

use crate::detour::round_up;
use crate::detour::{CUdeviceptrV1, CudaArrayDescriptor}; // 添加导入
use crate::global_trap;
use crate::limiter::Error;
use crate::with_device;
use crate::GLOBAL_LIMITER;

/// macro: check pod-level memory allocation and execute allocation
///
/// # Parameters
/// * `$request_size:expr` - Requested memory size in bytes
/// * `$alloc_name:expr` - Name of the allocation function, used for logging
/// * `$alloc_fn:expr` - Closure that performs the actual allocation
///
/// # Returns
/// Returns a CUDA result code
macro_rules! check_and_alloc {
    ($request_size:expr, $alloc_name:expr, $alloc_fn:expr) => {{
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        let result = with_device!(|device| { limiter.get_pod_memory_usage(device) });

        match result {
            Ok((used, mem_limit)) if used.saturating_add($request_size) > mem_limit => {
                tracing::warn!(
                    "Pod memory allocation denied ({}): pod_used ({}) + request ({}) > limit ({})",
                    $alloc_name,
                    used,
                    $request_size,
                    mem_limit
                );
                CUresult::CUDA_ERROR_OUT_OF_MEMORY
            }
            Ok(_) => cuda_alloc_with_retry($request_size, || $alloc_fn()),
            Err(_) => CUresult::CUDA_ERROR_UNKNOWN,
        }
    }};
}

// Helper function for allocation with retry logic
unsafe fn cuda_alloc_with_retry<F>(request_size: u64, alloc_fn: F) -> CUresult
where F: Fn() -> CUresult {
    loop {
        let result = alloc_fn();
        match result {
            CUresult::CUDA_SUCCESS => {
                // Assuming limiter state is tracked elsewhere or doesn't need update here
                return result;
            }
            CUresult::CUDA_ERROR_OUT_OF_MEMORY => {
                tracing::info!(
                    "cuda memory allocation pending, request size: {}",
                    request_size
                );
                let trap = global_trap();
                // OOM: enter trap and wait
                match trap.enter_trap_and_wait(TrapFrame::OutOfMemory {
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
                    Err(e) => {
                        // Wait failed or interrupted
                        tracing::warn!(
                            "OOM trap wait failed or interrupted for request size {}, err: {}.",
                            request_size,
                            e
                        );
                        return CUresult::CUDA_ERROR_OUT_OF_MEMORY;
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
    let request_size = bytesize;
    check_and_alloc!(request_size, "cuMemAlloc_v2", || {
        FN_CU_MEM_ALLOC_V2(dptr, bytesize)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_detour(dptr: *mut CUdeviceptrV1, bytesize: u64) -> CUresult {
    let request_size = bytesize;
    check_and_alloc!(request_size, "cuMemAlloc", || {
        FN_CU_MEM_ALLOC(dptr, bytesize)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_managed_detour(
    dptr: *mut CUdeviceptr,
    bytesize: u64,
    flags: c_uint,
) -> CUresult {
    let request_size = bytesize;
    check_and_alloc!(request_size, "cuMemAllocManaged", || {
        FN_CU_MEM_ALLOC_MANAGED(dptr, bytesize, flags)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_pitch_v2_detour(
    dptr: *mut CUdeviceptr,
    p_pitch: *mut usize,
    width_in_bytes: usize,
    height: usize,
    element_size_bytes: usize,
) -> CUresult {
    let request_size = round_up(width_in_bytes * height, element_size_bytes) as u64;
    check_and_alloc!(request_size, "cuMemAllocPitch_v2", || {
        FN_CU_MEM_ALLOC_PITCH_V2(dptr, p_pitch, width_in_bytes, height, element_size_bytes)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_pitch_detour(
    dptr: *mut CUdeviceptrV1,
    p_pitch: *mut usize,
    width_in_bytes: usize,
    height: usize,
    element_size_bytes: usize,
) -> CUresult {
    let request_size = (width_in_bytes * height) as u64;
    check_and_alloc!(request_size, "cuMemAllocPitch", || {
        FN_CU_MEM_ALLOC_PITCH(dptr, p_pitch, width_in_bytes, height, element_size_bytes)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_array_create_v2_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CudaArrayDescriptor,
) -> CUresult {
    let request_size = allocate_array_request_size(p_allocate_array);
    check_and_alloc!(request_size, "cuArrayCreate_v2", || {
        FN_CU_ARRAY_CREATE_V2(p_handle, p_allocate_array)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_array_create_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CUDA_ARRAY_DESCRIPTOR,
) -> CUresult {
    let request_size = allocate_array_request_size(p_allocate_array);
    check_and_alloc!(request_size, "cuArrayCreate", || {
        FN_CU_ARRAY_CREATE(p_handle, p_allocate_array)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_array_3d_create_v2_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CUDA_ARRAY3D_DESCRIPTOR,
) -> CUresult {
    let request_size = allocate_array_3d_request_size(p_allocate_array);
    check_and_alloc!(request_size, "cuArray3DCreate_v2", || {
        FN_CU_ARRAY_3D_CREATE_V2(p_handle, p_allocate_array)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_array_3d_create_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CUDA_ARRAY3D_DESCRIPTOR,
) -> CUresult {
    let request_size = allocate_array_3d_request_size(p_allocate_array);
    check_and_alloc!(request_size, "cuArray3DCreate", || {
        FN_CU_ARRAY_3D_CREATE(p_handle, p_allocate_array)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_mipmapped_array_create_detour(
    p_handle: *mut CUmipmappedArray,
    p_mipmapped_array_desc: *const CUDA_ARRAY3D_DESCRIPTOR,
    num_mipmap_levels: c_uint,
) -> CUresult {
    let desc_ref = &*p_mipmapped_array_desc;
    let base_size = get_array_base_size(desc_ref.Format);
    let request_size = base_size
        * (desc_ref.NumChannels as u64)
        * (desc_ref.Height as u64)
        * (desc_ref.Width as u64)
        * (desc_ref.Depth as u64);

    check_and_alloc!(request_size, "cuMipmappedArrayCreate", || {
        FN_CU_MIPMAPPED_ARRAY_CREATE(p_handle, p_mipmapped_array_desc, num_mipmap_levels)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_create_detour(
    handle: *mut c_ulonglong,
    size: u64,
    prop: *const u64,
    flags: c_uint,
) -> CUresult {
    let request_size = size;
    check_and_alloc!(request_size, "cuMemCreate", || {
        FN_CU_MEM_CREATE(handle, size, prop, flags)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_device_total_mem_v2_detour(bytes: *mut u64, _device: CUdevice) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    match with_device!(|device_idx| { limiter.get_mem_limit(device_idx) }) {
        Ok(limit) => {
            *bytes = limit;
            CUresult::CUDA_SUCCESS
        }
        Err(_) => CUresult::CUDA_ERROR_UNKNOWN,
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_device_total_mem_detour(bytes: *mut u64, _device: CUdevice) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    match with_device!(|device_idx| { limiter.get_mem_limit(device_idx) }) {
        Ok(limit) => {
            *bytes = limit;
            CUresult::CUDA_SUCCESS
        }
        Err(_) => CUresult::CUDA_ERROR_UNKNOWN,
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_get_info_v2_detour(free: *mut u64, total: *mut u64) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    let result = with_device!(|device| {
        let (used, mem_limit) = limiter.get_pod_memory_usage(device)?;
        Ok::<(u64, u64), Error>((mem_limit, used))
    });

    match result {
        Ok((mem_limit, used)) => {
            *free = mem_limit.saturating_sub(used);
            *total = mem_limit;
            CUresult::CUDA_SUCCESS
        }
        Err(_) => CUresult::CUDA_ERROR_UNKNOWN,
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_get_info_detour(free: *mut u64, total: *mut u64) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    match with_device!(|device| {
        let (used, mem_limit) = limiter.get_pod_memory_usage(device)?;
        Ok::<(u64, u64), Error>((mem_limit, used))
    }) {
        Ok((mem_limit, used)) => {
            *free = mem_limit.saturating_sub(used);
            *total = mem_limit;
            CUresult::CUDA_SUCCESS
        }
        Err(_) => CUresult::CUDA_ERROR_UNKNOWN,
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
fn get_array_base_size(format: CUarray_format) -> u64 {
    match format {
        CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8 | CUarray_format::CU_AD_FORMAT_SIGNED_INT8 => 8,
        CUarray_format::CU_AD_FORMAT_UNSIGNED_INT16
        | CUarray_format::CU_AD_FORMAT_SIGNED_INT16
        | CUarray_format::CU_AD_FORMAT_HALF => 16,
        CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32
        | CUarray_format::CU_AD_FORMAT_SIGNED_INT32
        | CUarray_format::CU_AD_FORMAT_FLOAT => 32,
        _ => 32,
    }
}

#[inline]
fn allocate_array_request_size(p_allocate_array: *const CUDA_ARRAY_DESCRIPTOR) -> u64 {
    let p_allocate_array = unsafe { &*p_allocate_array };
    let base_size = get_array_base_size(p_allocate_array.Format);
    base_size
        * (p_allocate_array.NumChannels as u64)
        * (p_allocate_array.Height as u64)
        * (p_allocate_array.Width as u64)
}

#[inline]
fn allocate_array_3d_request_size(p_allocate_array: *const CUDA_ARRAY3D_DESCRIPTOR) -> u64 {
    let p_allocate_array = unsafe { &*p_allocate_array };
    let base_size = get_array_base_size(p_allocate_array.Format);
    base_size
        * (p_allocate_array.NumChannels as u64)
        * (p_allocate_array.Height as u64)
        * (p_allocate_array.Width as u64)
        * (p_allocate_array.Depth as u64)
}
