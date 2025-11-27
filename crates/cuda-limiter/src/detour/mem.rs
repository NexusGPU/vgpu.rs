use std::ffi::c_uint;

use cudarc::driver::sys::CUarray;
use cudarc::driver::sys::CUarray_format;
use cudarc::driver::sys::CUdevice;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::sys::CUmemGenericAllocationHandle;
use cudarc::driver::sys::CUmemoryPool;
use cudarc::driver::sys::CUmipmappedArray;
use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;
use cudarc::driver::sys::CUDA_ARRAY3D_DESCRIPTOR;
use cudarc::driver::sys::CUDA_ARRAY_DESCRIPTOR;
use tf_macro::hook_fn;
use utils::hooks::HookManager;
use utils::replace_symbol;

use crate::detour::round_up;
use crate::limiter::Error;
use crate::with_device;
use crate::Limiter;
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
        let device_result = with_device!(|limiter: &crate::limiter::Limiter, device_idx: usize| {
            (limiter.get_pod_memory_usage(device_idx), device_idx)
        });
        match device_result {
            Ok((result, device_idx)) => {
                match result {
                    Ok((used, mem_limit)) if used.saturating_add($request_size) > mem_limit => {
                        tracing::warn!(
                            "Allocation denied by limiter ({}): used ({}) + request ({}) > limit ({}) device_idx: {}",
                            $alloc_name,
                            used,
                            $request_size,
                            mem_limit,
                            device_idx
                        );
                        CUresult::CUDA_ERROR_OUT_OF_MEMORY
                    }
                    Ok(_) => cuda_alloc_with_retry($request_size, || $alloc_fn()),
                    Err(e @ Error::DeviceNotHealthy { .. }) => {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs();
                        tracing::warn!(now = now, "{e}");
                        $alloc_fn()
                    }
                    Err(e) => {
                        tracing::error!("Failed to get pod memory usage: {e}");
                        CUresult::CUDA_ERROR_UNKNOWN
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Device context error: {e}, falling back to native call");
                $alloc_fn()
            }
        }
    }};
}

// Helper function for allocation with retry logic
unsafe fn cuda_alloc_with_retry<F>(request_size: u64, alloc_fn: F) -> CUresult
where
    F: Fn() -> CUresult,
{
    // loop {
    let result = alloc_fn();
    match result {
        CUresult::CUDA_SUCCESS => {
            // Assuming limiter state is tracked elsewhere or doesn't need update here
            result
        }
        CUresult::CUDA_ERROR_OUT_OF_MEMORY => {
            tracing::info!(
                "cuda memory allocation pending, request size: {}",
                request_size
            );
            result
            // Temporarily block trap to avoid recursion
            // let trap = global_trap();
            // // OOM: enter trap and wait
            // match trap.enter_trap_and_wait(TrapFrame::OutOfMemory {
            //     requested_bytes: request_size,
            // }) {
            //     Ok(_) => {
            //         // Wait succeeded, loop to retry allocation
            //         tracing::debug!(
            //             "OOM trap wait succeeded for request size {}, retrying allocation.",
            //             request_size
            //         );
            //         continue;
            //     }
            //     Err(e) => {
            //         // Wait failed or interrupted
            //         tracing::warn!(
            //             "OOM trap wait failed or interrupted for request size {}, err: {}.",
            //             request_size,
            //             e
            //         );
            //         return CUresult::CUDA_ERROR_OUT_OF_MEMORY;
            //     }
            // }
        }
        _ => {
            // Other CUDA error
            result
        }
    }
    // }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_v2_detour(dptr: *mut CUdeviceptr, bytesize: u64) -> CUresult {
    let request_size = bytesize;
    check_and_alloc!(request_size, "cuMemAlloc_v2", || {
        FN_CU_MEM_ALLOC_V2(dptr, bytesize)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_detour(dptr: *mut CUdeviceptr, bytesize: u64) -> CUresult {
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
    dptr: *mut CUdeviceptr,
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
    p_allocate_array: *const CUDA_ARRAY_DESCRIPTOR,
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
    handle: *mut CUmemGenericAllocationHandle,
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
pub(crate) unsafe fn cu_mem_alloc_async_detour(
    dptr: *mut CUdeviceptr,
    bytesize: usize,
    h_stream: CUstream,
) -> CUresult {
    let request_size = bytesize;
    check_and_alloc!(request_size as u64, "cuMemAllocAsync", || {
        FN_CU_MEM_ALLOC_ASYNC(dptr, bytesize, h_stream)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_alloc_from_pool_async_detour(
    dptr: *mut CUdeviceptr,
    bytesize: usize,
    pool: CUmemoryPool,
    h_stream: CUstream,
) -> CUresult {
    let request_size = bytesize;
    check_and_alloc!(request_size as u64, "cuMemAllocFromPoolAsync", || {
        FN_CU_MEM_ALLOC_FROM_POOL_ASYNC(dptr, bytesize, pool, h_stream)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_device_total_mem_v2_detour(bytes: *mut u64, device: CUdevice) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    match limiter.get_pod_memory_usage_cu(device) {
        Ok((_, limit)) => {
            *bytes = limit;
            CUresult::CUDA_SUCCESS
        }
        Err(e) => {
            tracing::error!("Failed to get pod memory usage: {e}");
            CUresult::CUDA_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_device_total_mem_detour(bytes: *mut u64, device: CUdevice) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    match limiter.get_pod_memory_usage_cu(device) {
        Ok((_, limit)) => {
            *bytes = limit;
            CUresult::CUDA_SUCCESS
        }
        Err(_) => CUresult::CUDA_ERROR_UNKNOWN,
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_get_info_v2_detour(free: *mut u64, total: *mut u64) -> CUresult {
    let result = {
        with_device!(|limiter: &Limiter, device_idx: usize| {
            match limiter.get_pod_memory_usage(device_idx) {
                Ok((used, mem_limit)) => {
                    *total = mem_limit;
                    *free = mem_limit - used;
                    CUresult::CUDA_SUCCESS
                }
                Err(e @ Error::DeviceNotHealthy { .. }) => {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    tracing::warn!(now = now, "{e}");
                    CUresult::CUDA_ERROR_UNKNOWN
                }
                Err(e) => {
                    tracing::error!("Failed to get pod memory usage: {e}");
                    CUresult::CUDA_ERROR_UNKNOWN
                }
            }
        })
    };

    match result {
        Ok(cuda_result) => cuda_result,
        Err(e) => {
            tracing::warn!("Device context error: {e}, falling back to native call");
            FN_CU_MEM_GET_INFO_V2(free, total)
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_get_info_detour(free: *mut u64, total: *mut u64) -> CUresult {
    let result = {
        with_device!(|limiter: &Limiter, device_idx: usize| {
            match limiter.get_pod_memory_usage(device_idx) {
                Ok((used, mem_limit)) => {
                    *total = mem_limit;
                    *free = mem_limit - used;
                    CUresult::CUDA_SUCCESS
                }
                Err(e @ Error::DeviceNotHealthy { .. }) => {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    tracing::warn!(now = now, "{e}");
                    CUresult::CUDA_ERROR_UNKNOWN
                }
                Err(e) => {
                    tracing::error!("Failed to get pod memory usage: {e}");
                    CUresult::CUDA_ERROR_UNKNOWN
                }
            }
        })
    };

    match result {
        Ok(cuda_result) => cuda_result,
        Err(e) => {
            tracing::warn!("Device context error: {e}, falling back to native call");
            FN_CU_MEM_GET_INFO(free, total)
        }
    }
}

/// Enables hooks for CUDA memory management functions.
pub(crate) unsafe fn enable_hooks(hook_manager: &mut HookManager) -> Result<(), utils::Error> {
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemAlloc_v2",
        cu_mem_alloc_v2_detour,
        FnCu_mem_alloc_v2,
        FN_CU_MEM_ALLOC_V2
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemAlloc",
        cu_mem_alloc_detour,
        FnCu_mem_alloc,
        FN_CU_MEM_ALLOC
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemAllocManaged",
        cu_mem_alloc_managed_detour,
        FnCu_mem_alloc_managed,
        FN_CU_MEM_ALLOC_MANAGED
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemAllocPitch_v2",
        cu_mem_alloc_pitch_v2_detour,
        FnCu_mem_alloc_pitch_v2,
        FN_CU_MEM_ALLOC_PITCH_V2
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemAllocPitch",
        cu_mem_alloc_pitch_detour,
        FnCu_mem_alloc_pitch,
        FN_CU_MEM_ALLOC_PITCH
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuArrayCreate_v2",
        cu_array_create_v2_detour,
        FnCu_array_create_v2,
        FN_CU_ARRAY_CREATE_V2
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuArrayCreate",
        cu_array_create_detour,
        FnCu_array_create,
        FN_CU_ARRAY_CREATE
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuArray3DCreate_v2",
        cu_array_3d_create_v2_detour,
        FnCu_array_3d_create_v2,
        FN_CU_ARRAY_3D_CREATE_V2
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuArray3DCreate",
        cu_array_3d_create_detour,
        FnCu_array_3d_create,
        FN_CU_ARRAY_3D_CREATE
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMipmappedArrayCreate",
        cu_mipmapped_array_create_detour,
        FnCu_mipmapped_array_create,
        FN_CU_MIPMAPPED_ARRAY_CREATE
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuDeviceTotalMem_v2",
        cu_device_total_mem_v2_detour,
        FnCu_device_total_mem_v2,
        FN_CU_DEVICE_TOTAL_MEM_V2
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuDeviceTotalMem",
        cu_device_total_mem_detour,
        FnCu_device_total_mem,
        FN_CU_DEVICE_TOTAL_MEM
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemAllocAsync",
        cu_mem_alloc_async_detour,
        FnCu_mem_alloc_async,
        FN_CU_MEM_ALLOC_ASYNC
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemAllocFromPoolAsync",
        cu_mem_alloc_from_pool_async_detour,
        FnCu_mem_alloc_from_pool_async,
        FN_CU_MEM_ALLOC_FROM_POOL_ASYNC
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemGetInfo_v2",
        cu_mem_get_info_v2_detour,
        FnCu_mem_get_info_v2,
        FN_CU_MEM_GET_INFO_V2
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemGetInfo",
        cu_mem_get_info_detour,
        FnCu_mem_get_info,
        FN_CU_MEM_GET_INFO
    )?;
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuMemCreate",
        cu_mem_create_detour,
        FnCu_mem_create,
        FN_CU_MEM_CREATE
    )?;

    Ok(())
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
