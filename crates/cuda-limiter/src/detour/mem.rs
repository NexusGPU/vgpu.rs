use std::ffi::c_uint;
use std::ffi::c_ulonglong;

use cudarc::driver::sys::CUresult;
use tf_macro::hook_fn;
use trap::Trap;
use trap::TrapFrame;
use utils::hooks::HookManager;
use utils::replace_symbol;

use super::CUarray;
use super::CUdevice;
use super::CUdeviceptr;
use super::CUdeviceptrV1;
use super::CUmipmappedArray;
use super::CuarrayFormatEnum;
use super::CudaArray3dDescriptor;
use super::CudaArrayDescriptor;
use crate::detour::round_up;
use crate::global_trap;
use crate::limiter::Error;
use crate::with_device;
use crate::GLOBAL_LIMITER;

/// macro: check memory allocation and execute allocation
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
        let result = with_device!(|device| {
            let used = limiter.get_used_gpu_memory(device)?;
            let mem_limit = limiter.get_mem_limit(device)?;
            Ok::<(u64, u64), Error>((used, mem_limit))
        });

        match result {
            Ok((used, mem_limit)) if used.saturating_add($request_size) > mem_limit => {
                tracing::warn!(
                    "Allocation denied by limiter ({}): used ({}) + request ({}) > limit ({})",
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
                    Err(_) => {
                        // Wait failed or interrupted
                        tracing::warn!(
                            "OOM trap wait failed or interrupted for request size {}.",
                            request_size
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
    p_allocate_array: *const CudaArrayDescriptor,
) -> CUresult {
    let request_size = allocate_array_request_size(p_allocate_array);
    check_and_alloc!(request_size, "cuArrayCreate", || {
        FN_CU_ARRAY_CREATE(p_handle, p_allocate_array)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_array_3d_create_v2_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CudaArray3dDescriptor,
) -> CUresult {
    let request_size = allocate_array_3d_request_size(p_allocate_array);
    check_and_alloc!(request_size, "cuArray3DCreate_v2", || {
        FN_CU_ARRAY_3D_CREATE_V2(p_handle, p_allocate_array)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_array_3d_create_detour(
    p_handle: *mut CUarray,
    p_allocate_array: *const CudaArray3dDescriptor,
) -> CUresult {
    let request_size = allocate_array_3d_request_size(p_allocate_array);
    check_and_alloc!(request_size, "cuArray3DCreate", || {
        FN_CU_ARRAY_3D_CREATE(p_handle, p_allocate_array)
    })
}

#[hook_fn]
pub(crate) unsafe fn cu_mipmapped_array_create_detour(
    p_handle: *mut CUmipmappedArray,
    p_mipmapped_array_desc: *const CudaArray3dDescriptor,
    num_mipmap_levels: c_uint,
) -> CUresult {
    let desc_ref = &*p_mipmapped_array_desc;
    let base_size = get_array_base_size(desc_ref.format as _);
    let request_size =
        base_size * desc_ref.num_channels * desc_ref.height * desc_ref.width * desc_ref.depth;

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
pub(crate) unsafe fn cu_device_total_mem_v2_detour(bytes: *mut u64, device: CUdevice) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    limiter
        .get_mem_limit_cu(device)
        .map(|limit| {
            *bytes = limit;
            CUresult::CUDA_SUCCESS
        })
        .unwrap_or(CUresult::CUDA_ERROR_UNKNOWN)
}

#[hook_fn]
pub(crate) unsafe fn cu_device_total_mem_detour(bytes: *mut u64, device: CUdevice) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    limiter
        .get_mem_limit_cu(device)
        .map(|limit| {
            *bytes = limit;
            CUresult::CUDA_SUCCESS
        })
        .unwrap_or(CUresult::CUDA_ERROR_UNKNOWN)
}

#[hook_fn]
pub(crate) unsafe fn cu_mem_get_info_v2_detour(free: *mut u64, total: *mut u64) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    let result = with_device!(|device| {
        let mem_limit = limiter.get_mem_limit(device)?;
        let used = limiter.get_used_gpu_memory(device)?;
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
        let mem_limit = limiter.get_mem_limit(device)?;
        let used = limiter.get_used_gpu_memory(device)?;
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
