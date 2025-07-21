use std::ffi::c_int;
use std::ffi::c_uint;
use std::ffi::c_void;

use cudarc::driver::sys::CUfunction;
use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;
use tf_macro::hook_fn;
use utils::hooks::HookManager;
use utils::replace_symbol;

use crate::with_device;
use crate::GLOBAL_LIMITER;

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_launch_kernel_ptsz_detour(
    f: CUfunction,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    h_stream: CUstream,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    // Use with_device macro directly
    with_device!(|_, device_uuid| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        if let Err(e) = limiter.rate_limiter(
            device_uuid,
            grid_dim_x * grid_dim_y * grid_dim_z,
            block_dim_x * block_dim_y * block_dim_z,
        ) {
            tracing::error!("Rate limiter failed: {}", e);
        }
    });

    FN_CU_LAUNCH_KERNEL(
        f,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        h_stream,
        kernel_params,
        extra,
    )
}

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_launch_kernel_detour(
    f: CUfunction,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    h_stream: CUstream,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    // Use with_device macro directly
    with_device!(|_, device_uuid| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        if let Err(e) = limiter.rate_limiter(
            device_uuid,
            grid_dim_x * grid_dim_y * grid_dim_z,
            block_dim_x * block_dim_y * block_dim_z,
        ) {
            tracing::error!("Rate limiter failed: {}", e);
        }
    });

    FN_CU_LAUNCH_KERNEL(
        f,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        h_stream,
        kernel_params,
        extra,
    )
}

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_launch_detour(f: CUfunction) -> CUresult {
    // Use with_device macro directly
    with_device!(|device_idx, device_uuid| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");

        // Get block dimensions
        match limiter.get_block_dimensions(device_idx) {
            Ok((block_x, block_y, block_z)) => {
                // Use the block dimensions to limit the rate
                if let Err(e) = limiter.rate_limiter(device_uuid, 1, block_x * block_y * block_z) {
                    tracing::error!("Rate limiter failed: {}", e);
                }
            }
            Err(_) => {
                tracing::warn!("Failed to get block dimensions, using default");
                if let Err(e) = limiter.rate_limiter(device_uuid, 1, 1) {
                    tracing::error!("Rate limiter failed: {}", e);
                }
            }
        }
    });

    FN_CU_LAUNCH(f)
}

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_launch_cooperative_kernel_ptsz_detour(
    f: CUfunction,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    h_stream: CUstream,
    kernel_params: *mut *mut c_void,
) -> CUresult {
    // Use with_device macro directly
    with_device!(|_, device_uuid| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        if let Err(e) = limiter.rate_limiter(
            device_uuid,
            grid_dim_x * grid_dim_y * grid_dim_z,
            block_dim_x * block_dim_y * block_dim_z,
        ) {
            tracing::error!("Rate limiter failed: {}", e);
        }
    });

    FN_CU_LAUNCH_COOPERATIVE_KERNEL_PTSZ(
        f,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        h_stream,
        kernel_params,
    )
}

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_launch_cooperative_kernel_detour(
    f: CUfunction,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    h_stream: CUstream,
    kernel_params: *mut *mut c_void,
) -> CUresult {
    // Use with_device macro directly
    with_device!(|_, device_uuid| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        if let Err(e) = limiter.rate_limiter(
            device_uuid,
            grid_dim_x * grid_dim_y * grid_dim_z,
            block_dim_x * block_dim_y * block_dim_z,
        ) {
            tracing::error!("Rate limiter failed: {}", e);
        }
    });

    FN_CU_LAUNCH_COOPERATIVE_KERNEL(
        f,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        h_stream,
        kernel_params,
    )
}

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_launch_grid_detour(
    f: CUfunction,
    grid_width: c_int,
    grid_height: c_int,
) -> CUresult {
    // Use with_device macro directly
    with_device!(|device_idx, device_uuid| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");

        // Get block dimensions
        match limiter.get_block_dimensions(device_idx) {
            Ok((block_x, block_y, block_z)) => {
                // Use the block dimensions to limit the rate
                if let Err(e) = limiter.rate_limiter(
                    device_uuid,
                    (grid_width * grid_height) as u32,
                    block_x * block_y * block_z,
                ) {
                    tracing::error!("Rate limiter failed: {}", e);
                }
            }
            Err(_) => {
                tracing::warn!("Failed to get block dimensions, using default");
                if let Err(e) =
                    limiter.rate_limiter(device_uuid, (grid_width * grid_height) as u32, 1)
                {
                    tracing::error!("Rate limiter failed: {}", e);
                }
            }
        }
    });

    FN_CU_LAUNCH_GRID(f, grid_width, grid_height)
}

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_launch_grid_async_detour(
    f: CUfunction,
    grid_width: c_int,
    grid_height: c_int,
    h_stream: CUstream,
) -> CUresult {
    // Use with_device macro directly
    with_device!(|device_idx, device_uuid| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");

        // Get block dimensions
        match limiter.get_block_dimensions(device_idx) {
            Ok((block_x, block_y, block_z)) => {
                // Use the block dimensions to limit the rate
                if let Err(e) = limiter.rate_limiter(
                    device_uuid,
                    (grid_width * grid_height) as u32,
                    block_x * block_y * block_z,
                ) {
                    tracing::error!("Rate limiter failed: {}", e);
                }
            }
            Err(_) => {
                tracing::warn!("Failed to get block dimensions, using default");
                if let Err(e) =
                    limiter.rate_limiter(device_uuid, (grid_width * grid_height) as u32, 1)
                {
                    tracing::error!("Rate limiter failed: {}", e);
                }
            }
        }
    });

    FN_CU_LAUNCH_GRID_ASYNC(f, grid_width, grid_height, h_stream)
}

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_func_set_block_shape_detour(
    hfunc: CUfunction,
    x: c_int,
    y: c_int,
    z: c_int,
) -> CUresult {
    // Use with_device macro directly
    with_device!(|device_idx, _| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");

        // Set block dimensions
        if let Err(err) = limiter.set_block_dimensions(device_idx, x as u32, y as u32, z as u32) {
            tracing::warn!(
                "Failed to set block dimensions: ({}, {}, {}): {:?}",
                x,
                y,
                z,
                err
            );
        }
    });

    FN_CU_FUNC_SET_BLOCK_SHAPE(hfunc, x, y, z)
}

pub(crate) unsafe fn enable_hooks(hook_manager: &mut HookManager) {
    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuLaunchKernel_ptsz",
        cu_launch_kernel_ptsz_detour,
        FnCu_launch_kernel_ptsz,
        FN_CU_LAUNCH_KERNEL_PTSZ
    );

    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuLaunchKernel",
        cu_launch_kernel_detour,
        FnCu_launch_kernel,
        FN_CU_LAUNCH_KERNEL
    );

    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuLaunch",
        cu_launch_detour,
        FnCu_launch,
        FN_CU_LAUNCH
    );

    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuLaunchCooperativeKernel_ptsz",
        cu_launch_cooperative_kernel_ptsz_detour,
        FnCu_launch_cooperative_kernel_ptsz,
        FN_CU_LAUNCH_COOPERATIVE_KERNEL_PTSZ
    );

    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuLaunchCooperativeKernel",
        cu_launch_cooperative_kernel_detour,
        FnCu_launch_cooperative_kernel,
        FN_CU_LAUNCH_COOPERATIVE_KERNEL
    );

    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuLaunchGrid",
        cu_launch_grid_detour,
        FnCu_launch_grid,
        FN_CU_LAUNCH_GRID
    );

    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuLaunchGridAsync",
        cu_launch_grid_async_detour,
        FnCu_launch_grid_async,
        FN_CU_LAUNCH_GRID_ASYNC
    );

    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuFuncSetBlockShape",
        cu_func_set_block_shape_detour,
        FnCu_func_set_block_shape,
        FN_CU_FUNC_SET_BLOCK_SHAPE
    );
}
