use std::ffi::c_int;
use std::ffi::c_uint;
use std::ffi::c_void;
use std::time::Duration;

use cudarc::driver::sys::cuDeviceGetCount;
use cudarc::driver::sys::CUresult;
use tf_macro::hook_fn;
use utils::hooks::HookManager;
use utils::replace_symbol;

use super::CUfunction;
use super::CUstream;
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
    with_device!(|device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        limiter.rate_limiter(
            device,
            grid_dim_x * grid_dim_y * grid_dim_z,
            block_dim_x * block_dim_y * block_dim_z,
        );
    });

    FN_CU_LAUNCH_KERNEL_PTSZ(
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
    with_device!(|device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        limiter.rate_limiter(
            device,
            grid_dim_x * grid_dim_y * grid_dim_z,
            block_dim_x * block_dim_y * block_dim_z,
        );
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
    with_device!(|device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");

        // Get block dimensions
        match limiter.get_block_dimensions(device) {
            Ok((block_x, block_y, block_z)) => {
                // Use the block dimensions to limit the rate
                limiter.rate_limiter(device, 1, block_x * block_y * block_z);
            }
            Err(_) => {
                tracing::warn!("Failed to get block dimensions, using default");
                limiter.rate_limiter(device, 1, 1);
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
    with_device!(|device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        limiter.rate_limiter(
            device,
            grid_dim_x * grid_dim_y * grid_dim_z,
            block_dim_x * block_dim_y * block_dim_z,
        );
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
    with_device!(|device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        limiter.rate_limiter(
            device,
            grid_dim_x * grid_dim_y * grid_dim_z,
            block_dim_x * block_dim_y * block_dim_z,
        );
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
    with_device!(|device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");

        // Get block dimensions
        match limiter.get_block_dimensions(device) {
            Ok((block_x, block_y, block_z)) => {
                // Use the block dimensions to limit the rate
                limiter.rate_limiter(
                    device,
                    (grid_width * grid_height) as u32,
                    block_x * block_y * block_z,
                );
            }
            Err(_) => {
                tracing::warn!("Failed to get block dimensions, using default");
                limiter.rate_limiter(device, (grid_width * grid_height) as u32, 1);
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
    with_device!(|device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");

        // Get block dimensions
        match limiter.get_block_dimensions(device) {
            Ok((block_x, block_y, block_z)) => {
                // Use the block dimensions to limit the rate
                limiter.rate_limiter(
                    device,
                    (grid_width * grid_height) as u32,
                    block_x * block_y * block_z,
                );
            }
            Err(_) => {
                tracing::warn!("Failed to get block dimensions, using default");
                limiter.rate_limiter(device, (grid_width * grid_height) as u32, 1);
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
    with_device!(|device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");

        // Set block dimensions
        if let Err(err) = limiter.set_block_dimensions(device, x as u32, y as u32, z as u32) {
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

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_init_detour(flag: c_uint) -> CUresult {
    // Call original cuInit first
    let result = FN_CU_INIT(flag);

    // If initialization failed, return the error
    if result != CUresult::CUDA_SUCCESS {
        return result;
    }

    // Get the number of CUDA devices
    let mut device_count: c_int = 0;
    let count_result = cuDeviceGetCount(&mut device_count as *mut c_int);

    if count_result != CUresult::CUDA_SUCCESS || device_count <= 0 {
        tracing::warn!("Failed to get device count or no devices found");
        return result;
    }

    // Get the global limiter instance
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");

    // Start a watcher thread for each device
    for device_idx in 0..device_count as u32 {
        let device_id = device_idx; // Create a copy for the closure
        std::thread::Builder::new()
            .name(format!("utilization-watcher-{}", device_id))
            .spawn(move || {
                if let Err(err) = limiter.run_watcher(device_id, Duration::from_millis(120)) {
                    tracing::error!("Watcher for device {} failed: {:?}", device_id, err);
                }
            })
            .unwrap_or_else(|_| {
                panic!("spawn utilization-watcher thread for device {}", device_id)
            });
    }

    result
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

    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuInit",
        cu_init_detour,
        FnCu_init,
        FN_CU_INIT
    );
}
