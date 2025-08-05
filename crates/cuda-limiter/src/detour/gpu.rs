use std::ffi::c_int;
use std::ffi::c_uint;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use cudarc::driver::sys::CUdevice;
use cudarc::driver::sys::CUfunction;
use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUstream;
use tf_macro::hook_fn;
use utils::hooks::HookManager;
use utils::replace_symbol;

use crate::command_handler;
use crate::config;
use crate::culib;
use crate::limiter::Error;
use crate::with_device;
use crate::Limiter;
use crate::GLOBAL_LIMITER;
use crate::GLOBAL_NGPU_LIBRARY;

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
    with_device!(|limiter: &Limiter, device_idx: usize| {
        if let Err(e) = limiter.rate_limiter(
            device_idx,
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
    with_device!(|limiter: &Limiter, device_idx: usize| {
        if let Err(e) = limiter.rate_limiter(
            device_idx,
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
    with_device!(|limiter: &Limiter, device_idx: usize| {
        // Get block dimensions
        match limiter.get_block_dimensions(device_idx) {
            Ok((block_x, block_y, block_z)) => {
                // Use the block dimensions to limit the rate
                if let Err(e) = limiter.rate_limiter(device_idx, 1, block_x * block_y * block_z) {
                    tracing::error!("Rate limiter failed: {}", e);
                }
            }
            Err(_) => {
                tracing::warn!("Failed to get block dimensions, using default");
                if let Err(e) = limiter.rate_limiter(device_idx, 1, 1) {
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
    with_device!(|limiter: &Limiter, device_idx: usize| {
        if let Err(e) = limiter.rate_limiter(
            device_idx,
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
    with_device!(|limiter: &Limiter, device_idx: usize| {
        if let Err(e) = limiter.rate_limiter(
            device_idx,
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
    with_device!(|limiter: &Limiter, device_idx: usize| {
        // Get block dimensions
        match limiter.get_block_dimensions(device_idx) {
            Ok((block_x, block_y, block_z)) => {
                // Use the block dimensions to limit the rate
                if let Err(e) = limiter.rate_limiter(
                    device_idx,
                    (grid_width * grid_height) as u32,
                    block_x * block_y * block_z,
                ) {
                    tracing::error!("Rate limiter failed: {}", e);
                }
            }
            Err(_) => {
                tracing::warn!("Failed to get block dimensions, using default");
                if let Err(e) =
                    limiter.rate_limiter(device_idx, (grid_width * grid_height) as u32, 1)
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
    with_device!(|limiter: &Limiter, device_idx: usize| {
        // Get block dimensions
        match limiter.get_block_dimensions(device_idx) {
            Ok((block_x, block_y, block_z)) => {
                // Use the block dimensions to limit the rate
                if let Err(e) = limiter.rate_limiter(
                    device_idx,
                    (grid_width * grid_height) as u32,
                    block_x * block_y * block_z,
                ) {
                    tracing::error!("Rate limiter failed: {}", e);
                }
            }
            Err(_) => {
                tracing::warn!("Failed to get block dimensions, using default");
                if let Err(e) =
                    limiter.rate_limiter(device_idx, (grid_width * grid_height) as u32, 1)
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
    with_device!(|limiter: &Limiter, device_idx: usize| {
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

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_init_detour(flags: c_uint) -> CUresult {
    // Track worker initialization state - allow retry if init_worker fails
    static WORKER_INITIALIZED: AtomicBool = AtomicBool::new(false);
    static WORKER_INIT_MUTEX: Mutex<()> = Mutex::new(());

    // Check if already initialized successfully
    if !WORKER_INITIALIZED.load(Ordering::Acquire) {
        // Use mutex to ensure only one thread attempts initialization at a time
        if let Ok(_guard) = WORKER_INIT_MUTEX.lock() {
            // Double-check pattern to avoid race conditions
            if !WORKER_INITIALIZED.load(Ordering::Acquire) {
                // Try to initialize worker with hypervisor - using guard clauses for cleaner code
                let Some((hypervisor_ip, hypervisor_port)) = config::get_hypervisor_config() else {
                    tracing::debug!(
                        "Hypervisor config not available, skipping worker initialization"
                    );
                    return FN_CU_INIT(flags);
                };

                let Some(container_name) = config::get_container_name() else {
                    tracing::warn!("Container name not available, skipping worker initialization");
                    return FN_CU_INIT(flags);
                };

                let result = match config::init_worker(
                    &hypervisor_ip,
                    &hypervisor_port,
                    &container_name,
                ) {
                    Ok(result) => {
                        tracing::info!(
                            "Worker initialized successfully via cu_init_detour: host_pid={}, gpu_uuids={:?}",
                            result.host_pid,
                            result.gpu_uuids
                        );
                        result
                    }
                    Err(e) => {
                        tracing::error!("Failed to initialize worker via cu_init_detour: {}, will retry on next call", e);
                        return FN_CU_INIT(flags);
                    }
                };

                // Load tensor-fusion/ngpu.so
                if let Ok(ngpu_path) = std::env::var("TENSOR_FUSION_NGPU_PATH") {
                    tracing::debug!("loading ngpu.so from: {ngpu_path}");

                    match unsafe { libloading::Library::new(ngpu_path.as_str()) } {
                        Ok(lib) => {
                            GLOBAL_NGPU_LIBRARY
                                .set(lib)
                                .expect("set GLOBAL_NGPU_LIBRARY");
                            tracing::debug!("loaded ngpu.so");
                        }
                        Err(e) => {
                            tracing::error!("failed to load ngpu.so: {e}, path: {ngpu_path}");
                        }
                    }
                    command_handler::start_background_handler(
                        &hypervisor_ip,
                        &hypervisor_port,
                        result.host_pid,
                    );
                }

                // Mark as successfully initialized only after everything succeeds
                WORKER_INITIALIZED.store(true, Ordering::Release);
            }
        }
    }

    let result = FN_CU_INIT(flags);

    tracing::debug!("cu_init_detour flags: {:?}, result: {:?}", flags, result);

    result
}

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_device_get_detour(
    device: *mut CUdevice,
    ordinal: ::core::ffi::c_int,
) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    let result = FN_CU_DEVICE_GET(device, ordinal);

    if result == CUresult::CUDA_SUCCESS {
        tracing::debug!(
            "cuDeviceGet_detour ordinal: {:?}, device: {:?}",
            ordinal,
            device
        );

        if let Err(e) = limiter.insert_cu_device_if_not_exists(*device, || {
            let uuid = unsafe { culib::device_uuid(*device).map_err(|e| Error::Cuda(e))? };
            Ok(uuid)
        }) {
            tracing::error!("Failed to insert CUDA device: {}", e);
        }
    }

    result
}

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_device_get_count_detour(count: *mut c_int) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    *count = limiter.get_device_count() as c_int;
    CUresult::CUDA_SUCCESS
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

    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuDeviceGet",
        cu_device_get_detour,
        FnCu_device_get,
        FN_CU_DEVICE_GET
    );

    replace_symbol!(
        hook_manager,
        Some("libcuda."),
        "cuDeviceGetCount",
        cu_device_get_count_detour,
        FnCu_device_get_count,
        FN_CU_DEVICE_GET_COUNT
    );
}
