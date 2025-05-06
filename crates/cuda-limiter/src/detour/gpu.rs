use std::{
    ffi::{c_int, c_uint, c_void},
    sync::atomic::Ordering,
    time::Duration,
};

use super::CUresult;
use crate::GLOBAL_LIMITER;
use tf_macro::hook_fn;
use utils::{hooks::HookManager, replace_symbol};

use super::{CUfunction, CUstream};

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
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");

    limiter.rate_limiter(
        grid_dim_x * grid_dim_y * grid_dim_z,
        block_dim_x * block_dim_y * block_dim_z,
    );
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
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");

    limiter.rate_limiter(
        grid_dim_x * grid_dim_y * grid_dim_z,
        block_dim_x * block_dim_y * block_dim_z,
    );
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
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");

    limiter.rate_limiter(
        1,
        limiter.block_x.load(Ordering::Acquire)
            * limiter.block_y.load(Ordering::Acquire)
            * limiter.block_z.load(Ordering::Acquire),
    );
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
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");

    limiter.rate_limiter(
        grid_dim_x * grid_dim_y * grid_dim_z,
        block_dim_x * block_dim_y * block_dim_z,
    );
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
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");

    limiter.rate_limiter(
        grid_dim_x * grid_dim_y * grid_dim_z,
        block_dim_x * block_dim_y * block_dim_z,
    );
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
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");

    limiter.rate_limiter(
        (grid_width * grid_height) as u32,
        limiter.block_x.load(Ordering::Acquire)
            * limiter.block_y.load(Ordering::Acquire)
            * limiter.block_z.load(Ordering::Acquire),
    );

    FN_CU_LAUNCH_GRID(f, grid_width, grid_height)
}

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_launch_grid_async_detour(
    f: CUfunction,
    grid_width: c_int,
    grid_height: c_int,
    h_stream: CUstream,
) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");

    limiter.rate_limiter(
        (grid_width * grid_height) as u32,
        limiter.block_x.load(Ordering::Acquire)
            * limiter.block_y.load(Ordering::Acquire)
            * limiter.block_z.load(Ordering::Acquire),
    );
    FN_CU_LAUNCH_GRID_ASYNC(f, grid_width, grid_height, h_stream)
}

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_func_set_block_shape_detour(
    hfunc: CUfunction,
    x: c_int,
    y: c_int,
    z: c_int,
) -> CUresult {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");

    limiter.block_x.store(x as u32, Ordering::Release);
    limiter.block_y.store(y as u32, Ordering::Release);
    limiter.block_z.store(z as u32, Ordering::Release);
    FN_CU_FUNC_SET_BLOCK_SHAPE(hfunc, x, y, z)
}

#[hook_fn]
pub(crate) unsafe extern "C" fn cu_init_detour(flag: c_uint) -> CUresult {
    std::thread::Builder::new()
        .name("utilization-watcher".to_string())
        .spawn(|| {
            let limiter = GLOBAL_LIMITER.get().expect("get limiter");

            limiter.run_watcher(Duration::from_millis(120))
        })
        .expect("spawn utilization-watcher thread");
    FN_CU_INIT(flag)
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
