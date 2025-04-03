use ctor::ctor;

use limiter::Limiter;
use std::{
    cell::RefCell,
    ffi::{c_char, c_int, c_void, CStr},
    sync::OnceLock,
};
use tf_macro::hook_fn;
use utils::{hooks::HookManager, logging, replace_symbol};

mod detour;
mod limiter;

static mut GLOABL_LIMITER: OnceLock<Limiter> = OnceLock::new();

thread_local! {
    static LIBCUDA_HOOKED: RefCell<bool> = const { RefCell::new(false) };
    static LIBNVML_HOOKED: RefCell<bool> = const { RefCell::new(false) };
}

#[no_mangle]
pub extern "C" fn set_limit(gpu: u32, mem: u64) {
    unsafe {
        let limiter = GLOABL_LIMITER.get().expect("get limiter");
        limiter.set_uplimit(gpu);
        limiter.set_mem_limit(mem);
    }
}

#[ctor]
unsafe fn entry_point() {
    logging::init();
    let pid = std::process::id();

    // Read up_limit and mem_limit from environment variables with defaults
    let up_limit = std::env::var("TENSOR_FUSION_CUDA_UP_LIMIT")
        .map(|v| v.parse().unwrap_or(0))
        .unwrap_or(0);

    let mem_limit = std::env::var("TENSOR_FUSION_CUDA_MEM_LIMIT")
        .map(|v| v.parse().unwrap_or(0))
        .unwrap_or(0);

    let limiter = match Limiter::init(pid, 0, up_limit, mem_limit) {
        Ok(limiter) => limiter,
        Err(err) => {
            tracing::error!("failed to init limiter, err: {err}");
            return;
        }
    };
    GLOABL_LIMITER.set(limiter).expect("set GLOABL_LIMITER");

    let mut hook_manager = HookManager::default();
    hook_manager.collect_module_names();

    if hook_manager
        .module_names
        .iter()
        .any(|m| m.starts_with("libcuda."))
    {
        LIBCUDA_HOOKED.with_borrow_mut(|hooked: &mut bool| {
            if !*hooked {
                detour::gpu::enable_hooks(&mut hook_manager);
                detour::mem::enable_hooks(&mut hook_manager);
                *hooked = true;
            }
        });
    }
    if hook_manager
        .module_names
        .iter()
        .any(|m| m.starts_with("libnvidia-ml."))
    {
        LIBNVML_HOOKED.with_borrow_mut(|hooked: &mut bool| {
            if !*hooked {
                detour::nvml::enable_hooks(&mut hook_manager);
                *hooked = true;
            }
        });
    }
    let cuda_hooked = LIBCUDA_HOOKED.with(|hooked| *hooked.borrow());
    let nvml_hooked = LIBNVML_HOOKED.with(|hooked| *hooked.borrow());

    if !cuda_hooked || !nvml_hooked {
        replace_symbol!(
            &mut hook_manager,
            None,
            "dlopen",
            dlopen_detour,
            FnDlopen,
            FN_DLOPEN
        );
    }
}

#[hook_fn]
pub(crate) unsafe extern "C" fn dlopen_detour(name: *const c_char, mode: c_int) -> *const c_void {
    let ret = FN_DLOPEN(name, mode);
    if !name.is_null() {
        let lib = CStr::from_ptr(name).to_str().unwrap();

        tracing::trace!("dlopen: {lib}, {ret:?}");

        if lib.contains("libcuda.") {
            LIBCUDA_HOOKED.with_borrow_mut(|hooked: &mut bool| {
                if !*hooked {
                    let mut hook_manager = HookManager::default();
                    hook_manager.collect_module_names();
                    detour::gpu::enable_hooks(&mut hook_manager);
                    detour::mem::enable_hooks(&mut hook_manager);
                    *hooked = true;
                }
            })
        }

        if lib.contains("libnvidia-ml.") {
            LIBNVML_HOOKED.with_borrow_mut(|hooked: &mut bool| {
                if !*hooked {
                    let mut hook_manager = HookManager::default();
                    hook_manager.collect_module_names();
                    detour::nvml::enable_hooks(&mut hook_manager);
                    *hooked = true;
                }
            })
        }
    }
    ret
}
