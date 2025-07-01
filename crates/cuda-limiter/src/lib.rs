#![feature(once_cell_try)]

use std::cell::RefCell;
use std::ffi::c_char;
use std::ffi::c_int;
use std::ffi::c_void;
use std::ffi::CStr;
use std::ffi::OsStr;
use std::sync::Mutex;
use std::sync::OnceLock;

use ctor::ctor;
use limiter::Limiter;
use nvml_wrapper::Nvml;
use tf_macro::hook_fn;
use trap::dummy::DummyTrap;
use trap::ipc::IpcTrap;
use utils::hooks::HookManager;
use utils::logging;
use utils::replace_symbol;

mod config;
mod detour;
mod limiter;

static GLOBAL_LIMITER: OnceLock<Limiter> = OnceLock::new();

thread_local! {
    static LIBCUDA_HOOKED: RefCell<bool> = const { RefCell::new(false) };
    static LIBNVML_HOOKED: RefCell<bool> = const { RefCell::new(false) };
}

#[no_mangle]
pub extern "C" fn set_limit(gpu: u32, up_limit: u32, mem_limit: u64) {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    if let Err(e) = limiter.set_uplimit(gpu, up_limit) {
        tracing::error!("Failed to set up_limit: {}", e);
    }
    if let Err(e) = limiter.set_mem_limit(gpu, mem_limit) {
        tracing::error!("Failed to set mem_limit: {}", e);
    }
}

enum TrapImpl {
    Dummy(DummyTrap),
    Ipc(IpcTrap),
}

impl trap::Trap for TrapImpl {
    fn enter_trap_and_wait(
        &self,
        frame: trap::TrapFrame,
    ) -> Result<trap::TrapAction, trap::TrapError> {
        match self {
            TrapImpl::Dummy(t) => t.enter_trap_and_wait(frame),
            TrapImpl::Ipc(t) => t.enter_trap_and_wait(frame),
        }
    }
}

impl Clone for TrapImpl {
    fn clone(&self) -> Self {
        match self {
            TrapImpl::Dummy(_) => TrapImpl::Dummy(DummyTrap {}),
            TrapImpl::Ipc(t) => TrapImpl::Ipc(t.clone()),
        }
    }
}

pub fn global_trap() -> impl trap::Trap {
    static GLOBAL_TRAP: OnceLock<Mutex<TrapImpl>> = OnceLock::new();

    let trap = GLOBAL_TRAP.get_or_init(|| {
        let ipc_server_path_name = std::env::var("TENSOR_FUSION_IPC_SERVER_PATH");
        match ipc_server_path_name {
            Ok(path) => match IpcTrap::connect(path) {
                Ok(trap) => TrapImpl::Ipc(trap).into(),
                Err(e) => {
                    panic!("failed to connect to ipc server, err: {e}");
                }
            },
            Err(_) => {
                tracing::warn!("using dummy trap");
                TrapImpl::Dummy(DummyTrap {}).into()
            }
        }
    });

    trap.lock().expect("poisoned").clone()
}

#[ctor]
unsafe fn entry_point() {
    logging::init();
    let pid = std::process::id();

    let nvml = match Nvml::init().and(
        Nvml::builder()
            .lib_path(OsStr::new("libnvidia-ml.so.1"))
            .init(),
    ) {
        Ok(nvml) => nvml,
        Err(e) => {
            tracing::error!("failed to initialize NVML: {}", e);
            return;
        }
    };

    // parse device limits
    let device_configs = config::parse_limits_and_create_device_configs(&nvml);

    if device_configs.is_empty() {
        tracing::info!("no device configs, skipping limiter");
        return;
    }

    let limiter = match Limiter::new(pid, nvml, &device_configs) {
        Ok(limiter) => limiter,
        Err(err) => {
            tracing::error!("failed to init limiter, err: {err}");
            return;
        }
    };
    GLOBAL_LIMITER.set(limiter).expect("set GLOBAL_LIMITER");

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
