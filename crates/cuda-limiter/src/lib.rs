#![feature(once_cell_try)]

use std::cell::RefCell;
use std::ffi::c_char;
use std::ffi::c_int;
use std::ffi::c_void;
use std::ffi::CStr;
use std::ffi::OsStr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Duration;

use ctor::ctor;
use limiter::Limiter;
use nvml_wrapper::Nvml;
use tf_macro::hook_fn;
use trap::dummy::DummyTrap;
use trap::http::BlockingHttpTrap;
use trap::http::HttpTrapConfig;
use utils::hooks::HookManager;
use utils::logging;
use utils::replace_symbol;

mod command_handler;
mod config;
mod detour;
mod limiter;

static GLOBAL_LIMITER: OnceLock<Limiter> = OnceLock::new();
static GLOBAL_NGPU_LIBRARY: OnceLock<libloading::Library> = OnceLock::new();

thread_local! {
    static LIBCUDA_HOOKED: RefCell<bool> = const { RefCell::new(false) };
    static LIBNVML_HOOKED: RefCell<bool> = const { RefCell::new(false) };
}

enum TrapImpl {
    Dummy(DummyTrap),
    Http(Arc<BlockingHttpTrap>),
}

impl trap::Trap for TrapImpl {
    fn enter_trap_and_wait(
        &self,
        frame: trap::TrapFrame,
    ) -> Result<trap::TrapAction, trap::TrapError> {
        match self {
            TrapImpl::Dummy(t) => t.enter_trap_and_wait(frame),
            TrapImpl::Http(t) => t.enter_trap_and_wait(frame),
        }
    }
}

impl Clone for TrapImpl {
    fn clone(&self) -> Self {
        match self {
            TrapImpl::Dummy(_) => TrapImpl::Dummy(DummyTrap {}),
            TrapImpl::Http(t) => TrapImpl::Http(Arc::clone(t)),
        }
    }
}

pub fn global_trap() -> impl trap::Trap {
    static GLOBAL_TRAP: OnceLock<Mutex<TrapImpl>> = OnceLock::new();

    let trap = GLOBAL_TRAP.get_or_init(|| {
        // Try to construct HTTP trap from env vars using blocking constructor
        if let (Ok(ip), Ok(port)) = (
            std::env::var("HYPERVISOR_IP"),
            std::env::var("HYPERVISOR_PORT"),
        ) {
            let server_url = format!("http://{ip}:{port}");
            let config = HttpTrapConfig {
                server_url,
                ..Default::default()
            };

            match BlockingHttpTrap::new(config) {
                Ok(trap) => {
                    return TrapImpl::Http(Arc::new(trap)).into();
                }
                Err(e) => {
                    tracing::warn!("Failed to create HttpTrap: {e}, falling back to DummyTrap")
                }
            }
        }

        // Fallback to DummyTrap
        tracing::warn!("using dummy trap");
        TrapImpl::Dummy(DummyTrap {}).into()
    });

    trap.lock().expect("poisoned").clone()
}

#[ctor]
unsafe fn entry_point() {
    logging::init();

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
    let device_config_result = match config::get_device_configs(&nvml) {
        Ok(result) => result,
        Err(e) => {
            tracing::error!("failed to get device configs: {}", e);
            return;
        }
    };

    if device_config_result.device_configs.is_empty() {
        tracing::info!("no device configs, skipping limiter");
        return;
    }

    // load tensor-fusion/ngpu.so
    if let Ok(ngpu_path) = std::env::var("TENSOR_FUSION_NGPU_PATH") {
        tracing::debug!("loading ngpu.so from: {ngpu_path}");
        match libloading::Library::new(ngpu_path.as_str()) {
            Ok(lib) => {
                GLOBAL_NGPU_LIBRARY
                    .set(lib)
                    .expect("set GLOBAL_NGPU_LIBRARY");
                tracing::debug!("loaded ngpu.so");
            }
            Err(e) => {
                tracing::error!("failed to load ngpu.so: {e}, path: {ngpu_path}");
                return;
            }
        }
    }

    let limiter = match Limiter::new(
        device_config_result.host_pid,
        nvml,
        &device_config_result.device_configs,
    ) {
        Ok(limiter) => limiter,
        Err(err) => {
            tracing::error!("failed to init limiter, err: {err}");
            return;
        }
    };
    GLOBAL_LIMITER.set(limiter).expect("set GLOBAL_LIMITER");

    let mut hook_manager = HookManager::default();
    hook_manager.collect_module_names();
    tracing::debug!("test1");
    if hook_manager
        .module_names
        .iter()
        .any(|m| m.starts_with("libcuda."))
    {
        tracing::debug!("test1.1");
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
        tracing::debug!("test1.2");
        LIBNVML_HOOKED.with_borrow_mut(|hooked: &mut bool| {
            if !*hooked {
                detour::nvml::enable_hooks(&mut hook_manager);
                *hooked = true;
            }
        });
    }
    tracing::debug!("test2");

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
    tracing::debug!("test3");
    // start Hypervisor command handler background thread (requires HYPERVISOR_IP / PORT)
    if let (Ok(ip), Ok(port)) = (
        std::env::var("HYPERVISOR_IP"),
        std::env::var("HYPERVISOR_PORT"),
    ) {
        command_handler::start_background_handler(&ip, &port, device_config_result.host_pid);
    } else {
        tracing::info!("HYPERVISOR_IP or HYPERVISOR_PORT not set, skip command handler");
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
