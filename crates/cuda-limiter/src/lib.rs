#![feature(once_cell_try)]

use std::ffi::c_char;
use std::ffi::c_int;
use std::ffi::c_void;
use std::ffi::CStr;
use std::ffi::OsStr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Once;
use std::sync::OnceLock;

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

static NGPU_INITIALIZED: Once = Once::new();
static CUDA_HOOKS_INITIALIZED: Once = Once::new();
static NVML_HOOKS_INITIALIZED: Once = Once::new();

fn init_ngpu_library() {
    NGPU_INITIALIZED.call_once(|| {
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
        }
    });
}

fn init_cuda_hooks() {
    CUDA_HOOKS_INITIALIZED.call_once(|| {
        init_ngpu_library();

        let mut hook_manager = HookManager::default();
        hook_manager.collect_module_names();

        unsafe {
            detour::gpu::enable_hooks(&mut hook_manager);
            detour::mem::enable_hooks(&mut hook_manager);
        }

        tracing::debug!("CUDA hooks initialized successfully");
    });
}

fn init_nvml_hooks() {
    NVML_HOOKS_INITIALIZED.call_once(|| {
        init_ngpu_library();

        let mut hook_manager = HookManager::default();
        hook_manager.collect_module_names();

        unsafe {
            detour::nvml::enable_hooks(&mut hook_manager);
        }

        tracing::debug!("NVML hooks initialized successfully");
    });
}

fn init_hooks() {
    let mut hook_manager = HookManager::default();
    hook_manager.collect_module_names();

    let has_libcuda = hook_manager
        .module_names
        .iter()
        .any(|m| m.starts_with("libcuda."));

    let has_libnvml = hook_manager
        .module_names
        .iter()
        .any(|m| m.starts_with("libnvidia-ml."));

    if has_libcuda {
        init_cuda_hooks();
    }

    if has_libnvml {
        init_nvml_hooks();
    }

    if !has_libcuda || !has_libnvml {
        unsafe {
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
}

#[ctor]
unsafe fn entry_point() {
    logging::init();

    // Get pod name from environment variable
    let pod_identifier = match limiter::get_pod_identifier() {
        Ok(name) => name,
        Err(_) => {
            tracing::error!("Failed to get pod name from environment, cuda-limiter disabled");
            return;
        }
    };

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

    let (hypervisor_ip, hypervisor_port) = match get_hypervisor_config() {
        Some((ip, port)) => (ip, port),
        None => {
            tracing::info!("HYPERVISOR_IP or HYPERVISOR_PORT not set, skip command handler");
            return;
        }
    };

    // Get device indices from environment variable
    let config = match config::get_device_configs(&nvml, &hypervisor_ip, &hypervisor_port) {
        Ok(config) => config,
        Err(err) => {
            tracing::error!("failed to get device configs: {err}");
            return;
        }
    };

    let limiter = match Limiter::new(
        std::process::id(),
        pod_identifier,
        config.device_configs.iter().map(|c| c.device_idx).collect(),
    ) {
        Ok(limiter) => limiter,
        Err(err) => {
            tracing::error!("failed to init limiter, err: {err}");
            return;
        }
    };
    GLOBAL_LIMITER.set(limiter).expect("set GLOBAL_LIMITER");

    init_hooks();

    tracing::debug!("CUDA limiter initialized successfully");
    // start Hypervisor command handler background thread (requires HYPERVISOR_IP / PORT)
    command_handler::start_background_handler(&hypervisor_ip, &hypervisor_port, std::process::id());
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
        if let Some((hypervisor_ip, hypervisor_port)) = get_hypervisor_config() {
            let server_url = format!("http://{hypervisor_ip}:{hypervisor_port}");
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

#[hook_fn]
pub(crate) unsafe extern "C" fn dlopen_detour(name: *const c_char, mode: c_int) -> *const c_void {
    let ret = FN_DLOPEN(name, mode);
    if !name.is_null() {
        let lib = CStr::from_ptr(name).to_str().unwrap();

        tracing::trace!("dlopen: {lib}, {ret:?}");

        if lib.contains("libcuda.") {
            init_cuda_hooks();
        }

        if lib.contains("libnvidia-ml.") {
            init_nvml_hooks();
        }
    }
    ret
}

fn get_hypervisor_config() -> Option<(String, String)> {
    let hypervisor_ip = std::env::var("HYPERVISOR_IP").ok()?;
    let hypervisor_port = std::env::var("HYPERVISOR_PORT").ok()?;
    Some((hypervisor_ip, hypervisor_port))
}
