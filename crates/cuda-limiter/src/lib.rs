use std::collections::HashSet;
use std::ffi::c_char;
use std::ffi::c_void;
use std::ffi::CStr;
use std::ffi::OsStr;
use std::os::raw;
use std::sync::atomic::{AtomicBool, Ordering};
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
static GLOBAL_NGPU_LIBRARY: OnceLock<NgpuLibrary> = OnceLock::new();

#[ctor]
unsafe fn entry_point() {
    logging::init();
    init_hooks();
}

#[derive(Debug)]
struct NgpuLibrary {
    pub handle: *mut raw::c_void,
}

impl NgpuLibrary {
    pub fn new(handle: *mut raw::c_void) -> Self {
        Self { handle }
    }
}

unsafe impl Send for NgpuLibrary {}
unsafe impl Sync for NgpuLibrary {}

fn init_ngpu_library() {
    static NGPU_INITIALIZED: Once = Once::new();
    NGPU_INITIALIZED.call_once(|| {
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
        let config = match config::get_device_configs(&hypervisor_ip, &hypervisor_port) {
            Ok(config) => config,
            Err(err) => {
                tracing::error!("failed to get device configs: {err}");
                return;
            }
        };

        if !config.gpu_uuids.is_empty() {
            let lower_case_uuids: HashSet<_> =
                config.gpu_uuids.iter().map(|u| u.to_lowercase()).collect();
            let device_count = nvml.device_count().expect("failed to get device count");

            let mut device_indices = Vec::new();
            for i in 0..device_count {
                let device = nvml
                    .device_by_index(i)
                    .expect("failed to get device by index");
                let uuid = device.uuid().expect("failed to get device uuid");
                if lower_case_uuids.contains(&uuid.to_lowercase()) {
                    device_indices.push(i.to_string());
                }
            }

            if !device_indices.is_empty() {
                let visible_devices = device_indices.join(",");
                tracing::info!(
                    "Setting CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES to {}",
                    &visible_devices
                );
                std::env::set_var("CUDA_VISIBLE_DEVICES", &visible_devices);
                std::env::set_var("NVIDIA_VISIBLE_DEVICES", &visible_devices);
            }
        }

        let limiter = match Limiter::new(config.host_pid, nvml, pod_identifier, &config.gpu_uuids) {
            Ok(limiter) => limiter,
            Err(err) => {
                tracing::error!("failed to init limiter, err: {err}");
                return;
            }
        };
        GLOBAL_LIMITER.set(limiter).expect("set GLOBAL_LIMITER");

        command_handler::start_background_handler(
            &hypervisor_ip,
            &hypervisor_port,
            config.host_pid,
        );

        // Load tensor-fusion/ngpu.so
        if let Ok(ngpu_path) = std::env::var("TENSOR_FUSION_NGPU_PATH") {
            tracing::debug!("loading ngpu.so from: {ngpu_path}");

            // Convert Rust String to CString for FFI
            let c_ngpu_path = match std::ffi::CString::new(ngpu_path.clone()) {
                Ok(cstr) => cstr,
                Err(e) => {
                    tracing::error!("failed to convert ngpu.so path to CString: {e}");
                    return;
                }
            };

            let handle =
                unsafe { libc::dlopen(c_ngpu_path.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
            if handle.is_null() {
                tracing::error!("failed to load ngpu.so: {ngpu_path}");
                return;
            }
            let ngpu_lib = NgpuLibrary::new(handle);
            GLOBAL_NGPU_LIBRARY
                .set(ngpu_lib)
                .expect("set GLOBAL_NGPU_LIBRARY");
            tracing::debug!("loaded ngpu.so");
        }
    });
}

fn init_cuda_hooks() {
    static CUDA_HOOKS_INITIALIZED: AtomicBool = AtomicBool::new(false);
    if CUDA_HOOKS_INITIALIZED.load(Ordering::Acquire) {
        return;
    }
    init_ngpu_library();

    let mut hook_manager = HookManager::default();
    hook_manager.collect_module_names();

    if hook_manager
        .module_names
        .iter()
        .any(|m| m.starts_with("libcuda."))
    {
        unsafe {
            detour::gpu::enable_hooks(&mut hook_manager);
            detour::mem::enable_hooks(&mut hook_manager);
        }
        CUDA_HOOKS_INITIALIZED.store(true, Ordering::Release);
        tracing::debug!("CUDA hooks initialized successfully");
    }
}

fn init_nvml_hooks() {
    static NVML_HOOKS_INITIALIZED: AtomicBool = AtomicBool::new(false);

    if NVML_HOOKS_INITIALIZED.load(Ordering::Acquire) {
        return;
    }
    init_ngpu_library();

    let mut hook_manager = HookManager::default();
    hook_manager.collect_module_names();

    if hook_manager
        .module_names
        .iter()
        .any(|m| m.starts_with("libnvidia-ml."))
    {
        unsafe {
            detour::nvml::enable_hooks(&mut hook_manager);
        }

        NVML_HOOKS_INITIALIZED.store(true, Ordering::Release);
        tracing::debug!("NVML hooks initialized successfully");
    }
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

    tracing::debug!("has_libcuda: {has_libcuda}, has_libnvml: {has_libnvml}");
    if has_libcuda {
        init_cuda_hooks();
    }

    if has_libnvml {
        init_nvml_hooks();
    }

    static DLSYM_HOOK_ONCE: Once = Once::new();
    DLSYM_HOOK_ONCE.call_once(|| unsafe {
        replace_symbol!(
            &mut hook_manager,
            None,
            "dlsym",
            dlsym_detour,
            FnDlsym,
            FN_DLSYM
        );
    });
}

#[hook_fn]
unsafe extern "C" fn dlsym_detour(handle: *const c_void, symbol: *const c_char) -> *const c_void {
    if !symbol.is_null() {
        let symbol_str = CStr::from_ptr(symbol).to_str().unwrap();
        tracing::trace!("dlsym: {symbol_str}");
        let may_be_cuda = symbol_str.starts_with("cu");
        let may_be_nvml = symbol_str.starts_with("nvml");
        if may_be_cuda || may_be_nvml {
            if may_be_cuda {
                init_cuda_hooks();
            }

            if may_be_nvml {
                init_nvml_hooks();
            }
        }
    }

    return FN_DLSYM(handle, symbol);
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

fn get_hypervisor_config() -> Option<(String, String)> {
    let hypervisor_ip = std::env::var("HYPERVISOR_IP").ok()?;
    let hypervisor_port = std::env::var("HYPERVISOR_PORT").ok()?;
    Some((hypervisor_ip, hypervisor_port))
}
