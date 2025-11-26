use std::cell::Cell;
use std::collections::HashSet;
use std::env;
use std::ffi::c_char;
use std::ffi::c_void;
use std::ffi::CStr;
use std::ffi::OsStr;
use std::fs;
use std::path::PathBuf;
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
use trap::{Trap, TrapAction, TrapError, TrapFrame};
use utils::hooks::HookManager;
use utils::hooks::NativePointer;
use utils::logging;
use utils::replace_symbol;

use crate::auto_freeze::AutoFreezeManager;

mod auto_freeze;
mod checkpoint;
mod command_handler;
mod config;
mod culib;
mod detour;
mod limiter;

static GLOBAL_LIMITER: OnceLock<Limiter> = OnceLock::new();
static GLOBAL_AUTO_FREEZE_MANAGER: OnceLock<Arc<AutoFreezeManager>> = OnceLock::new();
static GLOBAL_NGPU_LIBRARY: OnceLock<libloading::Library> = OnceLock::new();
static HOOKS_INITIALIZED: (AtomicBool, AtomicBool) =
    (AtomicBool::new(false), AtomicBool::new(false));

#[ctor]
unsafe fn entry_point() {
    logging::init();

    let (enable_nvml_hooks, enable_cuda_hooks) = are_hooks_enabled();
    tracing::info!(
        "enable_nvml_hooks: {enable_nvml_hooks}, enable_cuda_hooks: {enable_cuda_hooks}"
    );

    // Store the enabled state
    HOOKS_INITIALIZED
        .0
        .store(!enable_nvml_hooks, Ordering::Release);
    HOOKS_INITIALIZED
        .1
        .store(!enable_cuda_hooks, Ordering::Release);

    init_hooks();
}

fn are_hooks_enabled() -> (bool, bool) {
    let enable_nvml_hooks = if let Ok(enable_nvml_hooks) = env::var("ENABLE_NVML_HOOKS") {
        enable_nvml_hooks != "false"
    } else {
        true
    };
    let enable_cuda_hooks = if let Ok(enable_cuda_hooks) = env::var("ENABLE_CUDA_HOOKS") {
        enable_cuda_hooks != "false"
    } else {
        true
    };
    (enable_nvml_hooks, enable_cuda_hooks)
}

pub(crate) fn mock_shm_path() -> Option<PathBuf> {
    env::var("TF_SHM_FILE")
        .map(PathBuf::from)
        .map(|mut p| {
            p.pop();
            p
        })
        .ok()
}

fn init_limiter() {
    static NGPU_INITIALIZED: Once = Once::new();
    NGPU_INITIALIZED.call_once(|| {
        let nvml =
            match Nvml::builder()
                .lib_path(&env::var_os("TF_NVML_LIB_PATH").unwrap_or(
                    OsStr::new("/lib/x86_64-linux-gnu/libnvidia-ml.so.1").to_os_string(),
                ))
                .init()
            {
                Ok(nvml) => nvml,
                Err(e) => {
                    panic!("failed to initialize NVML: {e}");
                }
            };

        let config = if mock_shm_path().is_none() {
            let (hypervisor_ip, hypervisor_port) = match config::get_hypervisor_config() {
                Some((ip, port)) => (ip, port),
                None => {
                    tracing::info!(
                        "HYPERVISOR_IP or HYPERVISOR_PORT not set, skip command handler"
                    );
                    return;
                }
            };

            // Get device indices from environment variable
            let config = match config::get_worker_config(&hypervisor_ip, &hypervisor_port) {
                Ok(config) => config,
                Err(err) => {
                    tracing::error!("failed to get device configs: {err}");
                    return;
                }
            };
            config
        } else {
            // In mock/test mode, derive UUIDs from either CUDA_VISIBLE_DEVICES or list all devices
            let uuids = if let Ok(visible_devices) = env::var("TF_VISIBLE_DEVICES") {
                visible_devices
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect::<Vec<_>>()
            } else {
                panic!("TF_VISIBLE_DEVICES not set");
            };

            config::PodConfig {
                gpu_uuids: uuids,
                compute_shard: false,
                auto_freeze: None,
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
                env::set_var("CUDA_VISIBLE_DEVICES", &visible_devices);
                env::set_var("NVIDIA_VISIBLE_DEVICES", &visible_devices);
            }
        }

        let limiter = Limiter::new(nvml, config.gpu_uuids, config.compute_shard)
            .expect("failed to initialize Limiter");
        GLOBAL_LIMITER.set(limiter).expect("set GLOBAL_LIMITER");

        if let Some(auto_freeze_config) = config.auto_freeze {
            if auto_freeze_config.enable {
                const DEFAULT_IDLE_TIMEOUT: core::time::Duration =
                    core::time::Duration::from_secs(5 * 60);

                let idle_timeout = auto_freeze_config
                    .freeze_to_mem_ttl
                    .as_deref()
                    .and_then(|ttl| config::parse_duration(ttl).ok())
                    .unwrap_or_else(|| {
                        tracing::warn!(
                            "Using default freeze timeout: {} seconds",
                            DEFAULT_IDLE_TIMEOUT.as_secs()
                        );
                        DEFAULT_IDLE_TIMEOUT
                    });

                tracing::info!(
                    timeout_secs = idle_timeout.as_secs(),
                    "Initializing auto-freeze with idle timeout"
                );

                let manager = AutoFreezeManager::new(idle_timeout);
                if GLOBAL_AUTO_FREEZE_MANAGER.set(manager).is_err() {
                    tracing::warn!("Global auto-freeze manager already initialized");
                }
            }
        }
    });
}

fn try_install_cuda_hooks() {
    if HOOKS_INITIALIZED.1.load(Ordering::Acquire) {
        return;
    }

    let mut hook_manager = HookManager::default();
    hook_manager.collect_module_names();

    if !hook_manager
        .module_names
        .iter()
        .any(|m| m.starts_with("libcuda."))
    {
        return;
    }

    tracing::debug!("Installing CUDA hooks...");

    let install_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        detour::gpu::enable_hooks(&mut hook_manager);
        detour::mem::enable_hooks(&mut hook_manager);
    }));

    match install_result {
        Ok(_) => {
            HOOKS_INITIALIZED.1.store(true, Ordering::Release);
            tracing::debug!("CUDA hooks installed successfully");
        }
        Err(e) => {
            tracing::error!("CUDA hooks installation panicked: {:?}", e);
        }
    }
}

fn try_install_nvml_hooks() {
    if HOOKS_INITIALIZED.0.load(Ordering::Acquire) {
        return;
    }

    let mut hook_manager = HookManager::default();
    hook_manager.collect_module_names();

    if !hook_manager
        .module_names
        .iter()
        .any(|m| m.starts_with("libnvidia-ml."))
    {
        return;
    }

    tracing::debug!("Installing NVML hooks...");

    let install_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        detour::nvml::enable_hooks(&mut hook_manager, is_mapping_device_idx());
    }));

    match install_result {
        Ok(_) => {
            HOOKS_INITIALIZED.0.store(true, Ordering::Release);
            tracing::debug!("NVML hooks installed successfully");
        }
        Err(e) => {
            tracing::error!("NVML hooks installation panicked: {:?}", e);
        }
    }
}

fn init_hooks() {
    if cfg!(test) {
        tracing::debug!("Test mode detected, skipping hook initialization");
        return;
    }

    unsafe {
        // Load CUDA library to ensure it's loaded before hooks are installed
        let _ = culib::culib();
    }
    init_limiter();

    let is_compute_shard = GLOBAL_LIMITER
        .get()
        .expect("GLOBAL_LIMITER")
        .is_compute_shard();
    if is_compute_shard {
        tracing::debug!("Compute shard detected, skipping hook initialization");
        return;
    }

    // Try to install hooks immediately if libraries are already loaded
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
        try_install_cuda_hooks();
    }

    if has_libnvml {
        try_install_nvml_hooks();
    }

    // Install dlsym hook to catch dynamic library loading
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
    tracing::debug!("Hook initialization completed");
}

thread_local! {
    static IN_DLSYM_DETOUR: Cell<bool> = const { Cell::new(false) };
}

#[hook_fn]
unsafe extern "C" fn dlsym_detour(handle: *const c_void, symbol: *const c_char) -> *const c_void {
    if symbol.is_null() {
        return FN_DLSYM(handle, symbol);
    }

    let Ok(symbol_str) = CStr::from_ptr(symbol).to_str() else {
        return FN_DLSYM(handle, symbol);
    };
    let may_be_cuda = symbol_str.starts_with("cu");
    let may_be_nvml = symbol_str.starts_with("nvml");

    if !may_be_cuda && !may_be_nvml {
        return FN_DLSYM(handle, symbol);
    }

    // Prevent recursion
    if IN_DLSYM_DETOUR.with(|flag| flag.get()) {
        return FN_DLSYM(handle, symbol);
    }

    IN_DLSYM_DETOUR.with(|flag| flag.set(true));

    // Ensure we reset the flag when exiting
    struct ResetGuard;
    impl Drop for ResetGuard {
        fn drop(&mut self) {
            IN_DLSYM_DETOUR.with(|flag| flag.set(false));
        }
    }
    let _guard = ResetGuard;

    // Try to install hooks if not already done
    if may_be_cuda && !HOOKS_INITIALIZED.1.load(Ordering::Acquire) {
        try_install_cuda_hooks();
    }

    if may_be_nvml && !HOOKS_INITIALIZED.0.load(Ordering::Acquire) {
        try_install_nvml_hooks();
    }

    let sym_ptr = FN_DLSYM(handle, symbol);

    if may_be_cuda {
        // Skip checkpoint APIs to avoid recursive calls in auto-freeze manager
        if symbol_str.starts_with("cuCheckpoint") {
            return sym_ptr;
        }

        let Some(api) = checkpoint::checkpoint_api() else {
            return sym_ptr;
        };

        if !api.is_supported() {
            return sym_ptr;
        }

        let maybe_auto_freeze_manager = GLOBAL_AUTO_FREEZE_MANAGER.get();
        let auto_freeze_manager = match maybe_auto_freeze_manager {
            Some(auto_freeze_manager) => auto_freeze_manager,
            None => {
                return sym_ptr;
            }
        };
        let native_pointer = NativePointer(sym_ptr as *mut c_void);
        if !auto_freeze_manager.contains_native_pointer(&native_pointer) {
            if let Err(e) = auto_freeze_manager.attach_to_pointer(native_pointer) {
                tracing::error!(
                    error = %e,
                    pointer = ?native_pointer,
                    "Failed to attach auto-freeze listener"
                );
            }
        }
    }
    sym_ptr
}

enum TrapImpl {
    Dummy(DummyTrap),
    Http(Arc<BlockingHttpTrap>),
}

impl Trap for TrapImpl {
    fn enter_trap_and_wait(&self, frame: TrapFrame) -> Result<TrapAction, TrapError> {
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

pub fn global_trap() -> impl Trap {
    static GLOBAL_TRAP: OnceLock<Mutex<TrapImpl>> = OnceLock::new();

    let trap = GLOBAL_TRAP.get_or_init(|| {
        if let Some((hypervisor_ip, hypervisor_port)) = config::get_hypervisor_config() {
            let server_url = format!("http://{hypervisor_ip}:{hypervisor_port}");
            let config = HttpTrapConfig {
                server_url: server_url.into(),
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

fn is_mapping_device_idx() -> bool {
    match fs::read_to_string("/proc/self/cmdline") {
        Ok(cmdline) => cmdline.contains("nvidia-smi"),
        Err(_) => false,
    }
}
