use std::cell::Cell;
use std::collections::HashSet;
use std::ffi::c_char;
use std::ffi::c_void;
use std::ffi::CStr;
use std::ffi::OsStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::Once;
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
// Synchronization pairs for CUDA and NVML symbol detection and hook initialization
static CUDA_SYMBOL_SYNC: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());
static NVML_SYMBOL_SYNC: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());
static CUDA_DLSYM_SYNC: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());
static NVML_DLSYM_SYNC: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());
static CUDA_HOOKS_INITIALIZED: AtomicBool = AtomicBool::new(false);
static NVML_HOOKS_INITIALIZED: AtomicBool = AtomicBool::new(false);

#[ctor]
unsafe fn entry_point() {
    logging::init();

    let (enable_nvml_hooks, enable_cuda_hooks) = are_hooks_enabled();
    tracing::info!(
        "enable_nvml_hooks: {enable_nvml_hooks}, enable_cuda_hooks: {enable_cuda_hooks}"
    );
    init_hooks(enable_nvml_hooks, enable_cuda_hooks);
}

fn are_hooks_enabled() -> (bool, bool) {
    let enable_nvml_hooks = if let Ok(enable_nvml_hooks) = std::env::var("ENABLE_NVML_HOOKS") {
        enable_nvml_hooks != "false"
    } else {
        true
    };
    let enable_cuda_hooks = if let Ok(enable_cuda_hooks) = std::env::var("ENABLE_CUDA_HOOKS") {
        enable_cuda_hooks != "false"
    } else {
        true
    };
    (enable_nvml_hooks, enable_cuda_hooks)
}

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

fn init_cuda_hooks(enable_cuda_hooks: bool) -> bool {
    IN_HOOK_INITIALIZATION.with(|flag| flag.set(true));

    // Ensure we reset the flag when exiting, even in case of panic
    struct ResetGuard;
    impl Drop for ResetGuard {
        fn drop(&mut self) {
            IN_HOOK_INITIALIZATION.with(|flag| flag.set(false));
        }
    }
    let _guard = ResetGuard;

    init_ngpu_library();

    if !enable_cuda_hooks {
        return true;
    }

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
        tracing::debug!("CUDA_HOOKS_INITIALIZED flag set to true");
        return true;
    }
    false
}

fn init_nvml_hooks(enable_nvml_hooks: bool) -> bool {
    IN_HOOK_INITIALIZATION.with(|flag| flag.set(true));

    // Ensure we reset the flag when exiting, even in case of panic
    struct ResetGuard;
    impl Drop for ResetGuard {
        fn drop(&mut self) {
            IN_HOOK_INITIALIZATION.with(|flag| flag.set(false));
        }
    }
    let _guard = ResetGuard;

    init_ngpu_library();
    if !enable_nvml_hooks {
        return true;
    }

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
        tracing::debug!("NVML_HOOKS_INITIALIZED flag set to true");
        return true;
    }
    false
}

fn init_hooks(enable_nvml_hooks: bool, enable_cuda_hooks: bool) {
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
        let cuda_result = init_cuda_hooks(enable_cuda_hooks);
        tracing::debug!("Entry point CUDA hooks initialization result: {cuda_result}");
    }

    if has_libnvml {
        let nvml_result = init_nvml_hooks(enable_nvml_hooks);
        tracing::debug!("Entry point NVML hooks initialization result: {nvml_result}");
    }

    // Create initialization threads BEFORE installing dlsym hook
    // to avoid race condition where dlsym is called before threads exist
    
    // CUDA initialization thread
    if enable_cuda_hooks {
        tracing::debug!("Creating CUDA initialization thread");
        std::thread::spawn(move || {
            let (mutex, condvar) = &CUDA_SYMBOL_SYNC;
            let (dlsym_mutex, dlsym_condvar) = &CUDA_DLSYM_SYNC;

            loop {
                if let Ok(mut guard) = mutex.lock() {
                    while !*guard {
                        guard = condvar.wait(guard).unwrap_or_else(|e| e.into_inner());
                    }
                    // reset notification state
                    *guard = false;
                    tracing::debug!("CUDA thread received symbol notification, waking up");
                }

                tracing::debug!("CUDA symbol detected, attempting to initialize CUDA hooks");

                if !CUDA_HOOKS_INITIALIZED.load(Ordering::Acquire) {
                    // Force re-collection of module names in case library was loaded after startup
                    let mut hook_manager = HookManager::default();
                    hook_manager.collect_module_names();

                    let has_libcuda = hook_manager
                        .module_names
                        .iter()
                        .any(|m| m.starts_with("libcuda."));

                    tracing::debug!("CUDA thread: has_libcuda = {has_libcuda}");

                    if has_libcuda && init_cuda_hooks(true) {
                        tracing::debug!("CUDA hooks initialized successfully in thread");

                        // notify waiting dlsym calls
                        if let Ok(mut guard) = dlsym_mutex.lock() {
                            *guard = true;
                            dlsym_condvar.notify_all();
                        }
                        break;
                    } else if has_libcuda {
                        tracing::warn!("CUDA hooks initialization failed in thread, will retry");

                        // Always notify waiting dlsym calls even on failure
                        // so they don't wait indefinitely
                        if let Ok(mut guard) = dlsym_mutex.lock() {
                            *guard = true;
                            dlsym_condvar.notify_all();
                        }
                    } else {
                        tracing::debug!(
                            "CUDA library not yet loaded, will retry after next notification"
                        );

                        // Notify waiting dlsym calls that library isn't loaded yet
                        if let Ok(mut guard) = dlsym_mutex.lock() {
                            *guard = true;
                            dlsym_condvar.notify_all();
                        }
                    }
                } else {
                    // hooks are already initialized, notify waiting dlsym calls
                    // but continue monitoring for potential re-initialization needs
                    tracing::trace!("CUDA hooks already initialized, notifying waiting dlsym calls");
                    if let Ok(mut guard) = dlsym_mutex.lock() {
                        *guard = true;
                        dlsym_condvar.notify_all();
                    }
                    // Add a small delay to prevent busy-waiting when hooks are already initialized
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    // Continue the loop instead of breaking to handle potential re-initialization
                }
            }
        });
    }

    // NVML initialization thread  
    if enable_nvml_hooks {
        tracing::debug!("Creating NVML initialization thread");
        std::thread::spawn(move || {
            let (mutex, condvar) = &NVML_SYMBOL_SYNC;
            let (dlsym_mutex, dlsym_condvar) = &NVML_DLSYM_SYNC;

            loop {
                if let Ok(mut guard) = mutex.lock() {
                    while !*guard {
                        guard = condvar.wait(guard).unwrap_or_else(|e| e.into_inner());
                    }
                    // reset notification state
                    *guard = false;
                    tracing::debug!("NVML thread received symbol notification, waking up");
                }

                tracing::debug!("NVML symbol detected, attempting to initialize NVML hooks");

                if !NVML_HOOKS_INITIALIZED.load(Ordering::Acquire) {
                    // Force re-collection of module names in case library was loaded after startup
                    let mut hook_manager = HookManager::default();
                    hook_manager.collect_module_names();

                    let has_libnvml = hook_manager
                        .module_names
                        .iter()
                        .any(|m| m.starts_with("libnvidia-ml."));

                    tracing::debug!("NVML thread: has_libnvml = {has_libnvml}");

                    if has_libnvml && init_nvml_hooks(true) {
                        tracing::debug!("NVML hooks initialized successfully in thread");

                        // notify waiting dlsym calls
                        if let Ok(mut guard) = dlsym_mutex.lock() {
                            *guard = true;
                            dlsym_condvar.notify_all();
                        }
                        break;
                    } else if has_libnvml {
                        tracing::warn!("NVML hooks initialization failed in thread, will retry");

                        // Always notify waiting dlsym calls even on failure
                        // so they don't wait indefinitely
                        if let Ok(mut guard) = dlsym_mutex.lock() {
                            *guard = true;
                            dlsym_condvar.notify_all();
                        }
                    } else {
                        tracing::debug!(
                            "NVML library not yet loaded, will retry after next notification"
                        );

                        // Notify waiting dlsym calls that library isn't loaded yet
                        if let Ok(mut guard) = dlsym_mutex.lock() {
                            *guard = true;
                            dlsym_condvar.notify_all();
                        }
                    }
                } else {
                    // hooks are already initialized, notify waiting dlsym calls
                    // but continue monitoring for potential re-initialization needs
                    tracing::trace!("NVML hooks already initialized, notifying waiting dlsym calls");
                    if let Ok(mut guard) = dlsym_mutex.lock() {
                        *guard = true;
                        dlsym_condvar.notify_all();
                    }
                    // Add a small delay to prevent busy-waiting when hooks are already initialized
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    // Continue the loop instead of breaking to handle potential re-initialization
                }
            }
        });
    }

    // Install dlsym hook AFTER all initialization threads are created
    // to prevent race conditions
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
    
    tracing::debug!("All initialization threads created, dlsym hook installed");
}

thread_local! {
    static IN_DLSYM_DETOUR: Cell<bool> = const { Cell::new(false) };
    static IN_HOOK_INITIALIZATION: Cell<bool> = const { Cell::new(false) };
}

#[hook_fn]
unsafe extern "C" fn dlsym_detour(handle: *const c_void, symbol: *const c_char) -> *const c_void {
    if symbol.is_null() {
        return FN_DLSYM(handle, symbol);
    }
    let symbol_str = CStr::from_ptr(symbol).to_str().unwrap();
    let may_be_cuda = symbol_str.starts_with("cu");
    let may_be_nvml = symbol_str.starts_with("nvml");

    if !may_be_cuda && !may_be_nvml {
        return FN_DLSYM(handle, symbol);
    }

    // Check if we're already in dlsym or in hook initialization to prevent recursion
    if IN_DLSYM_DETOUR.with(|flag| flag.get()) || IN_HOOK_INITIALIZATION.with(|flag| flag.get()) {
        tracing::trace!(
            "dlsym recursion or hook initialization detected, calling original function directly"
        );
        return FN_DLSYM(handle, symbol);
    }
    // Set the flag to indicate we're in dlsym
    IN_DLSYM_DETOUR.with(|flag| flag.set(true));

    // Ensure we reset the flag when exiting, even in case of panic
    struct ResetGuard;
    impl Drop for ResetGuard {
        fn drop(&mut self) {
            IN_DLSYM_DETOUR.with(|flag| flag.set(false));
        }
    }
    let _guard = ResetGuard;

    tracing::trace!("dlsym: {symbol_str}");

    // check if the corresponding hooks have been initialized successfully
    let cuda_ready = CUDA_HOOKS_INITIALIZED.load(Ordering::Acquire);
    let nvml_ready = NVML_HOOKS_INITIALIZED.load(Ordering::Acquire);

    tracing::trace!("dlsym for {}: cuda_ready={}, nvml_ready={}", symbol_str, cuda_ready, nvml_ready);

    let should_notify = (may_be_cuda && !cuda_ready) || (may_be_nvml && !nvml_ready);

    if should_notify {
        if may_be_cuda && !cuda_ready {
            let (mutex, condvar) = &CUDA_SYMBOL_SYNC;
            if let Ok(mut guard) = mutex.lock() {
                *guard = true;
                condvar.notify_all();
            }
            tracing::debug!("Notified: detected CUDA symbol, hooks not ready yet");
        }

        if may_be_nvml && !nvml_ready {
            let (mutex, condvar) = &NVML_SYMBOL_SYNC;
            if let Ok(mut guard) = mutex.lock() {
                *guard = true;
                condvar.notify_all();
            }
            tracing::debug!("Notified: detected NVML symbol, hooks not ready yet");
        }

        // only wait if not in hook initialization
        if !IN_HOOK_INITIALIZATION.with(|flag| flag.get()) {
            // Wait for corresponding hook initialization thread based on symbol type
            if may_be_cuda && !cuda_ready {
                tracing::debug!("Starting CUDA dlsym wait for symbol: {}", symbol_str);
                let (dlsym_mutex, dlsym_condvar) = &CUDA_DLSYM_SYNC;
                if let Ok(mut guard) = dlsym_mutex.lock() {
                    loop {
                        if *guard || CUDA_HOOKS_INITIALIZED.load(Ordering::Acquire) {
                            // Reset the flag for next time and exit
                            *guard = false;
                            break;
                        }

                        match dlsym_condvar.wait_timeout(guard, Duration::from_millis(1000)) {
                            Ok((new_guard, timeout_result)) => {
                                if timeout_result.timed_out() {
                                    tracing::warn!(
                                        "CUDA dlsym wait timed out for symbol: {}, continuing",
                                        symbol_str
                                    );
                                    break;
                                }
                                guard = new_guard;
                            }
                            Err(poisoned) => {
                                let (new_guard, timeout_result) = poisoned.into_inner();
                                if timeout_result.timed_out() {
                                    tracing::warn!(
                                        "CUDA dlsym wait timed out (poisoned) for symbol: {}, continuing",
                                        symbol_str
                                    );
                                    break;
                                }
                                guard = new_guard;
                            }
                        }
                    }
                }
            }

            if may_be_nvml && !nvml_ready {
                tracing::debug!("Starting NVML dlsym wait for symbol: {}", symbol_str);
                let (dlsym_mutex, dlsym_condvar) = &NVML_DLSYM_SYNC;
                if let Ok(mut guard) = dlsym_mutex.lock() {
                    loop {
                        if *guard || NVML_HOOKS_INITIALIZED.load(Ordering::Acquire) {
                            // Reset the flag for next time and exit
                            *guard = false;
                            break;
                        }

                        match dlsym_condvar.wait_timeout(guard, Duration::from_millis(1000)) {
                            Ok((new_guard, timeout_result)) => {
                                if timeout_result.timed_out() {
                                    tracing::warn!(
                                        "NVML dlsym wait timed out for symbol: {}, continuing",
                                        symbol_str
                                    );
                                    break;
                                }
                                guard = new_guard;
                            }
                            Err(poisoned) => {
                                let (new_guard, timeout_result) = poisoned.into_inner();
                                if timeout_result.timed_out() {
                                    tracing::warn!(
                                        "NVML dlsym wait timed out (poisoned) for symbol: {}, continuing",
                                        symbol_str
                                    );
                                    break;
                                }
                                guard = new_guard;
                            }
                        }
                    }
                }
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
