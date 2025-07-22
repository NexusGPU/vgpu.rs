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
static CUDA_DLSYM_SYNC: (Mutex<u32>, Condvar) = (Mutex::new(0), Condvar::new());
static NVML_DLSYM_SYNC: (Mutex<u32>, Condvar) = (Mutex::new(0), Condvar::new());
static CUDA_HOOKS_INITIALIZED: AtomicBool = AtomicBool::new(false);
static NVML_HOOKS_INITIALIZED: AtomicBool = AtomicBool::new(false);
static IN_HOOK_INITIALIZATION_GLOBAL: AtomicBool = AtomicBool::new(false);
// Atomic counter to track the total number of ongoing hook initializations
// This provides cross-thread protection even when Frida operates on different threads
static HOOK_INITIALIZATION_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

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
    let current_thread = std::thread::current();
    let thread_name = current_thread.name().unwrap_or("unnamed");
    let thread_id = current_thread.id();
    tracing::debug!("init_cuda_hooks called on thread '{}' (id: {:?})", thread_name, thread_id);
    
    IN_HOOK_INITIALIZATION.with(|flag| flag.set(true));
    IN_HOOK_INITIALIZATION_GLOBAL.store(true, Ordering::Release);
    let _count = HOOK_INITIALIZATION_COUNT.fetch_add(1, Ordering::SeqCst);
    
    tracing::debug!("Set hook initialization flags (local=true, global=true, count={}) on thread '{}'", _count + 1, thread_name);

    // Ensure we reset all flags when exiting, even in case of panic
    struct ResetGuard {
        thread_name: String,
    }
    impl Drop for ResetGuard {
        fn drop(&mut self) {
            IN_HOOK_INITIALIZATION.with(|flag| flag.set(false));
            IN_HOOK_INITIALIZATION_GLOBAL.store(false, Ordering::Release);
            let count = HOOK_INITIALIZATION_COUNT.fetch_sub(1, Ordering::SeqCst);
            tracing::debug!("Reset hook initialization flags (count={}) on thread '{}'", count - 1, self.thread_name);
        }
    }
    let _guard = ResetGuard {
        thread_name: thread_name.to_string(),
    };

    init_ngpu_library();

    if !enable_cuda_hooks {
        return true;
    }

    // Early return if CUDA hooks are already initialized to prevent redundant initialization
    if CUDA_HOOKS_INITIALIZED.load(Ordering::Acquire) {
        tracing::debug!("CUDA hooks already initialized, skipping re-initialization on thread '{}'", thread_name);
        return true;
    }

    let mut hook_manager = HookManager::default();
    hook_manager.collect_module_names();

    if hook_manager
        .module_names
        .iter()
        .any(|m| m.starts_with("libcuda."))
    {
        tracing::debug!("Starting CUDA hooks installation...");
        
        // Use a timeout mechanism to prevent infinite hanging
        let start_time = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(5); // 5 second timeout
        
        let hook_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tracing::debug!("Starting detour::gpu::enable_hooks");
            unsafe {
                detour::gpu::enable_hooks(&mut hook_manager);
            }
            tracing::debug!("Completed detour::gpu::enable_hooks, starting detour::mem::enable_hooks");
            unsafe {
                detour::mem::enable_hooks(&mut hook_manager);
            }
            tracing::debug!("Completed detour::mem::enable_hooks");
        }));
        
        match hook_result {
            Ok(_) => {
                let elapsed = start_time.elapsed();
                tracing::debug!("CUDA hooks installation completed in {:?}", elapsed);
                
                if elapsed > timeout {
                    tracing::warn!("CUDA hooks installation took longer than expected: {:?}", elapsed);
                    // Still set to true, but with a warning
                }
                
                CUDA_HOOKS_INITIALIZED.store(true, Ordering::Release);
                tracing::debug!("CUDA_HOOKS_INITIALIZED flag set to true with Ordering::Release");
                return true;
            }
            Err(e) => {
                tracing::error!("CUDA hooks installation panicked: {:?}", e);
                // Don't set CUDA_HOOKS_INITIALIZED to true if installation failed
                return false;
            }
        }
    }
    false
}

fn init_nvml_hooks(enable_nvml_hooks: bool) -> bool {
    IN_HOOK_INITIALIZATION.with(|flag| flag.set(true));
    IN_HOOK_INITIALIZATION_GLOBAL.store(true, Ordering::Release);

    // Ensure we reset the flag when exiting, even in case of panic
    struct ResetGuard;
    impl Drop for ResetGuard {
        fn drop(&mut self) {
            IN_HOOK_INITIALIZATION.with(|flag| flag.set(false));
            IN_HOOK_INITIALIZATION_GLOBAL.store(false, Ordering::Release);
        }
    }
    let _guard = ResetGuard;

    init_ngpu_library();
    if !enable_nvml_hooks {
        return true;
    }

    // Early return if NVML hooks are already initialized to prevent redundant initialization
    if NVML_HOOKS_INITIALIZED.load(Ordering::Acquire) {
        tracing::debug!("NVML hooks already initialized, skipping re-initialization");
        return true;
    }

    let mut hook_manager = HookManager::default();
    hook_manager.collect_module_names();

    if hook_manager
        .module_names
        .iter()
        .any(|m| m.starts_with("libnvidia-ml."))
    {
        tracing::debug!("Starting NVML hooks installation...");
        
        // Use a timeout mechanism to prevent infinite hanging
        let start_time = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(5); // 5 second timeout
        
        let hook_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            unsafe {
                detour::nvml::enable_hooks(&mut hook_manager);
            }
        }));
        
        match hook_result {
            Ok(_) => {
                let elapsed = start_time.elapsed();
                tracing::debug!("NVML hooks installation completed in {:?}", elapsed);
                
                if elapsed > timeout {
                    tracing::warn!("NVML hooks installation took longer than expected: {:?}", elapsed);
                    // Still set to true, but with a warning
                }

                NVML_HOOKS_INITIALIZED.store(true, Ordering::Release);
                tracing::debug!("NVML_HOOKS_INITIALIZED flag set to true with Ordering::Release");
                return true;
            }
            Err(e) => {
                tracing::error!("NVML hooks installation panicked: {:?}", e);
                // Don't set NVML_HOOKS_INITIALIZED to true if installation failed
                return false;
            }
        }
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

    // CUDA initialization thread - use static Once to ensure only one thread is created
    if enable_cuda_hooks {
        static CUDA_THREAD_ONCE: Once = Once::new();
        CUDA_THREAD_ONCE.call_once(|| {
            tracing::debug!("Creating CUDA initialization thread");
            std::thread::spawn(move || {
                let (mutex, condvar) = &CUDA_SYMBOL_SYNC;
                let (dlsym_mutex, dlsym_condvar) = &CUDA_DLSYM_SYNC;

                loop {
                    let mut guard = match mutex.lock() {
                        Ok(guard) => guard,
                        Err(poisoned) => {
                            tracing::warn!("CUDA symbol sync mutex poisoned, recovering");
                            poisoned.into_inner()
                        }
                    };
                    while !*guard {
                        guard = condvar.wait(guard).unwrap_or_else(|e| {
                            tracing::warn!("CUDA condvar wait failed, recovering from poison");
                            e.into_inner()
                        });
                    }
                    // reset notification state
                    *guard = false;
                    tracing::debug!("CUDA thread received symbol notification, waking up");

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

                        let init_start = std::time::Instant::now();
                        let init_result = if has_libcuda {
                            init_cuda_hooks(true)
                        } else {
                            false
                        };
                        let init_duration = init_start.elapsed();
                        
                        if has_libcuda && init_result {
                            tracing::debug!("CUDA hooks initialized successfully in thread (took {:?})", init_duration);
                        } else if has_libcuda {
                            tracing::warn!(
                                "CUDA hooks initialization failed in thread (took {:?}), will retry",
                                init_duration
                            );
                            // Even if initialization failed, we should still notify waiting threads
                            // to prevent them from hanging forever
                        } else {
                            tracing::debug!(
                                "CUDA library not yet loaded, will retry after next notification"
                            );
                        }
                    } else {
                        tracing::trace!("CUDA hooks already initialized, skipping initialization");
                    }

                    // Always notify waiting dlsym calls regardless of initialization status
                    let mut guard = match dlsym_mutex.lock() {
                        Ok(guard) => guard,
                        Err(poisoned) => {
                            tracing::warn!("CUDA dlsym sync mutex poisoned, recovering");
                            poisoned.into_inner()
                        }
                    };
                    let old_guard_value = *guard;
                    *guard = guard.saturating_add(1);
                    let new_guard_value = *guard;
                    dlsym_condvar.notify_all();
                    tracing::debug!("CUDA thread notified all waiting dlsym calls: guard {} -> {}", old_guard_value, new_guard_value);
                }
            });
        });
    }

    // NVML initialization thread - use static Once to ensure only one thread is created
    if enable_nvml_hooks {
        static NVML_THREAD_ONCE: Once = Once::new();
        NVML_THREAD_ONCE.call_once(|| {
            tracing::debug!("Creating NVML initialization thread");
            std::thread::spawn(move || {
                let (mutex, condvar) = &NVML_SYMBOL_SYNC;
                let (dlsym_mutex, dlsym_condvar) = &NVML_DLSYM_SYNC;

                loop {
                    let mut guard = match mutex.lock() {
                        Ok(guard) => guard,
                        Err(poisoned) => {
                            tracing::warn!("NVML symbol sync mutex poisoned, recovering");
                            poisoned.into_inner()
                        }
                    };
                    while !*guard {
                        guard = condvar.wait(guard).unwrap_or_else(|e| {
                            tracing::warn!("NVML condvar wait failed, recovering from poison");
                            e.into_inner()
                        });
                    }
                    // reset notification state
                    *guard = false;
                    tracing::debug!("NVML thread received symbol notification, waking up");

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

                        let init_start = std::time::Instant::now();
                        let init_result = if has_libnvml {
                            init_nvml_hooks(true)
                        } else {
                            false
                        };
                        let init_duration = init_start.elapsed();
                        
                        if has_libnvml && init_result {
                            tracing::debug!("NVML hooks initialized successfully in thread (took {:?})", init_duration);
                        } else if has_libnvml {
                            tracing::warn!(
                                "NVML hooks initialization failed in thread (took {:?}), will retry",
                                init_duration
                            );
                            // Even if initialization failed, we should still notify waiting threads
                            // to prevent them from hanging forever
                        } else {
                            tracing::debug!(
                                "NVML library not yet loaded, will retry after next notification"
                            );
                        }
                    } else {
                        tracing::trace!("NVML hooks already initialized, skipping initialization");
                    }

                    // Always notify waiting dlsym calls regardless of initialization status
                    let mut guard = match dlsym_mutex.lock() {
                        Ok(guard) => guard,
                        Err(poisoned) => {
                            tracing::warn!("NVML dlsym sync mutex poisoned, recovering");
                            poisoned.into_inner()
                        }
                    };
                    let old_guard_value = *guard;
                    *guard = guard.saturating_add(1);
                    let new_guard_value = *guard;
                    dlsym_condvar.notify_all();
                    tracing::debug!("NVML thread notified all waiting dlsym calls: guard {} -> {}", old_guard_value, new_guard_value);
                }
            });
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
    let in_dlsym = IN_DLSYM_DETOUR.with(|flag| flag.get());
    let in_hook_init_local = IN_HOOK_INITIALIZATION.with(|flag| flag.get());
    let in_hook_init_global = IN_HOOK_INITIALIZATION_GLOBAL.load(Ordering::Acquire);
    let hook_init_count = HOOK_INITIALIZATION_COUNT.load(Ordering::Acquire);
    
    if in_dlsym || in_hook_init_local || in_hook_init_global || hook_init_count > 0 {
        tracing::debug!(
            "dlsym recursion or hook initialization detected for symbol '{}' (dlsym={}, local_init={}, global_init={}, count={}), calling original function directly",
            symbol_str, in_dlsym, in_hook_init_local, in_hook_init_global, hook_init_count
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

    let current_thread = std::thread::current();
    let thread_name = current_thread.name().unwrap_or("unnamed");
    let thread_id = current_thread.id();
    tracing::debug!("dlsym: {} on thread '{}' (id: {:?})", symbol_str, thread_name, thread_id);

    // check if the corresponding hooks have been initialized successfully
    let cuda_ready = CUDA_HOOKS_INITIALIZED.load(Ordering::Acquire);
    let nvml_ready = NVML_HOOKS_INITIALIZED.load(Ordering::Acquire);

    tracing::debug!(
        "dlsym for {}: cuda_ready={}, nvml_ready={} (Ordering::Acquire)",
        symbol_str,
        cuda_ready,
        nvml_ready
    );

    let should_notify = (may_be_cuda && !cuda_ready) || (may_be_nvml && !nvml_ready);

    if should_notify {
        if may_be_cuda && !cuda_ready {
            let (mutex, condvar) = &CUDA_SYMBOL_SYNC;
            let mut guard = match mutex.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    tracing::warn!("CUDA symbol sync mutex poisoned in dlsym, recovering");
                    poisoned.into_inner()
                }
            };
            *guard = true;
            condvar.notify_all();
            tracing::debug!("Notified: detected CUDA symbol, hooks not ready yet");
        }

        if may_be_nvml && !nvml_ready {
            let (mutex, condvar) = &NVML_SYMBOL_SYNC;
            let mut guard = match mutex.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    tracing::warn!("NVML symbol sync mutex poisoned in dlsym, recovering");
                    poisoned.into_inner()
                }
            };
            *guard = true;
            condvar.notify_all();
            tracing::debug!("Notified: detected NVML symbol, hooks not ready yet");
        }

        // only wait if not in hook initialization
        if !IN_HOOK_INITIALIZATION.with(|flag| flag.get()) {
            // Wait for corresponding hook initialization thread based on symbol type
            if may_be_cuda && !cuda_ready {
                // Check again after notification to avoid race condition  
                let hooks_ready_before_wait = CUDA_HOOKS_INITIALIZED.load(Ordering::Acquire);
                if !hooks_ready_before_wait {
                    tracing::debug!("Starting CUDA dlsym wait for symbol: {} (hooks_ready={})", symbol_str, hooks_ready_before_wait);
                    let (dlsym_mutex, dlsym_condvar) = &CUDA_DLSYM_SYNC;
                    let mut guard = match dlsym_mutex.lock() {
                        Ok(guard) => guard,
                        Err(poisoned) => {
                            tracing::warn!("CUDA dlsym sync mutex poisoned in dlsym wait, recovering");
                            poisoned.into_inner()
                        }
                    };
                    loop {
                        let cuda_hooks_ready = CUDA_HOOKS_INITIALIZED.load(Ordering::Acquire);
                        tracing::debug!(
                            "CUDA dlsym wait loop: symbol={}, guard={}, hooks_ready={}", 
                            symbol_str, *guard, cuda_hooks_ready
                        );
                        
                        if *guard > 0 || cuda_hooks_ready {
                            if *guard > 0 {
                                *guard = guard.saturating_sub(1);
                                tracing::debug!("CUDA dlsym wait exit via guard count: symbol={}, remaining_guard={}", symbol_str, *guard);
                            } else {
                                tracing::debug!("CUDA dlsym wait exit via hooks_ready: symbol={}", symbol_str);
                            }
                            break;
                        }

                        tracing::debug!("CUDA dlsym wait starting timeout: symbol={}", symbol_str);
                        match dlsym_condvar.wait_timeout(guard, Duration::from_millis(1000)) {
                            Ok((new_guard, timeout_result)) => {
                                if timeout_result.timed_out() {
                                    let final_hooks_ready = CUDA_HOOKS_INITIALIZED.load(Ordering::Acquire);
                                    tracing::warn!(
                                        "CUDA dlsym wait timed out for symbol: {}, final_guard={}, final_hooks_ready={}, continuing",
                                        symbol_str, *new_guard, final_hooks_ready
                                    );
                                    guard = new_guard;
                                    break;
                                }
                                guard = new_guard;
                                tracing::debug!("CUDA dlsym wait woke up from notification: symbol={}, guard={}", symbol_str, *guard);
                            }
                            Err(poisoned) => {
                                let (new_guard, timeout_result) = poisoned.into_inner();
                                if timeout_result.timed_out() {
                                    let final_hooks_ready = CUDA_HOOKS_INITIALIZED.load(Ordering::Acquire);
                                    tracing::warn!(
                                        "CUDA dlsym wait timed out (poisoned) for symbol: {}, final_guard={}, final_hooks_ready={}, continuing",
                                        symbol_str, *new_guard, final_hooks_ready
                                    );
                                    guard = new_guard;
                                    break;
                                }
                                guard = new_guard;
                                tracing::debug!("CUDA dlsym wait woke up from notification (poisoned): symbol={}, guard={}", symbol_str, *guard);
                            }
                        }
                    }
                }
            }

            if may_be_nvml && !nvml_ready {
                // Check again after notification to avoid race condition
                if !NVML_HOOKS_INITIALIZED.load(Ordering::Acquire) {
                    tracing::debug!("Starting NVML dlsym wait for symbol: {}", symbol_str);
                    let (dlsym_mutex, dlsym_condvar) = &NVML_DLSYM_SYNC;
                    let mut guard = match dlsym_mutex.lock() {
                        Ok(guard) => guard,
                        Err(poisoned) => {
                            tracing::warn!("NVML dlsym sync mutex poisoned in dlsym wait, recovering");
                            poisoned.into_inner()
                        }
                    };
                    loop {
                        if *guard > 0 || NVML_HOOKS_INITIALIZED.load(Ordering::Acquire) {
                            if *guard > 0 {
                                *guard = guard.saturating_sub(1);
                            }
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

    FN_DLSYM(handle, symbol)
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
