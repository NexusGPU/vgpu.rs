use std::cell::Cell;
use std::collections::HashSet;
use std::env;
use std::ffi::c_char;
use std::ffi::c_void;
use std::ffi::CStr;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Once;
use std::sync::OnceLock;

use ctor::ctor;
use limiter::Limiter;
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
use crate::nvmllib::init_nvml;

mod auto_freeze;
mod checkpoint;
mod config;
mod culib;
mod detour;
mod limiter;
mod nvmllib;

static GLOBAL_LIMITER: OnceLock<Limiter> = OnceLock::new();
static GLOBAL_AUTO_FREEZE_MANAGER: OnceLock<Arc<AutoFreezeManager>> = OnceLock::new();
static GLOBAL_LIMITER_ERROR: OnceLock<String> = OnceLock::new();
static HOOKS_INITIALIZED: (AtomicBool, AtomicBool) =
    (AtomicBool::new(false), AtomicBool::new(false));
static LIMITER_ERROR_REPORTED: AtomicBool = AtomicBool::new(false);

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

pub(crate) fn should_skip_hooks_on_no_limit() -> bool {
    static SKIP_HOOKS_ON_NO_LIMIT: OnceLock<bool> = OnceLock::new();
    *SKIP_HOOKS_ON_NO_LIMIT.get_or_init(|| {
        env::var("TF_SKIP_HOOKS_IF_NO_LIMIT")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false)
    })
}

fn record_limiter_error(message: impl Into<String>) {
    let message = message.into();
    tracing::error!("{message}");
    let _ = GLOBAL_LIMITER_ERROR.set(message);
}

pub(crate) fn report_limiter_not_initialized() {
    if LIMITER_ERROR_REPORTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
        .is_ok()
    {
        if let Some(reason) = GLOBAL_LIMITER_ERROR.get() {
            tracing::warn!("Limiter not initialized; last error: {reason}");
        } else {
            tracing::warn!("Limiter not initialized; init has not run");
        }
    }
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

fn remap_visible_devices(allocated_devices: &[String]) -> Result<String, String> {
    if env::var("TF_REMAPPED").is_ok() {
        if let Ok(current) = env::var("CUDA_VISIBLE_DEVICES") {
            return Ok(current.trim().to_string());
        }
        return Ok(allocated_devices.join(","));
    }

    let original = env::var("CUDA_VISIBLE_DEVICES").ok();
    let Some(original) = original else {
        return Ok(allocated_devices.join(","));
    };

    let trimmed = original.trim();
    if trimmed.is_empty() {
        return Ok(allocated_devices.join(","));
    }

    if trimmed.contains(',') {
        return Ok(allocated_devices.join(","));
    }

    let virtual_id = trimmed
        .parse::<usize>()
        .map_err(|_| format!("Invalid device ID in CUDA_VISIBLE_DEVICES: '{}'", trimmed))?;

    if virtual_id >= allocated_devices.len() {
        return Err(format!(
            "Virtual device ID {} out of range (only {} device(s) allocated)",
            virtual_id,
            allocated_devices.len()
        ));
    }

    env::set_var("TF_REMAPPED", "1");
    Ok(allocated_devices[virtual_id].clone())
}

fn init_limiter() {
    static LIMITER_INITIALIZED: Once = Once::new();
    LIMITER_INITIALIZED.call_once(|| {
        let nvml = match init_nvml() {
            Ok(nvml) => nvml,
            Err(e) => {
                record_limiter_error(format!("failed to initialize NVML: {e}"));
                return;
            }
        };

        let config = if mock_shm_path().is_none() {
            let (hypervisor_ip, hypervisor_port) = match config::get_hypervisor_config() {
                Some((ip, port)) => (ip, port),
                None => {
                    record_limiter_error(
                        "HYPERVISOR_IP or HYPERVISOR_PORT not set; skipping limiter init"
                            .to_string(),
                    );
                    return;
                }
            };

            // Get device indices from environment variable
            let config = match config::get_worker_config(&hypervisor_ip, &hypervisor_port) {
                Ok(config) => config,
                Err(err) => {
                    record_limiter_error(format!("failed to get device configs: {err}"));
                    return;
                }
            };
            config
        } else {
            // In mock/test mode, derive UUIDs from either CUDA_VISIBLE_DEVICES or list all devices
            let uuids = match env::var("TF_VISIBLE_DEVICES") {
                Ok(visible_devices) => visible_devices
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>(),
                Err(_) => {
                    record_limiter_error(
                        "TF_VISIBLE_DEVICES not set in mock/test mode; skipping limiter init"
                            .to_string(),
                    );
                    return;
                }
            };

            config::PodConfig {
                gpu_uuids: uuids,
                compute_shard: false,
                isolation: None,
                auto_freeze: None,
            }
        };

        if !config.gpu_uuids.is_empty() {
            let lower_case_uuids: HashSet<_> =
                config.gpu_uuids.iter().map(|u| u.to_lowercase()).collect();
            let device_count = match nvml.device_count() {
                Ok(count) => count,
                Err(err) => {
                    record_limiter_error(format!("failed to get device count: {err}"));
                    return;
                }
            };

            let mut device_indices = Vec::new();
            for i in 0..device_count {
                let device = match nvml.device_by_index(i) {
                    Ok(device) => device,
                    Err(err) => {
                        record_limiter_error(format!("failed to get device by index {i}: {err}"));
                        return;
                    }
                };
                let uuid = match device.uuid() {
                    Ok(uuid) => uuid,
                    Err(err) => {
                        record_limiter_error(format!(
                            "failed to get device uuid for index {i}: {err}"
                        ));
                        return;
                    }
                };
                if lower_case_uuids.contains(&uuid.to_lowercase()) {
                    device_indices.push(i.to_string());
                }
            }

            if !device_indices.is_empty() {
                device_indices.sort_by_key(|id| id.parse::<u32>().unwrap_or(u32::MAX));

                let is_inherited = env::var("TF_REMAPPED").is_ok();
                let original = env::var("CUDA_VISIBLE_DEVICES").ok();

                let visible_devices = match remap_visible_devices(&device_indices) {
                    Ok(devices) => devices,
                    Err(err) => {
                        record_limiter_error(err);
                        return;
                    }
                };

                if is_inherited {
                    tracing::info!(
                        "Device inherited, CUDA_VISIBLE_DEVICES set to '{}' (allocated devices: {})",
                        &visible_devices,
                        device_indices.join(",")
                    );
                } else if let Some(ref original) = original {
                    tracing::info!(
                        "Remapping CUDA_VISIBLE_DEVICES from '{}' to '{}' (allocated devices: {})",
                        original,
                        &visible_devices,
                        device_indices.join(",")
                    );
                } else {
                    tracing::info!(
                        "Setting CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES to {}",
                        &visible_devices
                    );
                }

                env::set_var("CUDA_VISIBLE_DEVICES", &visible_devices);
                env::set_var("NVIDIA_VISIBLE_DEVICES", &visible_devices);
            }
        }

        let limiter = match Limiter::new(
            nvml,
            config.gpu_uuids,
            config.compute_shard,
            config.isolation,
        ) {
            Ok(limiter) => limiter,
            Err(err) => {
                record_limiter_error(format!("failed to initialize limiter: {err}"));
                return;
            }
        };

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

    if !utils::hooks::is_module_loaded("libcuda.") {
        return;
    }

    tracing::debug!("Installing CUDA hooks...");

    let install_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let mut hook_manager = HookManager::default();
        detour::gpu::enable_hooks(&mut hook_manager)
            .and_then(|_| detour::mem::enable_hooks(&mut hook_manager))
    }));

    match install_result {
        Ok(Ok(())) => {
            HOOKS_INITIALIZED.1.store(true, Ordering::Release);
            tracing::debug!("CUDA hooks installed successfully");
        }
        Ok(Err(e)) => {
            tracing::error!("CUDA hooks installation failed: {}", e);
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

    if !utils::hooks::is_module_loaded("libnvidia-ml.") {
        return;
    }

    tracing::debug!("Installing NVML hooks...");

    let install_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let mut hook_manager = HookManager::default();
        detour::nvml::enable_hooks(&mut hook_manager, is_nvidia_smi())
    }));

    match install_result {
        Ok(Ok(())) => {
            HOOKS_INITIALIZED.0.store(true, Ordering::Release);
            tracing::debug!("NVML hooks installed successfully");
        }
        Ok(Err(e)) => {
            tracing::error!("NVML hooks installation failed: {}", e);
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

    init_limiter();

    let limiter = match GLOBAL_LIMITER.get() {
        Some(limiter) => limiter,
        None => {
            if let Some(reason) = GLOBAL_LIMITER_ERROR.get() {
                panic!("GLOBAL_LIMITER initialization failed: {reason}");
            } else {
                panic!("GLOBAL_LIMITER not initialized: init has not run");
            }
        }
    };

    let is_nvidia_smi = is_nvidia_smi();

    let isolation = limiter.isolation();
    let should_skip_isolation =
        limiter.is_compute_shard() || isolation.is_some_and(|iso| iso == "soft" || iso == "hard");

    if should_skip_isolation && !is_nvidia_smi {
        tracing::info!(
            "Isolation level '{}' detected, skipping hook initialization",
            isolation.unwrap_or("soft")
        );
        return;
    }

    unsafe {
        // Load CUDA library to ensure it's loaded before hooks are installed
        let _ = culib::culib();
    }

    // Check if should skip hooks when all devices are unlimited
    let all_unlimited = GLOBAL_LIMITER
        .get()
        .map(|limiter| limiter.all_devices_unlimited())
        .unwrap_or(false);

    let should_skip_hooks = should_skip_hooks_on_no_limit() && all_unlimited;

    if should_skip_hooks {
        if is_nvidia_smi {
            tracing::info!(
                "All devices have up_limit >= 100, but nvidia-smi detected, will install NVML hooks only"
            );
        } else {
            tracing::info!("All devices have up_limit >= 100, skipping all hooks installation");
            return;
        }
    }

    // Try to install hooks immediately if libraries are already loaded
    let has_libcuda = utils::hooks::is_module_loaded("libcuda.");
    let has_libnvml = utils::hooks::is_module_loaded("libnvidia-ml.");

    tracing::debug!("has_libcuda: {has_libcuda}, has_libnvml: {has_libnvml}");

    if has_libcuda && !should_skip_hooks {
        try_install_cuda_hooks();
    }

    if has_libnvml && (!should_skip_hooks || is_nvidia_smi) {
        try_install_nvml_hooks();
    }

    // Install dlsym hook to catch dynamic library loading
    static DLSYM_HOOK_ONCE: Once = Once::new();
    DLSYM_HOOK_ONCE.call_once(|| {
        let mut hook_manager = HookManager::default();
        if let Err(err) = replace_symbol!(
            &mut hook_manager,
            None,
            "dlsym",
            dlsym_detour,
            FnDlsym,
            FN_DLSYM
        ) {
            tracing::error!("Failed to install dlsym hook: {}", err);
        }
    });
    tracing::debug!("Hook initialization completed");
}

thread_local! {
    static IN_DLSYM_DETOUR: Cell<bool> = const { Cell::new(false) };
}

fn call_original_dlsym(handle: *const c_void, symbol: *const c_char) -> *const c_void {
    if let Some(original) = FN_DLSYM.get() {
        unsafe { original(handle, symbol) }
    } else {
        unsafe { libc::dlsym(handle as *mut c_void, symbol) }
    }
}

#[hook_fn]
unsafe extern "C" fn dlsym_detour(handle: *const c_void, symbol: *const c_char) -> *const c_void {
    if symbol.is_null() {
        return call_original_dlsym(handle, symbol);
    }

    let Ok(symbol_str) = CStr::from_ptr(symbol).to_str() else {
        return call_original_dlsym(handle, symbol);
    };
    let may_be_cuda = symbol_str.starts_with("cu");
    let may_be_nvml = symbol_str.starts_with("nvml");

    if !may_be_cuda && !may_be_nvml {
        return call_original_dlsym(handle, symbol);
    }

    // Prevent recursion
    if IN_DLSYM_DETOUR.with(|flag| flag.get()) {
        return call_original_dlsym(handle, symbol);
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
        tracing::debug!("dlsym observed CUDA symbol {symbol_str}, ensuring hooks installed");
        try_install_cuda_hooks();
    }

    if may_be_nvml && !HOOKS_INITIALIZED.0.load(Ordering::Acquire) {
        tracing::debug!("dlsym observed NVML symbol {symbol_str}, ensuring hooks installed");
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

    trap.lock()
        .map(|guard| guard.clone())
        .unwrap_or_else(|poisoned| {
            tracing::warn!("global trap mutex poisoned, reusing inner trap");
            poisoned.into_inner().clone()
        })
}

pub(crate) fn is_nvidia_smi() -> bool {
    match fs::read_to_string("/proc/self/cmdline") {
        Ok(cmdline) => cmdline.contains("nvidia-smi"),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_remap_single_device_valid_first() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "0");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("2".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_single_device_valid_second() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "1");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("3".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_single_device_out_of_range() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "2");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Virtual device ID 2 out of range"));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_single_device_out_of_range_large() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "10");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Virtual device ID 10 out of range"));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_multiple_devices_no_remap() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "0,1");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("2,3".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_multiple_devices_with_spaces() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "0, 1, 2");
        let allocated = vec!["2".to_string(), "3".to_string(), "5".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("2,3,5".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_no_original_env() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("2,3".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_empty_original_env() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("2,3".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_whitespace_only_original_env() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "  ");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("2,3".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_empty_allocated_devices() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "0");
        let allocated: Vec<String> = vec![];
        let result = remap_visible_devices(&allocated);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Virtual device ID 0 out of range"));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_invalid_device_id_letters() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "abc");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid device ID"));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_invalid_device_id_special_chars() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "@#$");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid device ID"));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_single_device_with_whitespace() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "  1  ");
        let allocated = vec!["2".to_string(), "3".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("3".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_single_allocated_device() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "0");
        let allocated = vec!["5".to_string()];
        let result = remap_visible_devices(&allocated);
        assert_eq!(result, Ok("5".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_remap_single_allocated_device_out_of_range() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "1");
        let allocated = vec!["5".to_string()];
        let result = remap_visible_devices(&allocated);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Virtual device ID 1 out of range"));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_inherited_device_keeps_value() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "0");
        let first_allocated = vec!["1".to_string(), "2".to_string()];
        let first_result = remap_visible_devices(&first_allocated);
        assert_eq!(first_result, Ok("1".to_string()));

        env::set_var("CUDA_VISIBLE_DEVICES", "1");
        let second_allocated = vec!["3".to_string(), "4".to_string()];
        let second_result = remap_visible_devices(&second_allocated);
        assert_eq!(second_result, Ok("1".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_inherited_device_in_new_allocation() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "0");
        let first_allocated = vec!["1".to_string(), "2".to_string()];
        let first_result = remap_visible_devices(&first_allocated);
        assert_eq!(first_result, Ok("1".to_string()));

        env::set_var("CUDA_VISIBLE_DEVICES", "1");
        let second_allocated = vec!["1".to_string(), "3".to_string()];
        let second_result = remap_visible_devices(&second_allocated);
        assert_eq!(second_result, Ok("1".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_inherited_single_to_multiple_devices() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "0");
        let first_allocated = vec!["2".to_string(), "3".to_string()];
        let first_result = remap_visible_devices(&first_allocated);
        assert_eq!(first_result, Ok("2".to_string()));
        assert!(env::var("TF_REMAPPED").is_ok());

        env::set_var("CUDA_VISIBLE_DEVICES", "2,3");
        let second_allocated = vec!["2".to_string(), "3".to_string(), "4".to_string()];
        let second_result = remap_visible_devices(&second_allocated);
        assert_eq!(second_result, Ok("2,3".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_multiple_devices_no_remapped_flag() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "0,1");
        let first_allocated = vec!["2".to_string(), "3".to_string()];
        let first_result = remap_visible_devices(&first_allocated);
        assert_eq!(first_result, Ok("2,3".to_string()));
        assert!(env::var("TF_REMAPPED").is_err());

        let second_allocated = vec!["4".to_string(), "5".to_string()];
        let second_result = remap_visible_devices(&second_allocated);
        assert_eq!(second_result, Ok("4,5".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }

    #[test]
    #[serial]
    fn test_empty_env_no_remapped_flag() {
        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");

        env::set_var("CUDA_VISIBLE_DEVICES", "");
        let first_allocated = vec!["2".to_string(), "3".to_string()];
        let first_result = remap_visible_devices(&first_allocated);
        assert_eq!(first_result, Ok("2,3".to_string()));
        assert!(env::var("TF_REMAPPED").is_err());

        let second_allocated = vec!["4".to_string(), "5".to_string()];
        let second_result = remap_visible_devices(&second_allocated);
        assert_eq!(second_result, Ok("4,5".to_string()));

        env::remove_var("CUDA_VISIBLE_DEVICES");
        env::remove_var("TF_REMAPPED");
    }
}
