use ctor::ctor;

use limiter::{DeviceConfig, Limiter};
use nvml_wrapper::Nvml;
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{self, c_char, c_int, c_void, CStr},
    sync::{Mutex, OnceLock},
};
use tf_macro::hook_fn;
use trap::ipc::IpcTrap;
use utils::{hooks::HookManager, logging, replace_symbol};

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

pub fn global_trap() -> IpcTrap {
    static GLOBAL_TRAP: OnceLock<Mutex<IpcTrap>> = OnceLock::new();

    let trap = GLOBAL_TRAP.get_or_init(|| {
        let ipc_server_path_name =
            std::env::var("TENSOR_FUSION_IPC_SERVER_PATH").unwrap_or("cuda-limiter".to_string());
        // init IpcTrap
        let trap = if cfg!(test) {
            IpcTrap::dummy()
        } else {
            match IpcTrap::connect(ipc_server_path_name) {
                Ok(trap) => trap,
                Err(e) => {
                    panic!("failed to connect to ipc server, err: {e}");
                }
            }
        };
        Mutex::new(trap)
    });

    trap.lock().expect("poisoned").clone()
}

#[ctor]
unsafe fn entry_point() {
    logging::init();
    let pid = std::process::id();

    // Read up_limit and mem_limit from environment variables as JSON objects
    let up_limit_json =
        std::env::var("TENSOR_FUSION_CUDA_UP_LIMIT").unwrap_or_else(|_| "{}".to_string());

    let mem_limit_json =
        std::env::var("TENSOR_FUSION_CUDA_MEM_LIMIT").unwrap_or_else(|_| "{}".to_string());

    // Parse JSON objects
    let up_limit_map: HashMap<String, u32> = match serde_json::from_str(&up_limit_json) {
        Ok(map) => map,
        Err(e) => {
            tracing::error!("Failed to parse TENSOR_FUSION_CUDA_UP_LIMIT as JSON: {}", e);
            HashMap::new()
        }
    };

    let mem_limit_map: HashMap<String, u64> = match serde_json::from_str(&mem_limit_json) {
        Ok(map) => map,
        Err(e) => {
            tracing::error!(
                "Failed to parse TENSOR_FUSION_CUDA_MEM_LIMIT as JSON: {}",
                e
            );
            HashMap::new()
        }
    };

    // Initialize NVML
    let nvml = match Nvml::init().and(
        Nvml::builder()
            .lib_path(ffi::OsStr::new("libnvidia-ml.so.1"))
            .init(),
    ) {
        Ok(nvml) => nvml,
        Err(e) => {
            tracing::error!("Failed to initialize NVML: {}", e);
            return;
        }
    };

    // Create device configurations based on UUIDs
    let mut device_configs = Vec::new();
    let device_count = match nvml.device_count() {
        Ok(count) => count,
        Err(e) => {
            tracing::error!("Failed to get device count: {}", e);
            0
        }
    };

    for device_idx in 0..device_count {
        let device = match nvml.device_by_index(device_idx) {
            Ok(device) => device,
            Err(e) => {
                tracing::error!("Failed to get device at index {}: {}", device_idx, e);
                continue;
            }
        };

        let uuid = match device.uuid() {
            Ok(uuid) => uuid,
            Err(e) => {
                tracing::error!("Failed to get UUID for device {}: {}", device_idx, e);
                continue;
            }
        };

        // Get limits from maps based on UUID
        let up_limit = up_limit_map.get(&uuid).copied().unwrap_or(0);
        let mem_limit = mem_limit_map.get(&uuid).copied().unwrap_or(0);

        tracing::info!(
            "Device {}: UUID {}, up_limit: {}, mem_limit: {}",
            device_idx,
            uuid,
            up_limit,
            mem_limit
        );

        device_configs.push(DeviceConfig {
            device_idx,
            up_limit,
            mem_limit,
        });
    }

    // If no devices were found, use a default configuration
    if device_configs.is_empty() {
        device_configs.push(DeviceConfig {
            device_idx: 0,
            up_limit: 0,
            mem_limit: 0,
        });
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
