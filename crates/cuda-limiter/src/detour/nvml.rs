use nvml_wrapper_sys::bindings::nvmlDevice_t;
use nvml_wrapper_sys::bindings::nvmlEnableState_enum_NVML_FEATURE_DISABLED;
use nvml_wrapper_sys::bindings::nvmlEnableState_t;
use nvml_wrapper_sys::bindings::nvmlMemory_t;
use nvml_wrapper_sys::bindings::nvmlMemory_v2_t;
use nvml_wrapper_sys::bindings::nvmlReturn_enum_NVML_ERROR_NOT_FOUND;
use nvml_wrapper_sys::bindings::nvmlReturn_enum_NVML_ERROR_UNKNOWN;
use nvml_wrapper_sys::bindings::nvmlReturn_enum_NVML_SUCCESS;
use nvml_wrapper_sys::bindings::nvmlReturn_t;
use tf_macro::hook_fn;
use utils::hooks::HookManager;
use utils::replace_symbol;

use crate::GLOBAL_LIMITER;

#[hook_fn]
pub(crate) unsafe fn nvml_device_get_memory_info_detour(
    device: nvmlDevice_t,
    memory: *mut nvmlMemory_t,
) -> nvmlReturn_t {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");

    let device_index = match limiter.device_raw_index_by_nvml_handle(device) {
        Ok(device_index) => device_index,
        Err(e) => {
            tracing::error!("nvml_device_get_memory_info_detour: Failed to get device UUID: {e}");
            return nvmlReturn_enum_NVML_ERROR_NOT_FOUND;
        }
    };

    match limiter.get_pod_memory_usage(device_index) {
        Ok((used, mem_limit)) => {
            let memory_ref = &mut *memory;
            memory_ref.total = mem_limit;
            memory_ref.free = mem_limit.saturating_sub(used);
            memory_ref.used = used;
            nvmlReturn_enum_NVML_SUCCESS
        }
        Err(e) => {
            tracing::error!("Failed to get pod memory usage: {e}");
            nvmlReturn_enum_NVML_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn nvml_device_get_memory_info_v2_detour(
    device: nvmlDevice_t,
    memory: *mut nvmlMemory_v2_t,
) -> nvmlReturn_t {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    let device_index = match limiter.device_raw_index_by_nvml_handle(device) {
        Ok(device_index) => device_index,
        Err(e) => {
            tracing::error!(
                "nvml_device_get_memory_info_v2_detour: Failed to get device UUID: {e}"
            );
            return nvmlReturn_enum_NVML_ERROR_NOT_FOUND;
        }
    };

    match limiter.get_pod_memory_usage(device_index) {
        Ok((used, mem_limit)) => {
            let memory_ref = &mut *memory;
            memory_ref.total = mem_limit;
            memory_ref.free = mem_limit.saturating_sub(used);
            memory_ref.used = used;
            nvmlReturn_enum_NVML_SUCCESS
        }
        Err(e) => {
            tracing::error!("Failed to get memory limit: {e}");
            nvmlReturn_enum_NVML_ERROR_UNKNOWN
        }
    }
}

#[hook_fn]
pub(crate) unsafe fn nvml_device_get_persistence_mode_detour(
    _device: nvmlDevice_t,
    mode: *mut nvmlEnableState_t,
) -> nvmlReturn_t {
    // fix: https://forums.developer.nvidia.com/t/nvidia-smi-uses-all-of-ram-and-swap/295639/15
    let mode_ref = &mut *mode;
    *mode_ref = nvmlEnableState_enum_NVML_FEATURE_DISABLED;
    nvmlReturn_enum_NVML_SUCCESS
}

pub(crate) unsafe fn enable_hooks(hook_manager: &mut HookManager) {
    replace_symbol!(
        hook_manager,
        Some("libnvidia-ml."),
        "nvmlDeviceGetMemoryInfo",
        nvml_device_get_memory_info_detour,
        FnNvml_device_get_memory_info,
        FN_NVML_DEVICE_GET_MEMORY_INFO
    );
    replace_symbol!(
        hook_manager,
        Some("libnvidia-ml."),
        "nvmlDeviceGetMemoryInfo_v2",
        nvml_device_get_memory_info_v2_detour,
        FnNvml_device_get_memory_info_v2,
        FN_NVML_DEVICE_GET_MEMORY_INFO_V2
    );
    // replace_symbol!(
    //     hook_manager,
    //     Some("libnvidia-ml."),
    //     "nvmlDeviceGetCount_v2",
    //     nvml_device_get_count_v2_detour,
    //     FnNvml_device_get_count_v2,
    //     FN_NVML_DEVICE_GET_COUNT_V2
    // );
    // replace_symbol!(
    //     hook_manager,
    //     Some("libnvidia-ml."),
    //     "nvmlDeviceGetHandleByIndex_v2",
    //     nvml_device_get_handle_by_index_v2_detour,
    //     FnNvml_device_get_handle_by_index_v2,
    //     FN_NVML_DEVICE_GET_HANDLE_BY_INDEX_V2
    // );
    // replace_symbol!(
    //     hook_manager,
    //     Some("libnvidia-ml."),
    //     "nvmlDeviceGetHandleByIndex",
    //     nvml_device_get_handle_by_index_detour,
    //     FnNvml_device_get_handle_by_index,
    //     FN_NVML_DEVICE_GET_HANDLE_BY_INDEX
    // );
    // replace_symbol!(
    //     hook_manager,
    //     Some("libnvidia-ml."),
    //     "nvmlDeviceGetIndex",
    //     nvml_device_get_index_detour,
    //     FnNvml_device_get_index,
    //     FN_NVML_DEVICE_GET_INDEX
    // );
    replace_symbol!(
        hook_manager,
        Some("libnvidia-ml."),
        "nvmlDeviceGetPersistenceMode",
        nvml_device_get_persistence_mode_detour,
        FnNvml_device_get_persistence_mode,
        FN_NVML_DEVICE_GET_PERSISTENCE_MODE
    );
}
