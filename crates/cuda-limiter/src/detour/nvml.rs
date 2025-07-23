use std::ffi::c_uint;

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

    let device_uuid = match limiter.device_uuid_by_handle(device) {
        Ok(Some(device_uuid)) => device_uuid,
        Ok(None) => {
            return nvmlReturn_enum_NVML_ERROR_NOT_FOUND;
        }
        Err(e) => {
            tracing::error!("Failed to get device UUID: {e}, falling back to original function");
            return FN_NVML_DEVICE_GET_MEMORY_INFO(device, memory);
        }
    };

    match limiter.get_pod_memory_usage(&device_uuid) {
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

    let device_uuid = match limiter.device_uuid_by_handle(device) {
        Ok(Some(device_uuid)) => device_uuid,
        Ok(None) => {
            return nvmlReturn_enum_NVML_ERROR_NOT_FOUND;
        }
        Err(e) => {
            tracing::error!("Failed to get device UUID: {e}, falling back to original function");
            return FN_NVML_DEVICE_GET_MEMORY_INFO_V2(device, memory);
        }
    };

    match limiter.get_pod_memory_usage(&device_uuid) {
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
pub(crate) unsafe fn nvml_device_get_count_v2_detour(device_count: *mut c_uint) -> nvmlReturn_t {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    let device_count_ref: &mut u32 = &mut *device_count;
    *device_count_ref = limiter.get_device_count();
    tracing::info!(
        "nvml_device_get_count_v2_detour: device count: {}",
        *device_count_ref
    );
    nvmlReturn_enum_NVML_SUCCESS
}

#[hook_fn]
pub(crate) unsafe fn nvml_device_get_handle_by_index_v2_detour(
    index: c_uint,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    let nvml_index = limiter.nvml_index_mapping(index as usize);
    if let Ok(nvml_index) = nvml_index {
        FN_NVML_DEVICE_GET_HANDLE_BY_INDEX_V2(nvml_index, device)
    } else {
        nvmlReturn_enum_NVML_ERROR_NOT_FOUND
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

pub(crate) unsafe fn enable_hooks(hook_manager: &mut HookManager, mapping_device_idx: bool) {
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
    if mapping_device_idx {
        replace_symbol!(
            hook_manager,
            Some("libnvidia-ml."),
            "nvmlDeviceGetCount_v2",
            nvml_device_get_count_v2_detour,
            FnNvml_device_get_count_v2,
            FN_NVML_DEVICE_GET_COUNT_V2
        );
        replace_symbol!(
            hook_manager,
            Some("libnvidia-ml."),
            "nvmlDeviceGetHandleByIndex_v2",
            nvml_device_get_handle_by_index_v2_detour,
            FnNvml_device_get_handle_by_index_v2,
            FN_NVML_DEVICE_GET_HANDLE_BY_INDEX_V2
        );
        replace_symbol!(
            hook_manager,
            Some("libnvidia-ml."),
            "nvmlDeviceGetPersistenceMode",
            nvml_device_get_persistence_mode_detour,
            FnNvml_device_get_persistence_mode,
            FN_NVML_DEVICE_GET_PERSISTENCE_MODE
        );
    }
}
