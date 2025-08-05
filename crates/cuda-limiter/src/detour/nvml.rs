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

    let device_index = match limiter.device_index_by_nvml_handle(device) {
        Ok(Some(device_index)) => device_index,
        Ok(None) => {
            return nvmlReturn_enum_NVML_ERROR_NOT_FOUND;
        }
        Err(e) => {
            tracing::error!("Failed to get device UUID: {e}, falling back to original function");
            return FN_NVML_DEVICE_GET_MEMORY_INFO(device, memory);
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

    let device_index = match limiter.device_index_by_nvml_handle(device) {
        Ok(Some(device_index)) => device_index,
        Ok(None) => {
            return nvmlReturn_enum_NVML_ERROR_NOT_FOUND;
        }
        Err(e) => {
            tracing::error!("Failed to get device UUID: {e}, falling back to original function");
            return FN_NVML_DEVICE_GET_MEMORY_INFO_V2(device, memory);
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
    idx: c_uint,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    let raw_index = limiter.ordinal_to_raw_index(idx as usize);
    if let Some(raw_index) = raw_index {
        let result = FN_NVML_DEVICE_GET_HANDLE_BY_INDEX_V2(raw_index as c_uint, device);
        if result != nvmlReturn_enum_NVML_SUCCESS {
            tracing::error!(
                "failed to get handle by index v2: idx: {}, raw_index: {:?}, result: {:?}",
                idx,
                raw_index,
                result
            );
        }
        result
    } else {
        tracing::error!(
            "nvml_device_get_handle_by_index_v2_detour Invalid idx: {}, raw_index: {:?}",
            idx,
            raw_index
        );
        nvmlReturn_enum_NVML_ERROR_UNKNOWN
    }
}

#[hook_fn]
pub(crate) unsafe fn nvml_device_get_handle_by_index_detour(
    idx: c_uint,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    let raw_index = limiter.ordinal_to_raw_index(idx as usize);
    if let Some(raw_index) = raw_index {
        let result = FN_NVML_DEVICE_GET_HANDLE_BY_INDEX_V2(raw_index as c_uint, device);
        if result != nvmlReturn_enum_NVML_SUCCESS {
            tracing::error!(
                "failed to get handle by index: idx: {}, raw_index: {:?}, result: {:?}",
                idx,
                raw_index,
                result
            );
        }
        result
    } else {
        tracing::error!(
            "nvml_device_get_handle_by_index_detour Invalid idx: {}, raw_index: {:?}",
            idx,
            raw_index
        );
        nvmlReturn_enum_NVML_ERROR_UNKNOWN
    }
}

#[hook_fn]
pub(crate) unsafe fn nvml_device_get_index_detour(
    device: nvmlDevice_t,
    idx: *mut c_uint,
) -> nvmlReturn_t {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    let index = match limiter.device_index_by_nvml_handle(device) {
        Ok(Some(index)) => index,
        Ok(None) => {
            return nvmlReturn_enum_NVML_ERROR_NOT_FOUND;
        }
        Err(e) => {
            tracing::error!("Failed to get device UUID: {e}, falling back to original function");
            return nvmlReturn_enum_NVML_ERROR_UNKNOWN;
        }
    };
    *idx = index as c_uint;
    nvmlReturn_enum_NVML_SUCCESS
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
        "nvmlDeviceGetHandleByIndex",
        nvml_device_get_handle_by_index_detour,
        FnNvml_device_get_handle_by_index,
        FN_NVML_DEVICE_GET_HANDLE_BY_INDEX
    );
    replace_symbol!(
        hook_manager,
        Some("libnvidia-ml."),
        "nvmlDeviceGetIndex",
        nvml_device_get_index_detour,
        FnNvml_device_get_index,
        FN_NVML_DEVICE_GET_INDEX
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
