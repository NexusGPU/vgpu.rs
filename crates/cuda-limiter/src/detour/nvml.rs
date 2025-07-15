use nvml_wrapper_sys::bindings::nvmlDevice_t;
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
}
