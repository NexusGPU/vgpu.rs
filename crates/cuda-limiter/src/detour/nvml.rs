use tf_macro::hook_fn;
use utils::{hooks::HookManager, replace_symbol};

use crate::GLOBAL_LIMITER;

use super::{NvmlDeviceT, NvmlReturnT, NVML_ERROR_UNKNOWN, NVML_SUCCESS};

// NVML Memory Info structure
#[repr(C)]
pub(crate) struct NvmlMemoryT {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

// NVML Memory Info V2 structure
#[repr(C)]
pub(crate) struct NvmlMemoryV2T {
    pub version: u32,
    pub total: u64,
    pub reserved: u64,
    pub free: u64,
    pub used: u64,
}

#[hook_fn]
pub(crate) unsafe fn nvml_device_get_memory_info_detour(
    device: NvmlDeviceT,
    memory: *mut NvmlMemoryT,
) -> NvmlReturnT {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    match limiter.device_handle() {
        Ok(device_handle) => {
            if device_handle != device {
                return FN_NVML_DEVICE_GET_MEMORY_INFO(device, memory);
            }
        }
        Err(err) => {
            tracing::error!("get device handle failed: {}", err);
            return NVML_ERROR_UNKNOWN;
        }
    }

    let mem_limit = limiter.get_mem_limit();
    let used = match limiter.get_used_gpu_memory() {
        Ok(used) => used,
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            return NVML_ERROR_UNKNOWN;
        }
    };
    if mem_limit < u64::MAX {
        let memory_ref = &mut *memory;
        // Modify the memory info to reflect our limits
        let new_total = mem_limit;
        let new_used = used;
        let new_free = new_total.saturating_sub(new_used);
        memory_ref.total = new_total;
        memory_ref.used = new_used;
        memory_ref.free = new_free;
    }
    NVML_SUCCESS
}

#[hook_fn]
pub(crate) unsafe fn nvml_device_get_memory_info_v2_detour(
    device: NvmlDeviceT,
    memory: *mut NvmlMemoryV2T,
) -> NvmlReturnT {
    let limiter = GLOBAL_LIMITER.get().expect("get limiter");
    match limiter.device_handle() {
        Ok(device_handle) => {
            if device_handle != device {
                return FN_NVML_DEVICE_GET_MEMORY_INFO_V2(device, memory);
            }
        }
        Err(err) => {
            tracing::error!("get device handle failed: {}", err);
            return NVML_ERROR_UNKNOWN;
        }
    }

    let mem_limit = limiter.get_mem_limit();
    let used = match limiter.get_used_gpu_memory() {
        Ok(used) => used,
        Err(e) => {
            tracing::warn!("Failed to get used GPU memory: {:?}", e);
            return NVML_ERROR_UNKNOWN;
        }
    };
    if mem_limit < u64::MAX {
        let memory_ref = &mut *memory;
        // Modify the memory info to reflect our limits
        let new_total = mem_limit;
        let new_used = used;
        let new_free = new_total.saturating_sub(new_used);

        memory_ref.total = new_total;
        memory_ref.used = new_used;
        memory_ref.free = new_free;
    }
    NVML_SUCCESS
}

pub(crate) unsafe fn enable_hooks(hook_manager: &mut HookManager) {
    // Add hooks for NVML memory info functions
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
