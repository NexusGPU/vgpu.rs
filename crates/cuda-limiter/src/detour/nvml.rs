use nvml_wrapper_sys::bindings::nvmlDevice_t;
use tf_macro::hook_fn;
use utils::hooks::HookManager;
use utils::replace_symbol;

use super::NvmlReturnT;
use super::NVML_SUCCESS;
use crate::GLOBAL_LIMITER;

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
    device: nvmlDevice_t,
    memory: *mut NvmlMemoryT,
) -> NvmlReturnT {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    let device_idx = match limiter.device_idx_by_handle(device) {
        Some(device_idx) => device_idx,
        None => return FN_NVML_DEVICE_GET_MEMORY_INFO(device, memory),
    };
    let mem_limit = limiter
        .get_mem_limit(device_idx)
        .expect("Failed to get memory limit");

    let used = limiter
        .get_used_gpu_memory(device_idx)
        .expect("Failed to get used GPU memory");

    let memory_ref = &mut *memory;
    // Modify the memory info to reflect our limits
    let new_total = mem_limit;
    let new_used = used;

    let new_free = if new_total > new_used {
        new_total.saturating_sub(new_used)
    } else {
        0 // Ensure free memory is not negative
    };

    memory_ref.total = new_total;
    memory_ref.used = new_used;
    memory_ref.free = new_free;
    NVML_SUCCESS
}

#[hook_fn]
pub(crate) unsafe fn nvml_device_get_memory_info_v2_detour(
    device: nvmlDevice_t,
    memory: *mut NvmlMemoryV2T,
) -> NvmlReturnT {
    let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    let device_idx = match limiter.device_idx_by_handle(device) {
        Some(device_idx) => device_idx,
        None => return FN_NVML_DEVICE_GET_MEMORY_INFO_V2(device, memory),
    };
    let mem_limit = limiter
        .get_mem_limit(device_idx)
        .expect("Failed to get memory limit");

    let used = limiter
        .get_used_gpu_memory(device_idx)
        .expect("Failed to get used GPU memory");

    let memory_ref = &mut *memory;
    // Modify the memory info to reflect our limits
    let new_total = mem_limit;
    let new_used = used;
    let new_free = if new_total > new_used {
        new_total.saturating_sub(new_used)
    } else {
        0 // Ensure free memory is not negative
    };

    memory_ref.total = new_total;
    memory_ref.used = new_used;
    memory_ref.free = new_free;
    NVML_SUCCESS
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
