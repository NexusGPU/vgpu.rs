use tf_macro::hook_fn;
use utils::{hooks::HookManager, replace_symbol};

use crate::{with_device, GLOBAL_LIMITER};

use super::{NvmlDeviceT, NvmlReturnT, NVML_SUCCESS};

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
    // Get device handle for current device using with_device macro directly
    let device_match = with_device!(|current_device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        let handle = limiter
            .device_handle(current_device)
            .expect("Failed to get device handle");
        handle == device
    });

    if !device_match {
        return FN_NVML_DEVICE_GET_MEMORY_INFO(device, memory);
    }

    // Get memory limit and used memory directly with with_device
    let mem_limit = with_device!(|current_device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        limiter
            .get_mem_limit(current_device)
            .expect("Failed to get memory limit")
    });

    let used = with_device!(|current_device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        limiter
            .get_used_gpu_memory(current_device)
            .expect("Failed to get used GPU memory")
    });

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
    // Get device handle for current device using with_device macro directly
    let device_match = with_device!(|current_device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        let handle = limiter
            .device_handle(current_device)
            .expect("Failed to get device handle");
        handle == device
    });

    if !device_match {
        return FN_NVML_DEVICE_GET_MEMORY_INFO_V2(device, memory);
    }

    // Get memory limit and used memory directly with with_device
    let mem_limit = with_device!(|current_device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        limiter
            .get_mem_limit(current_device)
            .expect("Failed to get memory limit")
    });

    let used = with_device!(|current_device| {
        let limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
        limiter
            .get_used_gpu_memory(current_device)
            .expect("Failed to get used GPU memory")
    });

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
