use std::os::raw::c_int;
use std::os::raw::c_ulong;

// Use basic types from nvml-wrapper-sys where available
pub(crate) use nvml_wrapper_sys::bindings::{nvmlDevice_t, nvmlReturn_t};
use tf_macro::hook_fn;
use utils::hooks::HookManager;
use utils::replace_symbol;

use crate::GLOBAL_LIMITER;

// NVML Return Values - keep these as constants for easier usage
pub const NVML_SUCCESS: u32 = 0;
#[allow(dead_code)]
pub const NVML_ERROR_UNINITIALIZED: u32 = 1;
#[allow(dead_code)]
pub const NVML_ERROR_INVALID_ARGUMENT: u32 = 2;
#[allow(dead_code)]
pub const NVML_ERROR_NOT_SUPPORTED: u32 = 3;
#[allow(dead_code)]
pub const NVML_ERROR_NO_PERMISSION: u32 = 4;
#[allow(dead_code)]
pub const NVML_ERROR_ALREADY_INITIALIZED: u32 = 5;
#[allow(dead_code)]
pub const NVML_ERROR_NOT_FOUND: u32 = 6;
#[allow(dead_code)]
pub const NVML_ERROR_INSUFFICIENT_SIZE: u32 = 7;
#[allow(dead_code)]
pub const NVML_ERROR_INSUFFICIENT_POWER: u32 = 8;
#[allow(dead_code)]
pub const NVML_ERROR_DRIVER_NOT_LOADED: u32 = 9;
#[allow(dead_code)]
pub const NVML_ERROR_TIMEOUT: u32 = 10;
#[allow(dead_code)]
pub const NVML_ERROR_IRQ_ISSUE: u32 = 11;
#[allow(dead_code)]
pub const NVML_ERROR_LIBRARY_NOT_FOUND: u32 = 12;
#[allow(dead_code)]
pub const NVML_ERROR_FUNCTION_NOT_FOUND: u32 = 13;
#[allow(dead_code)]
pub const NVML_ERROR_CORRUPTED_INFOROM: u32 = 14;
#[allow(dead_code)]
pub const NVML_ERROR_GPU_IS_LOST: u32 = 15;
#[allow(dead_code)]
pub const NVML_ERROR_RESET_REQUIRED: u32 = 16;
#[allow(dead_code)]
pub const NVML_ERROR_OPERATING_SYSTEM: u32 = 17;
#[allow(dead_code)]
pub const NVML_ERROR_LIB_RM_VERSION_MISMATCH: u32 = 18;
#[allow(dead_code)]
pub const NVML_ERROR_UNKNOWN: u32 = 999;

// NVML Memory info structures
#[repr(C)]
pub struct nvmlMemory_t {
    pub total: c_ulong,
    pub free: c_ulong,
    pub used: c_ulong,
}

#[repr(C)]
pub struct nvmlMemory_v2_t {
    pub version: c_int,
    pub total: c_ulong,
    pub reserved: c_ulong,
    pub free: c_ulong,
    pub used: c_ulong,
}

#[hook_fn]
pub(crate) unsafe fn nvml_device_get_memory_info_detour(
    _device: nvmlDevice_t,
    memory: *mut nvmlMemory_t,
) -> nvmlReturn_t {
    let _limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    // For now, just return success and fill with dummy values
    if !memory.is_null() {
        (*memory).total = 1024 * 1024 * 1024; // 1 GB
        (*memory).free = 512 * 1024 * 1024; // 512 MB
        (*memory).used = 512 * 1024 * 1024; // 512 MB
    }
    NVML_SUCCESS
}

#[hook_fn]
pub(crate) unsafe fn nvml_device_get_memory_info_v2_detour(
    _device: nvmlDevice_t,
    memory: *mut nvmlMemory_v2_t,
) -> nvmlReturn_t {
    let _limiter = GLOBAL_LIMITER.get().expect("Limiter not initialized");
    // For now, just return success and fill with dummy values
    if !memory.is_null() {
        (*memory).version = 2;
        (*memory).total = 1024 * 1024 * 1024; // 1 GB
        (*memory).reserved = 0;
        (*memory).free = 512 * 1024 * 1024; // 512 MB
        (*memory).used = 512 * 1024 * 1024; // 512 MB
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
