use cudarc::driver::sys::CUdevice;
use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUuuid;
use cudarc::driver::sys::Lib;
use libloading::Library;
use std::ffi::c_int;
use std::ffi::c_uint;

pub unsafe fn culib() -> &'static Lib {
    let lib_path = std::env::var("TF_CUDA_LIB_PATH")
        .unwrap_or_else(|_| "/lib/x86_64-linux-gnu/libcuda.so".to_string());

    static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
    LIB.get_or_init(|| {
        tracing::info!("Loading CUDA library from {}", lib_path);
        let lib = Library::new(lib_path).expect("Failed to load CUDA library");
        Lib::from_library(lib).expect("Failed to convert library to Lib")
    })
}

pub unsafe fn cu_device_get_uuid(device: CUdevice) -> Result<CUuuid, CUresult> {
    let mut uuid = CUuuid { bytes: [0; 16] };
    let result = (culib().cuDeviceGetUuid_v2)(&mut uuid, device);
    if result == CUresult::CUDA_SUCCESS {
        Ok(uuid)
    } else {
        Err(result)
    }
}

#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
pub fn uuid_to_string_formatted(uuid: &[i8; 16]) -> String {
    format!(
        "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        uuid[0], uuid[1], uuid[2], uuid[3],
        uuid[4], uuid[5],
        uuid[6], uuid[7],
        uuid[8], uuid[9],
        uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15],
    )
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
pub fn uuid_to_string_formatted(uuid: &[u8; 16]) -> String {
    format!(
        "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        uuid[0], uuid[1], uuid[2], uuid[3],
        uuid[4], uuid[5],
        uuid[6], uuid[7],
        uuid[8], uuid[9],
        uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15],
    )
}

pub(crate) fn device_uuid(cu_device: CUdevice) -> Result<String, CUresult> {
    unsafe {
        let uuid = cu_device_get_uuid(cu_device)?;
        Ok(uuid_to_string_formatted(&uuid.bytes))
    }
}
