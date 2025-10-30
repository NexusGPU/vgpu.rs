use cudarc::driver::sys::CUdevice;
use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUuuid;
use cudarc::driver::sys::Lib;
use libloading::Library;

const DEFAULT_CUDA_LIB_PATH: &str = "/lib/x86_64-linux-gnu/libcuda.so";

/// Resolve CUDA library path from environment or system
fn resolve_cuda_lib_path() -> String {
    if let Ok(path) = std::env::var("TF_CUDA_LIB_PATH") {
        return path;
    }

    if std::path::Path::new(DEFAULT_CUDA_LIB_PATH).exists() {
        return DEFAULT_CUDA_LIB_PATH.to_string();
    }

    tracing::info!("Default CUDA library path not found, searching with ldconfig");
    find_libcuda_with_ldconfig().unwrap_or_else(|| {
        tracing::warn!("Could not find libcuda.so via ldconfig, falling back to default path");
        DEFAULT_CUDA_LIB_PATH.to_string()
    })
}

pub unsafe fn culib() -> &'static Lib {
    static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
    LIB.get_or_init(|| {
        let lib_path = resolve_cuda_lib_path();
        tracing::info!("Loading CUDA library from {}", lib_path);
        let lib = Library::new(&lib_path).expect("Failed to load CUDA library");
        Lib::from_library(lib).expect("Failed to convert library to Lib")
    })
}

/// Find libcuda.so using ldconfig
fn find_libcuda_with_ldconfig() -> Option<String> {
    let output = std::process::Command::new("ldconfig")
        .arg("-p")
        .output()
        .ok()?;

    if !output.status.success() {
        tracing::warn!("ldconfig command failed");
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse ldconfig output to find libcuda.so
    // Format: "	libcuda.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcuda.so.1"
    for line in stdout.lines() {
        if line.contains("libcuda.so") {
            // Extract the path after "=>"
            if let Some(path_part) = line.split("=>").nth(1) {
                let path = path_part.trim();
                // Verify the path exists
                if std::path::Path::new(path).exists() {
                    return Some(path.to_string());
                }
            }
        }
    }

    None
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
