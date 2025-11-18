use std::sync::OnceLock;

use cudarc::driver::sys::CUdevice;
use cudarc::driver::sys::CUresult;
use cudarc::driver::sys::CUuuid;
use cudarc::driver::sys::Lib;
use libloading::{Library, Symbol};

#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

const PRIMARY_CUDA_LIB: &str = "libcuda.so.1";
const FALLBACK_CUDA_LIB: &str = "libcuda.so";

/// Provide candidate CUDA library names to attempt in order
fn candidate_cuda_libs() -> Vec<String> {
    let mut candidates = Vec::with_capacity(3);
    if let Ok(path) = std::env::var("TF_CUDA_LIB_PATH") {
        candidates.push(path);
    }
    candidates.push(PRIMARY_CUDA_LIB.to_string());
    candidates.push(FALLBACK_CUDA_LIB.to_string());
    candidates
}

pub unsafe fn culib() -> &'static Lib {
    static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
    LIB.get_or_init(|| {
        let mut last_err: Option<libloading::Error> = None;
        for candidate in candidate_cuda_libs() {
            tracing::info!("Loading CUDA library from {}", candidate);
            match Library::new(&candidate) {
                Ok(lib) => {
                    return Lib::from_library(lib).expect("should convert library to Lib");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to load {}", candidate);
                    last_err = Some(e);
                }
            }
        }

        panic!("can not load libcuda, error: {last_err:?}")
    })
}

/// Find libcuda.so using ldconfig
fn _find_libcuda_with_ldconfig() -> Option<String> {
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

/// Trait for CUDA Checkpoint/Restore API
pub trait CheckpointApi: Send + Sync {
    fn checkpoint(&self) -> CUresult;
    fn restore(&self) -> CUresult;
    fn lock(&self) -> CUresult;
    fn unlock(&self) -> CUresult;
    fn is_supported(&self) -> bool;
}

/// Real implementation using dynamically loaded CUDA library
pub struct CudaCheckpointApi {
    // Store the library to keep it alive
    _lib: &'static Library,
    checkpoint_fn: Option<Symbol<'static, unsafe extern "C" fn() -> CUresult>>,
    restore_fn: Option<Symbol<'static, unsafe extern "C" fn() -> CUresult>>,
    lock_fn: Option<Symbol<'static, unsafe extern "C" fn() -> CUresult>>,
    unlock_fn: Option<Symbol<'static, unsafe extern "C" fn() -> CUresult>>,
}

impl CudaCheckpointApi {
    #[allow(dead_code)]
    unsafe fn new() -> Self {
        // Get the raw library pointer
        static RAW_LIB: OnceLock<Library> = OnceLock::new();
        let lib = RAW_LIB.get_or_init(|| {
            for candidate in candidate_cuda_libs() {
                if let Ok(lib) = Library::new(&candidate) {
                    return lib;
                }
            }
            panic!("Failed to load CUDA library for checkpoint API");
        });

        // Try to load each function
        let checkpoint_fn = lib
            .get::<unsafe extern "C" fn() -> CUresult>(b"cuCheckpointProcessCheckpoint\0")
            .ok()
            .map(|sym| core::mem::transmute(sym));

        let restore_fn = lib
            .get::<unsafe extern "C" fn() -> CUresult>(b"cuCheckpointProcessRestore\0")
            .ok()
            .map(|sym| core::mem::transmute(sym));

        let lock_fn = lib
            .get::<unsafe extern "C" fn() -> CUresult>(b"cuCheckpointProcessLock\0")
            .ok()
            .map(|sym| core::mem::transmute(sym));

        let unlock_fn = lib
            .get::<unsafe extern "C" fn() -> CUresult>(b"cuCheckpointProcessUnlock\0")
            .ok()
            .map(|sym| core::mem::transmute(sym));

        if checkpoint_fn.is_some() {
            tracing::info!("CUDA Checkpoint API: Available");
        } else {
            tracing::info!("CUDA Checkpoint API: Not available");
        }
        Self {
            _lib: lib,
            checkpoint_fn,
            restore_fn,
            lock_fn,
            unlock_fn,
        }
    }
}

impl CheckpointApi for CudaCheckpointApi {
    fn checkpoint(&self) -> CUresult {
        match &self.checkpoint_fn {
            Some(f) => unsafe { f() },
            None => CUresult::CUDA_ERROR_NOT_SUPPORTED,
        }
    }

    fn restore(&self) -> CUresult {
        match &self.restore_fn {
            Some(f) => unsafe { f() },
            None => CUresult::CUDA_ERROR_NOT_SUPPORTED,
        }
    }

    fn lock(&self) -> CUresult {
        match &self.lock_fn {
            Some(f) => unsafe { f() },
            None => CUresult::CUDA_ERROR_NOT_SUPPORTED,
        }
    }

    fn unlock(&self) -> CUresult {
        match &self.unlock_fn {
            Some(f) => unsafe { f() },
            None => CUresult::CUDA_ERROR_NOT_SUPPORTED,
        }
    }

    fn is_supported(&self) -> bool {
        self.checkpoint_fn.is_some()
            && self.restore_fn.is_some()
            && self.lock_fn.is_some()
            && self.unlock_fn.is_some()
    }
}

#[cfg(test)]
pub struct MockCheckpointApi {
    checkpoint_count: AtomicU32,
    restore_count: AtomicU32,
    lock_count: AtomicU32,
    unlock_count: AtomicU32,
    should_fail: AtomicBool,
}

#[cfg(test)]
impl MockCheckpointApi {
    const fn new() -> Self {
        Self {
            checkpoint_count: AtomicU32::new(0),
            restore_count: AtomicU32::new(0),
            lock_count: AtomicU32::new(0),
            unlock_count: AtomicU32::new(0),
            should_fail: AtomicBool::new(false),
        }
    }

    pub fn reset_counters(&self) {
        self.checkpoint_count.store(0, Ordering::SeqCst);
        self.restore_count.store(0, Ordering::SeqCst);
        self.lock_count.store(0, Ordering::SeqCst);
        self.unlock_count.store(0, Ordering::SeqCst);
        self.should_fail.store(false, Ordering::SeqCst);
    }

    pub fn checkpoint_call_count(&self) -> u32 {
        self.checkpoint_count.load(Ordering::SeqCst)
    }

    pub fn restore_call_count(&self) -> u32 {
        self.restore_count.load(Ordering::SeqCst)
    }

    pub fn set_should_fail(&self, should_fail: bool) {
        self.should_fail.store(should_fail, Ordering::SeqCst);
    }
}

#[cfg(test)]
impl CheckpointApi for MockCheckpointApi {
    fn checkpoint(&self) -> CUresult {
        self.checkpoint_count.fetch_add(1, Ordering::SeqCst);
        if self.should_fail.load(Ordering::SeqCst) {
            CUresult::CUDA_ERROR_UNKNOWN
        } else {
            CUresult::CUDA_SUCCESS
        }
    }

    fn restore(&self) -> CUresult {
        self.restore_count.fetch_add(1, Ordering::SeqCst);
        CUresult::CUDA_SUCCESS
    }

    fn lock(&self) -> CUresult {
        self.lock_count.fetch_add(1, Ordering::SeqCst);
        CUresult::CUDA_SUCCESS
    }

    fn unlock(&self) -> CUresult {
        self.unlock_count.fetch_add(1, Ordering::SeqCst);
        CUresult::CUDA_SUCCESS
    }

    fn is_supported(&self) -> bool {
        true
    }
}

#[cfg(test)]
static TEST_CHECKPOINT_API: MockCheckpointApi = MockCheckpointApi::new();

/// Get the global checkpoint API instance
pub fn checkpoint_api() -> &'static dyn CheckpointApi {
    #[cfg(not(test))]
    {
        static API: OnceLock<CudaCheckpointApi> = OnceLock::new();
        API.get_or_init(|| unsafe { CudaCheckpointApi::new() })
    }

    #[cfg(test)]
    {
        &TEST_CHECKPOINT_API
    }
}

#[cfg(test)]
pub fn mock_checkpoint_api() -> &'static MockCheckpointApi {
    &TEST_CHECKPOINT_API
}
