use core::ffi;

use cudarc::driver::sys::CUresult;
use libloading::{Library, Symbol};

#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

#[cfg(not(test))]
use std::sync::OnceLock;

#[allow(
    non_camel_case_types,
    non_snake_case,
    dead_code,
    reason = "FFI types must match CUDA API naming conventions"
)]
mod ffi_types {
    use super::*;

    pub(super) type Cuuint64T = u64;

    impl CUprocessState_enum {
        pub const CU_PROCESS_STATE_RUNNING: CUprocessState_enum = CUprocessState_enum(0);
        pub const CU_PROCESS_STATE_LOCKED: CUprocessState_enum = CUprocessState_enum(1);
        pub const CU_PROCESS_STATE_CHECKPOINTED: CUprocessState_enum = CUprocessState_enum(2);
        pub const CU_PROCESS_STATE_FAILED: CUprocessState_enum = CUprocessState_enum(3);
    }

    #[repr(transparent)]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub struct CUprocessState_enum(pub ffi::c_uint);

    pub use self::CUprocessState_enum as CUprocessState;

    #[repr(C)]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub(super) struct CUcheckpointLockArgs_st {
        pub timeoutMs: ffi::c_uint,
        pub reserved0: ffi::c_uint,
        pub reserved1: [Cuuint64T; 7usize],
    }

    pub(super) type CUcheckpointLockArgs = CUcheckpointLockArgs_st;

    #[repr(C)]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub(super) struct CUcheckpointCheckpointArgs_st {
        pub reserved: [Cuuint64T; 8usize],
    }

    pub(super) type CUcheckpointCheckpointArgs = CUcheckpointCheckpointArgs_st;

    #[repr(C)]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub(super) struct CUcheckpointGpuPair_st {
        pub oldUuid: cudarc::driver::sys::CUuuid,
        pub newUuid: cudarc::driver::sys::CUuuid,
    }

    pub(super) type CUcheckpointGpuPair = CUcheckpointGpuPair_st;

    #[repr(C)]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub(super) struct CUcheckpointRestoreArgs_st {
        pub gpuPairs: *mut CUcheckpointGpuPair,
        pub gpuPairsCount: ffi::c_uint,
        pub reserved: [ffi::c_char; 44usize],
        pub reserved1: Cuuint64T,
    }

    pub(super) type CUcheckpointRestoreArgs = CUcheckpointRestoreArgs_st;

    #[repr(C)]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub(super) struct CUcheckpointUnlockArgs_st {
        pub reserved: [Cuuint64T; 8usize],
    }

    pub(super) type CUcheckpointUnlockArgs = CUcheckpointUnlockArgs_st;
}

pub use ffi_types::CUprocessState_enum;

use ffi_types::*;

#[cfg(not(test))]
fn candidate_cuda_libs() -> Vec<String> {
    let mut candidates = Vec::with_capacity(3);
    if let Ok(path) = std::env::var("TF_CUDA_LIB_PATH") {
        candidates.push(path);
    }
    candidates.push("libcuda.so.1".to_string());
    candidates.push("libcuda.so".to_string());
    candidates
}

pub trait CheckpointApi: Send + Sync {
    fn checkpoint(&self) -> CUresult;
    fn restore(&self) -> CUresult;
    fn lock(&self) -> CUresult;
    fn unlock(&self) -> CUresult;
    fn is_supported(&self) -> bool;
    fn get_process_state(&self) -> Result<CUprocessState, CUresult>;
    fn pid(&self) -> u32;
}

pub struct CudaCheckpointApi {
    _lib: &'static Library,
    pid: ffi::c_int,
    process_state_fn: Option<
        Symbol<
            'static,
            unsafe extern "C" fn(pid: ffi::c_int, state: *mut CUprocessState) -> CUresult,
        >,
    >,
    checkpoint_fn: Option<
        Symbol<
            'static,
            unsafe extern "C" fn(
                pid: ffi::c_int,
                args: *mut CUcheckpointCheckpointArgs,
            ) -> CUresult,
        >,
    >,
    restore_fn: Option<
        Symbol<
            'static,
            unsafe extern "C" fn(pid: ffi::c_int, args: *mut CUcheckpointRestoreArgs) -> CUresult,
        >,
    >,
    lock_fn: Option<
        Symbol<
            'static,
            unsafe extern "C" fn(pid: ffi::c_int, args: *mut CUcheckpointLockArgs) -> CUresult,
        >,
    >,
    unlock_fn: Option<
        Symbol<
            'static,
            unsafe extern "C" fn(pid: ffi::c_int, args: *mut CUcheckpointUnlockArgs) -> CUresult,
        >,
    >,
}

impl CudaCheckpointApi {
    #[cfg(not(test))]
    unsafe fn new(pid: u32) -> Self {
        static RAW_LIB: OnceLock<Library> = OnceLock::new();
        let lib = RAW_LIB.get_or_init(|| {
            for candidate in candidate_cuda_libs() {
                if let Ok(lib) = Library::new(&candidate) {
                    return lib;
                }
            }
            panic!("Failed to load CUDA library for checkpoint API");
        });

        let process_state_fn = lib
            .get::<unsafe extern "C" fn() -> CUresult>(b"cuCheckpointProcessGetState\0")
            .ok()
            .map(|sym| core::mem::transmute(sym));

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
            tracing::info!(pid = pid, "CUDA Checkpoint API: Available");
        } else {
            tracing::info!("CUDA Checkpoint API: Not available");
        }

        Self {
            _lib: lib,
            pid: pid as ffi::c_int,
            process_state_fn,
            checkpoint_fn,
            restore_fn,
            lock_fn,
            unlock_fn,
        }
    }
}

impl CheckpointApi for CudaCheckpointApi {
    fn checkpoint(&self) -> CUresult {
        self.checkpoint_fn
            .as_ref()
            .map_or(CUresult::CUDA_ERROR_NOT_SUPPORTED, |f| unsafe {
                f(self.pid, core::ptr::null_mut())
            })
    }

    fn restore(&self) -> CUresult {
        self.restore_fn
            .as_ref()
            .map_or(CUresult::CUDA_ERROR_NOT_SUPPORTED, |f| unsafe {
                f(self.pid, core::ptr::null_mut())
            })
    }

    fn lock(&self) -> CUresult {
        self.lock_fn
            .as_ref()
            .map_or(CUresult::CUDA_ERROR_NOT_SUPPORTED, |f| unsafe {
                f(self.pid, core::ptr::null_mut())
            })
    }

    fn unlock(&self) -> CUresult {
        self.unlock_fn
            .as_ref()
            .map_or(CUresult::CUDA_ERROR_NOT_SUPPORTED, |f| unsafe {
                f(self.pid, core::ptr::null_mut())
            })
    }

    fn is_supported(&self) -> bool {
        self.checkpoint_fn.is_some()
            && self.restore_fn.is_some()
            && self.lock_fn.is_some()
            && self.unlock_fn.is_some()
    }

    fn get_process_state(&self) -> Result<CUprocessState, CUresult> {
        match &self.process_state_fn {
            Some(f) => {
                let mut state = CUprocessState_enum(0);
                let result = unsafe { f(self.pid, &mut state) };
                if result == CUresult::CUDA_SUCCESS {
                    Ok(state)
                } else {
                    Err(result)
                }
            }
            None => Err(CUresult::CUDA_ERROR_NOT_SUPPORTED),
        }
    }

    fn pid(&self) -> u32 {
        self.pid as u32
    }
}

#[cfg(test)]
pub struct MockCheckpointApi {
    checkpoint_count: AtomicU32,
    restore_count: AtomicU32,
    lock_count: AtomicU32,
    unlock_count: AtomicU32,
    should_fail: AtomicBool,
    state: std::sync::RwLock<CUprocessState>,
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
            state: std::sync::RwLock::new(CUprocessState_enum::CU_PROCESS_STATE_RUNNING),
        }
    }

    pub fn reset_counters(&self) {
        self.checkpoint_count.store(0, Ordering::SeqCst);
        self.restore_count.store(0, Ordering::SeqCst);
        self.lock_count.store(0, Ordering::SeqCst);
        self.unlock_count.store(0, Ordering::SeqCst);
        self.should_fail.store(false, Ordering::SeqCst);
        *self.state.write().expect("should acquire write lock") =
            CUprocessState_enum::CU_PROCESS_STATE_RUNNING;
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
            *self.state.write().expect("should acquire write lock") =
                CUprocessState_enum::CU_PROCESS_STATE_CHECKPOINTED;
            CUresult::CUDA_SUCCESS
        }
    }

    fn restore(&self) -> CUresult {
        self.restore_count.fetch_add(1, Ordering::SeqCst);
        *self.state.write().expect("should acquire write lock") =
            CUprocessState_enum::CU_PROCESS_STATE_RUNNING;
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

    fn get_process_state(&self) -> Result<CUprocessState, CUresult> {
        Ok(*self.state.read().expect("should acquire read lock"))
    }

    fn pid(&self) -> u32 {
        0
    }
}

#[cfg(test)]
static TEST_CHECKPOINT_API: MockCheckpointApi = MockCheckpointApi::new();

#[cfg(not(test))]
static CHECKPOINT_API: OnceLock<Box<dyn CheckpointApi>> = OnceLock::new();

pub fn init_checkpoint_api(pid: u32) {
    #[cfg(not(test))]
    {
        let api = Box::new(unsafe { CudaCheckpointApi::new(pid) });
        if CHECKPOINT_API.set(api).is_err() {
            tracing::warn!("Checkpoint API already initialized");
        }
    }

    #[cfg(test)]
    {
        let _ = pid;
    }
}

pub fn checkpoint_api() -> Option<&'static dyn CheckpointApi> {
    #[cfg(not(test))]
    {
        CHECKPOINT_API.get().map(|api| &**api)
    }

    #[cfg(test)]
    {
        Some(&TEST_CHECKPOINT_API)
    }
}

#[cfg(test)]
pub fn mock_checkpoint_api() -> &'static MockCheckpointApi {
    &TEST_CHECKPOINT_API
}
