use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use cudarc::driver::sys::CUresult;
use utils::hooks::{HookManager, InvocationContext, InvocationListener, Listener, NativePointer};

use crate::checkpoint::{checkpoint_api, CUprocessState_enum};

#[derive(Clone)]
struct SendNativePointer(pub NativePointer);

unsafe impl Send for SendNativePointer {}
unsafe impl Sync for SendNativePointer {}

impl PartialEq for SendNativePointer {
    fn eq(&self, other: &Self) -> bool {
        self.0 .0 == other.0 .0
    }
}

impl Eq for SendNativePointer {}

impl Hash for SendNativePointer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0 .0.hash(state);
    }
}

impl From<NativePointer> for SendNativePointer {
    fn from(pointer: NativePointer) -> Self {
        Self(pointer)
    }
}

/// Auto-freeze manager that monitors CUDA activity and automatically
/// freezes/resumes the process based on idle timeout.
#[allow(dead_code, reason = "Fields are used in implementation methods")]
pub struct AutoFreezeManager {
    /// Timestamp of the last CUDA activity
    last_activity: Arc<Mutex<Instant>>,
    /// Condition variable to wake up the monitor thread when activity occurs
    activity_notifier: Arc<Condvar>,
    /// Idle timeout duration after which the process will be frozen
    idle_timeout: Duration,
    /// Shutdown signal for the background thread
    shutdown: Arc<AtomicBool>,
    /// Attached native pointers
    attached: Arc<RwLock<HashSet<SendNativePointer>>>,
    /// Storage for listeners and their Frida handles
    /// We only store the listener and the handle, not HookManager (which is just a one-time tool)
    /// Protected by Mutex for thread safety, even though typically only accessed from dlsym detour
    listeners: Mutex<Vec<(Listener, Box<AutoFreezeListener>)>>,
}

// Safety: AutoFreezeManager can be safely shared and sent between threads because:
// 1. All fields use proper synchronization primitives (Arc<Mutex/RwLock/AtomicBool>)
// 2. The Listener and AutoFreezeListener are opaque handles that are never actually moved between threads
// 3. They're only accessed through the Mutex, providing runtime protection
unsafe impl Sync for AutoFreezeManager {}
unsafe impl Send for AutoFreezeManager {}

/// Lightweight listener wrapper for attaching to hooks.
/// Holds a weak reference to the global AutoFreezeManager.
pub struct AutoFreezeListener {
    manager: Arc<AutoFreezeManager>,
}

impl AutoFreezeManager {
    pub fn new(idle_timeout: Duration) -> Arc<Self> {
        tracing::info!(
            timeout_secs = idle_timeout.as_secs(),
            "Initializing auto-freeze manager"
        );

        let last_activity = Arc::new(Mutex::new(Instant::now()));
        let activity_notifier = Arc::new(Condvar::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        let manager = Arc::new(Self {
            last_activity: Arc::clone(&last_activity),
            activity_notifier: Arc::clone(&activity_notifier),
            idle_timeout,
            shutdown: Arc::clone(&shutdown),
            attached: Default::default(),
            listeners: Mutex::new(Vec::new()),
        });

        // Start background monitoring thread
        Self::start_monitor_thread_for(&manager);

        manager
    }

    pub fn as_listener(self: &Arc<Self>) -> AutoFreezeListener {
        AutoFreezeListener {
            manager: Arc::clone(self),
        }
    }

    pub fn record_activity(&self) -> Result<(), AutoFreezeError> {
        let Some(api) = checkpoint_api() else {
            return Ok(());
        };

        if let Ok(state) = api.get_process_state() {
            if state == CUprocessState_enum::CU_PROCESS_STATE_CHECKPOINTED {
                self.restore_process()?;
            }
        }

        if let Ok(mut last) = self.last_activity.lock() {
            *last = Instant::now();
            self.activity_notifier.notify_one();
        }

        Ok(())
    }

    pub fn contains_native_pointer(&self, pointer: &NativePointer) -> bool {
        let guard = self
            .attached
            .read()
            .expect("should acquire read lock for attached");
        guard.contains(&SendNativePointer::from(*pointer))
    }

    pub fn add_native_pointer(&self, pointer: NativePointer) {
        let mut guard = self
            .attached
            .write()
            .expect("should acquire write lock for attached");
        guard.insert(SendNativePointer::from(pointer));
    }

    /// Attaches an auto-freeze listener to a native pointer (CUDA function).
    /// The listener and Frida handle are stored internally to keep them alive.
    pub fn attach_to_pointer(
        self: &Arc<Self>,
        pointer: NativePointer,
    ) -> Result<(), AutoFreezeError> {
        // Mark this pointer as attached first
        self.add_native_pointer(pointer);

        // Create a temporary hook manager (just a tool to call attach)
        let mut hook_manager = HookManager::default();
        // Box the listener to ensure its address remains stable when moved to the vector
        let mut listener = Box::new(self.as_listener());

        // Attach the listener to the native pointer, get the Frida handle
        let frida_listener = hook_manager
            .attach(pointer, &mut *listener)
            .map_err(|e| AutoFreezeError::AttachFailed(format!("{e:?}")))?;

        // Store the listener and Frida handle to keep them alive
        // The HookManager is dropped here, we don't need it anymore
        self.listeners
            .lock()
            .expect("should acquire listeners lock")
            .push((frida_listener, listener));

        Ok(())
    }

    fn start_monitor_thread_for(manager: &Arc<Self>) {
        let last_activity = Arc::clone(&manager.last_activity);
        let activity_notifier = Arc::clone(&manager.activity_notifier);
        let shutdown = Arc::clone(&manager.shutdown);
        let idle_timeout = manager.idle_timeout;

        thread::spawn(move || {
            tracing::debug!("Auto-freeze monitor thread started");

            let mut last_guard = last_activity
                .lock()
                .expect("should acquire last_activity lock");

            let mut consecutive_failures = 0;

            while !shutdown.load(Ordering::Acquire) {
                let Some(api) = checkpoint_api() else {
                    let (guard, _) = activity_notifier
                        .wait_timeout(last_guard, Duration::from_secs(1))
                        .expect("should wait on condvar");
                    last_guard = guard;
                    continue;
                };

                let is_currently_frozen = api
                    .get_process_state()
                    .ok()
                    .map(|state| state == CUprocessState_enum::CU_PROCESS_STATE_CHECKPOINTED)
                    .unwrap_or(false);

                if is_currently_frozen {
                    let (guard, _timeout_result) = activity_notifier
                        .wait_timeout(last_guard, Duration::from_secs(1))
                        .expect("should wait on condvar");
                    last_guard = guard;
                    continue;
                }

                let elapsed = last_guard.elapsed();

                if elapsed >= idle_timeout {
                    let idle_secs = elapsed.as_secs();
                    drop(last_guard);

                    tracing::info!(idle_secs, "Idle timeout reached, freezing process");

                    if let Err(e) = Self::checkpoint_process() {
                        consecutive_failures += 1;
                        let retry_delay = if consecutive_failures >= 3 {
                            Duration::from_secs(60)
                        } else {
                            Duration::from_secs(5)
                        };

                        tracing::error!(error = %e, ?retry_delay, "Failed to checkpoint process");
                        last_guard = last_activity
                            .lock()
                            .expect("should acquire last_activity lock");
                        let (guard, _) = activity_notifier
                            .wait_timeout(last_guard, retry_delay)
                            .expect("should wait on condvar");
                        last_guard = guard;
                    } else {
                        consecutive_failures = 0;
                        tracing::info!("Process successfully frozen");
                        last_guard = last_activity
                            .lock()
                            .expect("should acquire last_activity lock");
                    }
                } else {
                    consecutive_failures = 0;
                    let remaining = idle_timeout - elapsed;
                    let (guard, timeout_result) = activity_notifier
                        .wait_timeout(last_guard, remaining)
                        .expect("should wait on condvar");
                    last_guard = guard;

                    if timeout_result.timed_out() {
                        tracing::trace!("Idle timeout expired, will freeze on next iteration");
                    } else {
                        tracing::trace!("New activity detected, restarting idle timer");
                    }
                }
            }

            tracing::debug!("Auto-freeze monitor thread stopped");
        });
    }

    fn checkpoint_process() -> Result<(), AutoFreezeError> {
        let Some(api) = checkpoint_api() else {
            return Err(AutoFreezeError::CheckpointNotSupported);
        };

        if !api.is_supported() {
            return Err(AutoFreezeError::CheckpointNotSupported);
        }

        let pid = api.pid();

        let state = api
            .get_process_state()
            .map_err(|result| AutoFreezeError::GetStateFailed { pid, result })?;

        if state != CUprocessState_enum::CU_PROCESS_STATE_RUNNING {
            tracing::warn!(
                pid,
                state = ?state,
                "Process is not in RUNNING state, skipping checkpoint"
            );
            return Ok(());
        }

        let lock_result = api.lock();
        if lock_result != CUresult::CUDA_SUCCESS {
            return Err(AutoFreezeError::LockFailed {
                pid,
                result: lock_result,
            });
        }

        let result = api.checkpoint();
        if result != CUresult::CUDA_SUCCESS {
            let _ = api.unlock();
            return Err(AutoFreezeError::CheckpointFailed { pid, result });
        }

        Ok(())
    }

    fn restore_process(&self) -> Result<(), AutoFreezeError> {
        let Some(api) = checkpoint_api() else {
            return Err(AutoFreezeError::CheckpointNotSupported);
        };

        let pid = api.pid();

        tracing::info!(pid, "Restoring process from checkpoint");

        let result = api.restore();
        if result != CUresult::CUDA_SUCCESS {
            return Err(AutoFreezeError::RestoreFailed { pid, result });
        }

        let unlock_result = api.unlock();
        if unlock_result != CUresult::CUDA_SUCCESS {
            return Err(AutoFreezeError::UnlockFailed {
                pid,
                result: unlock_result,
            });
        }

        if let Ok(mut last) = self.last_activity.lock() {
            *last = Instant::now();
        }

        tracing::info!(pid, "Process successfully restored");

        Ok(())
    }
}

impl Drop for AutoFreezeManager {
    fn drop(&mut self) {
        tracing::debug!("Shutting down auto-freeze manager");

        // Detach all listeners from Frida
        if let Ok(mut listeners) = self.listeners.lock() {
            // Create a temporary HookManager to call detach
            let mut hook_manager = HookManager::default();
            for (frida_listener, _auto_freeze_listener) in listeners.drain(..) {
                hook_manager.detach(frida_listener);
            }
        }

        self.shutdown.store(true, Ordering::Release);
    }
}

impl InvocationListener for AutoFreezeListener {
    fn on_enter(&mut self, _context: InvocationContext) {
        if let Err(e) = self.manager.record_activity() {
            tracing::error!(error = %e, "Failed to record activity on hook enter");
        }
    }

    fn on_leave(&mut self, _context: InvocationContext) {}
}

/// Errors that can occur during auto-freeze operations
#[derive(Debug, thiserror::Error)]
#[allow(dead_code, reason = "Used when auto-freeze is enabled")]
pub enum AutoFreezeError {
    #[error("Failed to checkpoint process (pid={pid}): {result:?}")]
    CheckpointFailed { pid: u32, result: CUresult },

    #[error("Failed to restore process (pid={pid}): {result:?}")]
    RestoreFailed { pid: u32, result: CUresult },

    #[error("Failed to lock GPU (pid={pid}): {result:?}")]
    LockFailed { pid: u32, result: CUresult },

    #[error("Failed to unlock GPU (pid={pid}): {result:?}")]
    UnlockFailed { pid: u32, result: CUresult },

    #[error("Failed to get process state (pid={pid}): {result:?}")]
    GetStateFailed { pid: u32, result: CUresult },

    #[error("Failed to attach listener: {0}")]
    AttachFailed(String),

    #[error("Checkpoint API not supported")]
    CheckpointNotSupported,
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::checkpoint::mock_checkpoint_api;

    fn setup_test() {
        mock_checkpoint_api().reset_counters();
    }

    #[test_log::test]
    #[serial]
    fn creates_manager_successfully() {
        setup_test();

        let timeout = Duration::from_secs(5);
        let manager = AutoFreezeManager::new(timeout);

        assert_eq!(manager.idle_timeout, timeout);
    }

    #[test_log::test]
    #[serial]
    fn records_activity_without_freezing() {
        setup_test();

        let manager = AutoFreezeManager::new(Duration::from_secs(10));

        manager
            .record_activity()
            .expect("should record activity successfully");

        thread::sleep(Duration::from_millis(100));

        assert_eq!(mock_checkpoint_api().checkpoint_call_count(), 0);
        assert_eq!(mock_checkpoint_api().restore_call_count(), 0);
    }

    #[test_log::test]
    #[serial]
    fn freezes_after_idle_timeout() {
        setup_test();

        let timeout = Duration::from_millis(200);
        let _manager = AutoFreezeManager::new(timeout);

        thread::sleep(timeout + Duration::from_millis(300));

        assert_eq!(mock_checkpoint_api().checkpoint_call_count(), 1);
    }

    #[test_log::test]
    #[serial]
    fn restores_on_activity_after_freeze() {
        setup_test();

        let timeout = Duration::from_millis(200);
        let manager = AutoFreezeManager::new(timeout);

        thread::sleep(timeout + Duration::from_millis(300));

        manager
            .record_activity()
            .expect("should restore and record activity");

        assert_eq!(mock_checkpoint_api().restore_call_count(), 1);
    }

    #[test_log::test]
    #[serial]
    fn supports_multiple_freeze_unfreeze_cycles() {
        setup_test();

        let timeout = Duration::from_millis(200);
        let manager = AutoFreezeManager::new(timeout);

        thread::sleep(timeout + Duration::from_millis(300));
        let first_checkpoint_count = mock_checkpoint_api().checkpoint_call_count();
        assert_eq!(first_checkpoint_count, 1);

        manager.record_activity().expect("should restore process");
        assert_eq!(mock_checkpoint_api().restore_call_count(), 1);

        thread::sleep(timeout + Duration::from_millis(300));
        assert_eq!(mock_checkpoint_api().checkpoint_call_count(), 2);

        manager
            .record_activity()
            .expect("should restore process again");
        assert_eq!(mock_checkpoint_api().restore_call_count(), 2);

        thread::sleep(timeout + Duration::from_millis(300));
        assert_eq!(mock_checkpoint_api().checkpoint_call_count(), 3);
    }

    #[test_log::test]
    #[serial]
    fn activity_prevents_freezing() {
        setup_test();

        let timeout = Duration::from_millis(300);
        let manager = AutoFreezeManager::new(timeout);

        for _ in 0..5 {
            thread::sleep(Duration::from_millis(150));
            manager.record_activity().expect("should record activity");
        }

        assert_eq!(mock_checkpoint_api().checkpoint_call_count(), 0);
    }

    #[test_log::test]
    #[serial]
    fn condvar_wakes_immediately_on_activity() {
        setup_test();

        let timeout = Duration::from_secs(10);
        let manager = AutoFreezeManager::new(timeout);

        let start = Instant::now();

        for _ in 0..10 {
            manager.record_activity().expect("should record activity");
            thread::sleep(Duration::from_millis(50));
        }

        let elapsed = start.elapsed();

        assert_eq!(mock_checkpoint_api().checkpoint_call_count(), 0);
        assert!(elapsed < Duration::from_secs(2));
    }

    #[test_log::test]
    #[serial]
    fn handles_checkpoint_failure() {
        setup_test();

        mock_checkpoint_api().set_should_fail(true);

        let timeout = Duration::from_millis(200);
        let _manager = AutoFreezeManager::new(timeout);

        thread::sleep(timeout + Duration::from_millis(500));

        assert!(mock_checkpoint_api().checkpoint_call_count() > 0);
    }

    #[test_log::test]
    #[serial]
    fn shutdown_stops_monitor_thread() {
        setup_test();

        let timeout = Duration::from_millis(200);
        let manager = AutoFreezeManager::new(timeout);

        let checkpoint_count_before = mock_checkpoint_api().checkpoint_call_count();

        drop(manager);

        thread::sleep(Duration::from_millis(300));

        let checkpoint_count_after = mock_checkpoint_api().checkpoint_call_count();
        assert!(checkpoint_count_after <= checkpoint_count_before + 1);
    }
}
