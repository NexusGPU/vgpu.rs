use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use cudarc::driver::sys::CUresult;
use utils::hooks::{HookManager, InvocationContext, InvocationListener, Listener, NativePointer};

use crate::culib::checkpoint_api;

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
    /// Whether the process is currently frozen
    is_frozen: Arc<AtomicBool>,
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
        let is_frozen = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));

        let manager = Arc::new(Self {
            last_activity: Arc::clone(&last_activity),
            activity_notifier: Arc::clone(&activity_notifier),
            idle_timeout,
            is_frozen: Arc::clone(&is_frozen),
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
        if self.is_frozen.load(Ordering::Acquire) {
            self.restore_process()?;
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
        let is_frozen = Arc::clone(&manager.is_frozen);
        let shutdown = Arc::clone(&manager.shutdown);
        let idle_timeout = manager.idle_timeout;

        thread::spawn(move || {
            tracing::debug!("Auto-freeze monitor thread started");

            let mut last_guard = last_activity
                .lock()
                .expect("should acquire last_activity lock");

            while !shutdown.load(Ordering::Acquire) {
                let is_currently_frozen = is_frozen.load(Ordering::Acquire);

                if is_currently_frozen {
                    // If frozen, just wait for activity notification or shutdown
                    // Use a 1-second timeout to periodically check shutdown flag
                    let (guard, _timeout_result) = activity_notifier
                        .wait_timeout(last_guard, Duration::from_secs(1))
                        .expect("should wait on condvar");
                    last_guard = guard;
                    continue;
                }

                // Calculate remaining time until timeout
                let elapsed = last_guard.elapsed();

                if elapsed >= idle_timeout {
                    // Timeout reached, freeze the process
                    let idle_secs = elapsed.as_secs();
                    drop(last_guard); // Release lock before checkpoint

                    tracing::info!(idle_secs, "Idle timeout reached, freezing process");

                    if let Err(e) = Self::checkpoint_process() {
                        tracing::error!(error = %e, "Failed to checkpoint process");
                        // On error, reacquire lock and retry after a short delay
                        last_guard = last_activity
                            .lock()
                            .expect("should acquire last_activity lock");
                        let (guard, _) = activity_notifier
                            .wait_timeout(last_guard, Duration::from_secs(1))
                            .expect("should wait on condvar");
                        last_guard = guard;
                    } else {
                        is_frozen.store(true, Ordering::Release);
                        tracing::info!("Process successfully frozen");
                        // Reacquire lock to continue loop
                        last_guard = last_activity
                            .lock()
                            .expect("should acquire last_activity lock");
                    }
                } else {
                    // Wait for the remaining time or until notified of new activity
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

    /// Attempts to checkpoint (freeze) the current process.
    fn checkpoint_process() -> Result<(), AutoFreezeError> {
        let api = checkpoint_api();

        if !api.is_supported() {
            return Err(AutoFreezeError::CheckpointNotSupported);
        }

        // Lock GPU before checkpoint
        let lock_result = api.lock();
        if lock_result != CUresult::CUDA_SUCCESS {
            return Err(AutoFreezeError::LockFailed(lock_result));
        }

        let result = api.checkpoint();
        if result != CUresult::CUDA_SUCCESS {
            // If checkpoint fails, try to unlock before returning error
            let _ = api.unlock();
            return Err(AutoFreezeError::CheckpointFailed(result));
        }

        Ok(())
    }

    /// Restores the process from a checkpoint.
    fn restore_process(&self) -> Result<(), AutoFreezeError> {
        tracing::info!("Restoring process from checkpoint");

        let api = checkpoint_api();
        let result = api.restore();
        if result != CUresult::CUDA_SUCCESS {
            return Err(AutoFreezeError::RestoreFailed(result));
        }

        // Unlock GPU after restore
        let unlock_result = api.unlock();
        if unlock_result != CUresult::CUDA_SUCCESS {
            return Err(AutoFreezeError::UnlockFailed(unlock_result));
        }

        // Mark as unfrozen and reset timer
        // Note: We set is_frozen to false BEFORE calling record_activity()
        // to prevent infinite recursion, since record_activity() checks this flag
        self.is_frozen.store(false, Ordering::Release);

        // Reset the idle timer (will not trigger restore again since is_frozen is now false)
        if let Ok(mut last) = self.last_activity.lock() {
            *last = Instant::now();
        }

        tracing::info!("Process successfully restored");

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
    #[error("Failed to checkpoint process: {0:?}")]
    CheckpointFailed(CUresult),

    #[error("Failed to restore process: {0:?}")]
    RestoreFailed(CUresult),

    #[error("Failed to lock GPU: {0:?}")]
    LockFailed(CUresult),

    #[error("Failed to unlock GPU: {0:?}")]
    UnlockFailed(CUresult),

    #[error("Failed to attach listener: {0}")]
    AttachFailed(String),

    #[error("Checkpoint API not supported")]
    CheckpointNotSupported,
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::culib::mock_checkpoint_api;

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
        assert!(!manager.is_frozen.load(Ordering::Acquire));
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

        assert!(!manager.is_frozen.load(Ordering::Acquire));
        assert_eq!(mock_checkpoint_api().checkpoint_call_count(), 0);
        assert_eq!(mock_checkpoint_api().restore_call_count(), 0);
    }

    #[test_log::test]
    #[serial]
    fn freezes_after_idle_timeout() {
        setup_test();

        // Use a very short timeout for testing
        let timeout = Duration::from_millis(200);
        let manager = AutoFreezeManager::new(timeout);

        // Wait for timeout plus some buffer
        thread::sleep(timeout + Duration::from_millis(300));

        // Should have frozen by now
        assert!(manager.is_frozen.load(Ordering::Acquire));
        assert_eq!(mock_checkpoint_api().checkpoint_call_count(), 1);
    }

    #[test_log::test]
    #[serial]
    fn restores_on_activity_after_freeze() {
        setup_test();

        let timeout = Duration::from_millis(200);
        let manager = AutoFreezeManager::new(timeout);

        // Wait for freeze
        thread::sleep(timeout + Duration::from_millis(300));
        assert!(manager.is_frozen.load(Ordering::Acquire));

        // Record activity should restore
        manager
            .record_activity()
            .expect("should restore and record activity");

        assert!(!manager.is_frozen.load(Ordering::Acquire));
        assert_eq!(mock_checkpoint_api().restore_call_count(), 1);
    }

    #[test_log::test]
    #[serial]
    fn supports_multiple_freeze_unfreeze_cycles() {
        setup_test();

        let timeout = Duration::from_millis(200);
        let manager = AutoFreezeManager::new(timeout);

        // First cycle: freeze
        thread::sleep(timeout + Duration::from_millis(300));
        assert!(manager.is_frozen.load(Ordering::Acquire));
        let first_checkpoint_count = mock_checkpoint_api().checkpoint_call_count();
        assert_eq!(first_checkpoint_count, 1);

        // Unfreeze with activity
        manager.record_activity().expect("should restore process");
        assert!(!manager.is_frozen.load(Ordering::Acquire));
        assert_eq!(mock_checkpoint_api().restore_call_count(), 1);

        // Second cycle: freeze again
        thread::sleep(timeout + Duration::from_millis(300));
        assert!(manager.is_frozen.load(Ordering::Acquire));
        assert_eq!(mock_checkpoint_api().checkpoint_call_count(), 2);

        // Unfreeze again
        manager
            .record_activity()
            .expect("should restore process again");
        assert!(!manager.is_frozen.load(Ordering::Acquire));
        assert_eq!(mock_checkpoint_api().restore_call_count(), 2);

        // Third cycle to prove unlimited rounds
        thread::sleep(timeout + Duration::from_millis(300));
        assert!(manager.is_frozen.load(Ordering::Acquire));
        assert_eq!(mock_checkpoint_api().checkpoint_call_count(), 3);
    }

    #[test_log::test]
    #[serial]
    fn activity_prevents_freezing() {
        setup_test();

        let timeout = Duration::from_millis(300);
        let manager = AutoFreezeManager::new(timeout);

        // Record activity periodically to prevent freezing
        for _ in 0..5 {
            thread::sleep(Duration::from_millis(150));
            manager.record_activity().expect("should record activity");
        }

        // Should not have frozen
        assert!(!manager.is_frozen.load(Ordering::Acquire));
        assert_eq!(mock_checkpoint_api().checkpoint_call_count(), 0);
    }

    #[test_log::test]
    #[serial]
    fn condvar_wakes_immediately_on_activity() {
        setup_test();

        let timeout = Duration::from_secs(10);
        let manager = AutoFreezeManager::new(timeout);

        let start = Instant::now();

        // Record activity multiple times in quick succession
        for _ in 0..10 {
            manager.record_activity().expect("should record activity");
            thread::sleep(Duration::from_millis(50));
        }

        let elapsed = start.elapsed();

        // Should not have frozen even though we're constantly notifying
        assert!(!manager.is_frozen.load(Ordering::Acquire));
        assert_eq!(mock_checkpoint_api().checkpoint_call_count(), 0);

        // Verify the test ran quickly (no long sleep delays)
        assert!(elapsed < Duration::from_secs(2));
    }

    #[test_log::test]
    #[serial]
    fn handles_checkpoint_failure() {
        setup_test();

        mock_checkpoint_api().set_should_fail(true);

        let timeout = Duration::from_millis(200);
        let manager = AutoFreezeManager::new(timeout);

        // Wait for attempted freeze
        thread::sleep(timeout + Duration::from_millis(500));

        // Should have attempted checkpoint but remain unfrozen due to failure
        assert!(!manager.is_frozen.load(Ordering::Acquire));
        assert!(mock_checkpoint_api().checkpoint_call_count() > 0);
    }

    #[test_log::test]
    #[serial]
    fn shutdown_stops_monitor_thread() {
        setup_test();

        let timeout = Duration::from_millis(200);
        let manager = AutoFreezeManager::new(timeout);

        let checkpoint_count_before = mock_checkpoint_api().checkpoint_call_count();

        // Trigger shutdown by dropping
        drop(manager);

        // Give thread time to notice shutdown
        thread::sleep(Duration::from_millis(300));

        // No new checkpoints should occur after shutdown
        let checkpoint_count_after = mock_checkpoint_api().checkpoint_call_count();
        assert!(checkpoint_count_after <= checkpoint_count_before + 1);
    }
}
