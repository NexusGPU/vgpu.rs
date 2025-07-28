use std::cell::UnsafeCell;
use std::hint::spin_loop;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::yield_now;

#[repr(C)]
pub struct ShmMutex<T> {
    lock: AtomicUsize,
    data: UnsafeCell<T>,
    pid: usize,
}

unsafe impl<T: Send> Send for ShmMutex<T> {}
unsafe impl<T: Send> Sync for ShmMutex<T> {}

pub struct ShmMutexGuard<'a, T> {
    mutex: &'a ShmMutex<T>,
}

impl<'a, T> Drop for ShmMutexGuard<'a, T> {
    fn drop(&mut self) {
        self.mutex.unlock();
    }
}

impl<'a, T> Deref for ShmMutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T> DerefMut for ShmMutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mutex.data.get() }
    }
}

impl<T> ShmMutex<T> {
    pub fn new(data: T) -> Self {
        let pid = std::process::id() as usize;
        ShmMutex {
            lock: AtomicUsize::new(0),
            data: UnsafeCell::new(data),
            pid,
        }
    }

    /// Checks if a process with the given PID exists
    fn is_process_alive(pid: usize) -> bool {
        if pid == 0 {
            return false;
        }

        // On Unix systems, we can use kill(pid, 0) to check if process exists
        #[cfg(unix)]
        {
            unsafe { libc::kill(pid as i32, 0) == 0 }
        }

        #[cfg(not(unix))]
        {
            // Fallback for non-Unix systems - always assume alive to be safe
            true
        }
    }

    /// Attempts to clean up the lock if the holding process is dead
    fn try_cleanup_dead_lock(&self) -> bool {
        let current_holder = self.lock.load(Ordering::Acquire);
        if current_holder != 0 && !Self::is_process_alive(current_holder) {
            // The process holding the lock is dead, try to clean it up
            match self
                .lock
                .compare_exchange(current_holder, 0, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => {
                    tracing::warn!(
                        "Cleaned up orphaned lock held by dead process {}",
                        current_holder
                    );
                    true
                }
                Err(_) => {
                    // Someone else changed the lock state, retry
                    false
                }
            }
        } else {
            false
        }
    }

    /// Forcefully cleans up any orphaned locks during startup
    pub fn cleanup_orphaned_lock(&self) {
        let current_holder = self.lock.load(Ordering::Acquire);
        if current_holder != 0 && !Self::is_process_alive(current_holder) {
            self.lock.store(0, Ordering::Release);
            tracing::info!(
                "Cleaned up orphaned lock held by dead process {} during startup",
                current_holder
            );
        }
    }

    pub fn lock(&self) -> ShmMutexGuard<T> {
        loop {
            // First try to acquire the lock normally
            if self
                .lock
                .compare_exchange(0, self.pid, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }

            // If that fails, check if we can clean up a dead lock holder
            if self.try_cleanup_dead_lock() {
                // Successfully cleaned up, try to acquire again
                continue;
            }

            // Normal retry loop with backoff
            for _ in 0..100 {
                spin_loop();
            }
            yield_now();
        }

        ShmMutexGuard { mutex: self }
    }

    fn unlock(&self) {
        let result = self
            .lock
            .compare_exchange(self.pid, 0, Ordering::Release, Ordering::Relaxed);

        if result.is_err() {
            let current = self.lock.load(Ordering::Relaxed);
            if current != self.pid {
                tracing::warn!(
                    "process {} tried to unlock mutex owned by {}",
                    self.pid,
                    current
                );
            }
        }
    }

    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn test_basic_lock_unlock() {
        let mutex = ShmMutex::new(42);

        {
            let guard = mutex.lock();
            assert_eq!(*guard, 42);
        }

        let guard = mutex.lock();
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_mutable_access() {
        let mutex = ShmMutex::new(0);

        {
            let mut guard = mutex.lock();
            *guard = 100;
        }

        let guard = mutex.lock();
        assert_eq!(*guard, 100);
    }

    #[test]
    fn test_concurrent_access() {
        let mutex = Arc::new(ShmMutex::new(0));
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = vec![];

        for _ in 0..4 {
            let mutex_clone = Arc::clone(&mutex);
            let barrier_clone = Arc::clone(&barrier);

            let handle = thread::spawn(move || {
                barrier_clone.wait();

                for _ in 0..100 {
                    let mut guard = mutex_clone.lock();
                    *guard += 1;
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let guard = mutex.lock();
        assert_eq!(*guard, 400);
    }

    #[test]
    fn test_guard_drop_unlocks() {
        let mutex = Arc::new(ShmMutex::new(String::from("test")));
        let mutex_clone = Arc::clone(&mutex);

        let handle = thread::spawn(move || {
            let _guard = mutex_clone.lock();
            thread::sleep(Duration::from_millis(100));
        });

        thread::sleep(Duration::from_millis(50));

        let start = Instant::now();
        let _guard = mutex.lock();
        let elapsed = start.elapsed();

        handle.join().unwrap();
        assert!(elapsed >= Duration::from_millis(40));
    }

    #[test]
    fn test_different_data_types() {
        let int_mutex = ShmMutex::new(42i32);
        let string_mutex = ShmMutex::new(String::from("hello"));
        let vec_mutex = ShmMutex::new(vec![1, 2, 3]);

        {
            let guard = int_mutex.lock();
            assert_eq!(*guard, 42);
        }

        {
            let mut guard = string_mutex.lock();
            guard.push_str(" world");
            assert_eq!(*guard, "hello world");
        }

        {
            let mut guard = vec_mutex.lock();
            guard.push(4);
            assert_eq!(*guard, vec![1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_into_inner() {
        let mutex = ShmMutex::new(vec![1, 2, 3]);
        let data = mutex.into_inner();
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn test_get_mut() {
        let mut mutex = ShmMutex::new(42);
        {
            let data = mutex.get_mut();
            *data = 100;
        }

        let guard = mutex.lock();
        assert_eq!(*guard, 100);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<ShmMutex<i32>>();
        assert_sync::<ShmMutex<i32>>();
    }

    #[test]
    fn test_panic_during_lock_releases() {
        let mutex = Arc::new(ShmMutex::new(0));
        let mutex_clone = Arc::clone(&mutex);

        let handle = thread::spawn(move || {
            let _guard = mutex_clone.lock();
            panic!("intentional panic");
        });

        // The thread should panic
        let result = handle.join();
        assert!(result.is_err());

        // After panic, we should still be able to acquire the lock
        let guard = mutex.lock();
        assert_eq!(*guard, 0);
    }
}
