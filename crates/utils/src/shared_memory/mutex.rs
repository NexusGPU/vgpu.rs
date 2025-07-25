use std::hint::spin_loop;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::yield_now;

#[repr(C)]
pub struct ShmSpinLock {
    lock: AtomicUsize,
}

impl Default for ShmSpinLock {
    fn default() -> Self {
        Self::new()
    }
}

impl ShmSpinLock {
    pub const fn new() -> Self {
        ShmSpinLock {
            lock: AtomicUsize::new(0),
        }
    }

    pub fn lock(&self, pid: usize) {
        while self
            .lock
            .compare_exchange(0, pid, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            for _ in 0..100 {
                spin_loop();
            }
            yield_now();
        }
    }

    pub fn unlock(&self, pid: usize) {
        let result = self
            .lock
            .compare_exchange(pid, 0, Ordering::Release, Ordering::Relaxed);

        if result.is_err() {
            let current = self.lock.load(Ordering::Relaxed);
            if current != pid {
                tracing::warn!(
                    "process {} tried to unlock spinlock owned by {}",
                    pid,
                    current
                );
            }
        }
    }
}
