//! Generic per-key async lock with automatic cleanup using weak references

use std::hash::Hash;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Weak};

use dashmap::DashMap;
use tokio::sync::{Mutex, OwnedMutexGuard};

/// Custom guard that automatically cleans up the weak reference when dropped
///
/// This guard holds an owned mutex guard and checks reference count on drop.
/// When the guard is dropped, it checks if this is the last strong reference to the mutex.
/// If so, it removes the weak reference from the DashMap, achieving immediate cleanup.
pub struct KeyedLockGuard<K>
where
    K: Hash + Eq + Clone,
{
    /// The owned mutex guard
    guard: Option<OwnedMutexGuard<()>>,
    /// Arc to the mutex for reference counting check
    arc: Arc<Mutex<()>>,
    /// Key for cleanup
    key: K,
    /// Reference to the locks map for cleanup
    locks: Arc<DashMap<K, Weak<Mutex<()>>>>,
}

impl<K> KeyedLockGuard<K>
where
    K: Hash + Eq + Clone,
{
    fn new(
        guard: OwnedMutexGuard<()>,
        arc: Arc<Mutex<()>>,
        key: K,
        locks: Arc<DashMap<K, Weak<Mutex<()>>>>,
    ) -> Self {
        Self {
            guard: Some(guard),
            arc,
            key,
            locks,
        }
    }
}

impl<K> Drop for KeyedLockGuard<K>
where
    K: Hash + Eq + Clone,
{
    fn drop(&mut self) {
        // First release the mutex guard
        drop(self.guard.take());

        // Check if we're the last holder of the Arc
        // Arc::strong_count() includes:
        // 1. self.arc
        // 2. The Arc held by OwnedMutexGuard (now dropped)
        // 3. Any other waiting tasks
        //
        // If strong_count == 1, only self.arc remains
        if Arc::strong_count(&self.arc) == 1 {
            // Immediately remove from DashMap - no one else is using this lock
            self.locks.remove(&self.key);
        }
        // self.arc drops here, potentially destroying the Mutex
    }
}

impl<K> Deref for KeyedLockGuard<K>
where
    K: Hash + Eq + Clone,
{
    type Target = ();

    fn deref(&self) -> &Self::Target {
        self.guard.as_ref().expect("guard should exist")
    }
}

impl<K> DerefMut for KeyedLockGuard<K>
where
    K: Hash + Eq + Clone,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.as_mut().expect("guard should exist")
    }
}

/// A generic per-key async lock manager with automatic cleanup using weak references.
///
/// This structure allows serializing operations on a per-key basis, ensuring that
/// concurrent operations on the same key are executed sequentially while allowing
/// operations on different keys to proceed in parallel.
///
/// # Automatic Cleanup
///
/// The lock uses `Weak<Mutex<()>>` internally, which means locks are automatically
/// cleaned up when no operations are holding them. This prevents unbounded memory growth.
///
/// When the last guard for a key is dropped, the lock is immediately removed from memory.
///
/// # Example
///
/// ```rust,ignore
/// use utils::shared_memory::PodIdentifier;
/// use crate::util::keyed_lock::KeyedAsyncLock;
///
/// let locks = KeyedAsyncLock::<PodIdentifier>::new();
/// let pod_id = PodIdentifier::new("default", "my-pod");
///
/// // Acquire lock for this pod
/// let guard = locks.lock(&pod_id).await;
/// // ... perform operations ...
/// drop(guard); // Lock is released and immediately cleaned up
/// ```
pub struct KeyedAsyncLock<K>
where
    K: Hash + Eq + Clone,
{
    locks: Arc<DashMap<K, Weak<Mutex<()>>>>,
}

impl<K> KeyedAsyncLock<K>
where
    K: Hash + Eq + Clone,
{
    /// Creates a new `KeyedAsyncLock`.
    pub fn new() -> Self {
        Self {
            locks: Arc::new(DashMap::new()),
        }
    }

    /// Acquires a lock for the specified key.
    ///
    /// This method returns a custom guard that automatically cleans up the lock
    /// when it's dropped and no other tasks are using it.
    ///
    /// Multiple calls with the same key will serialize, while different keys can
    /// proceed concurrently.
    ///
    /// # Automatic Cleanup
    ///
    /// When the returned guard is dropped, if it's the last reference to that lock,
    /// the lock entry is immediately removed from the internal map.
    pub async fn lock(&self, key: &K) -> KeyedLockGuard<K> {
        let arc = self.get_or_create_lock(key);
        // Clone Arc before locking so we can track reference count
        let arc_clone = Arc::clone(&arc);
        let guard = arc.lock_owned().await;

        KeyedLockGuard::new(guard, arc_clone, key.clone(), Arc::clone(&self.locks))
    }

    /// Gets an existing lock or creates a new one for the specified key.
    ///
    /// This method handles the weak reference upgrade/cleanup logic automatically.
    fn get_or_create_lock(&self, key: &K) -> Arc<Mutex<()>> {
        loop {
            // Try to get or insert the lock
            let entry = self.locks.entry(key.clone());

            match entry {
                dashmap::mapref::entry::Entry::Occupied(occupied) => {
                    // Try to upgrade weak reference to strong
                    if let Some(strong) = occupied.get().upgrade() {
                        return strong;
                    }
                    // Weak reference is dead, remove it and retry
                    occupied.remove();
                }
                dashmap::mapref::entry::Entry::Vacant(vacant) => {
                    // Create new lock and store as weak reference
                    let strong = Arc::new(Mutex::new(()));
                    vacant.insert(Arc::downgrade(&strong));
                    return strong;
                }
            }
        }
    }

    /// Returns the current number of locks stored (including expired weak references).
    ///
    /// This is primarily useful for testing and debugging. The actual number of active
    /// locks may be lower due to expired weak references that haven't been cleaned up yet.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.locks.len()
    }

    /// Returns true if there are no locks stored.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.locks.is_empty()
    }

    /// Returns the number of active (non-expired) locks.
    ///
    /// This is useful for testing to verify that locks are being properly cleaned up.
    #[cfg(test)]
    pub fn active_locks(&self) -> usize {
        self.locks
            .iter()
            .filter(|entry| entry.value().strong_count() > 0)
            .count()
    }
}

impl<K> Default for KeyedAsyncLock<K>
where
    K: Hash + Eq + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn lock_serializes_same_key() {
        let locks = Arc::new(KeyedAsyncLock::<String>::new());
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Spawn 10 tasks that all try to increment the counter
        for _ in 0..10 {
            let locks = Arc::clone(&locks);
            let counter = Arc::clone(&counter);
            handles.push(tokio::spawn(async move {
                let _guard = locks.lock(&"same-key".to_string()).await;
                let val = counter.load(Ordering::SeqCst);
                sleep(Duration::from_millis(1)).await; // Simulate work
                counter.store(val + 1, Ordering::SeqCst);
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // All increments should have been serialized
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[tokio::test]
    async fn different_keys_run_concurrently() {
        let locks = Arc::new(KeyedAsyncLock::<String>::new());
        let start = std::time::Instant::now();

        let mut handles = vec![];
        for i in 0..5 {
            let locks = Arc::clone(&locks);
            let key = format!("key-{i}");
            handles.push(tokio::spawn(async move {
                let _guard = locks.lock(&key).await;
                sleep(Duration::from_millis(50)).await;
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let elapsed = start.elapsed();
        // Should take ~50ms if concurrent, ~250ms if serialized
        assert!(
            elapsed < Duration::from_millis(150),
            "Tasks should run concurrently"
        );
    }

    #[tokio::test]
    async fn automatic_cleanup() {
        let locks = KeyedAsyncLock::<String>::new();

        // Create and immediately drop a lock
        {
            let _guard = locks.lock(&"test-key".to_string()).await;
        }

        // With active cleanup, the lock should be immediately removed
        assert!(locks.is_empty(), "locks should be empty");
        assert_eq!(locks.active_locks(), 0);

        // Creating a new lock should work fine
        {
            let _guard = locks.lock(&"test-key".to_string()).await;
            // While holding the lock, it should exist
            assert_eq!(locks.len(), 1);
            assert_eq!(locks.active_locks(), 1);
        }

        // After dropping, it should be cleaned up again
        assert!(locks.is_empty(), "locks should be empty");
        assert_eq!(locks.active_locks(), 0);
    }

    #[tokio::test]
    async fn active_locks_count() {
        let locks = Arc::new(KeyedAsyncLock::<String>::new());

        // No active locks initially
        assert_eq!(locks.active_locks(), 0);

        // Hold a lock
        let guard = locks.lock(&"key1".to_string()).await;
        assert_eq!(locks.active_locks(), 1);

        // Hold another lock
        let guard2 = locks.lock(&"key2".to_string()).await;
        assert_eq!(locks.active_locks(), 2);

        // Drop first lock
        drop(guard);
        assert_eq!(locks.active_locks(), 1);

        // Drop second lock
        drop(guard2);
        assert_eq!(locks.active_locks(), 0);
    }

    #[tokio::test]
    async fn concurrent_access_to_same_key() {
        let locks = Arc::new(KeyedAsyncLock::<u32>::new());
        let success_count = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];
        for _ in 0..100 {
            let locks = Arc::clone(&locks);
            let success_count = Arc::clone(&success_count);
            handles.push(tokio::spawn(async move {
                let _guard = locks.lock(&42).await;
                // Simulate some work
                tokio::task::yield_now().await;
                success_count.fetch_add(1, Ordering::SeqCst);
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(success_count.load(Ordering::SeqCst), 100);
    }

    #[tokio::test]
    async fn weak_reference_cleanup_under_load() {
        let locks = Arc::new(KeyedAsyncLock::<u32>::new());

        // Create many locks and let them expire
        for i in 0..1000 {
            let _guard = locks.lock(&i).await;
            // Lock is dropped immediately - with active cleanup, it's removed
        }

        // With active cleanup, all locks should be immediately removed after dropping
        assert!(locks.is_empty(), "locks should be empty");
        assert_eq!(locks.active_locks(), 0);

        // Create multiple locks simultaneously to test cleanup
        let mut guards = vec![];
        for i in 0..10 {
            guards.push(locks.lock(&i).await);
        }

        // While holding guards, locks should exist
        assert_eq!(locks.len(), 10);
        assert_eq!(locks.active_locks(), 10);

        // Drop all guards
        drop(guards);

        // After dropping, all should be cleaned up
        assert!(locks.is_empty(), "locks should be empty");
        assert_eq!(locks.active_locks(), 0);
    }

    #[tokio::test]
    async fn concurrent_registration_simulation() {
        use std::sync::atomic::{AtomicBool, AtomicUsize};

        let locks = Arc::new(KeyedAsyncLock::<String>::new());
        let registration_count = Arc::new(AtomicUsize::new(0));
        let check_count = Arc::new(AtomicUsize::new(0));
        let is_registered = Arc::new(AtomicBool::new(false));

        let pod_key = "test-pod".to_string();
        let mut handles = vec![];

        // Simulate 50 concurrent requests trying to register the same pod
        for i in 0..50 {
            let locks = Arc::clone(&locks);
            let registration_count = Arc::clone(&registration_count);
            let check_count = Arc::clone(&check_count);
            let is_registered = Arc::clone(&is_registered);
            let pod_key = pod_key.clone();

            handles.push(tokio::spawn(async move {
                // Simulate fast path check (without lock)
                if is_registered.load(Ordering::SeqCst) {
                    check_count.fetch_add(1, Ordering::SeqCst);
                    return;
                }

                // Acquire lock
                let _guard = locks.lock(&pod_key).await;

                // Double-check after acquiring lock
                if is_registered.load(Ordering::SeqCst) {
                    check_count.fetch_add(1, Ordering::SeqCst);
                    return;
                }

                // Simulate expensive registration operation
                sleep(Duration::from_millis(10)).await;

                // Mark as registered
                is_registered.store(true, Ordering::SeqCst);
                registration_count.fetch_add(1, Ordering::SeqCst);

                // Simulate some work after registration
                tokio::task::yield_now().await;
            }));

            // Small delay to increase chance of contention
            if i % 10 == 0 {
                tokio::task::yield_now().await;
            }
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify that only one registration actually happened
        assert_eq!(
            registration_count.load(Ordering::SeqCst),
            1,
            "Only one request should perform actual registration"
        );

        // Verify that 49 requests were blocked by double-check
        assert_eq!(
            check_count.load(Ordering::SeqCst),
            49,
            "49 requests should skip registration due to double-check"
        );

        // Verify lock was cleaned up
        assert!(
            locks.is_empty(),
            "Lock should be cleaned up after all tasks complete"
        );
        assert_eq!(locks.active_locks(), 0);
    }

    #[tokio::test]
    async fn concurrent_registration_multiple_pods() {
        use std::collections::HashMap;

        let locks = Arc::new(KeyedAsyncLock::<String>::new());
        let registration_counts = Arc::new(Mutex::new(HashMap::new()));

        let mut handles = vec![];

        // Simulate concurrent registration for 10 different pods
        // Each pod has 20 concurrent registration attempts
        for pod_idx in 0..10 {
            for _ in 0..20 {
                let locks = Arc::clone(&locks);
                let registration_counts = Arc::clone(&registration_counts);
                let pod_key = format!("pod-{pod_idx}");

                handles.push(tokio::spawn(async move {
                    // Acquire lock for this pod
                    let _guard = locks.lock(&pod_key).await;

                    // Check if already registered
                    let mut counts = registration_counts.lock().await;
                    if counts.contains_key(&pod_key) {
                        return;
                    }

                    // Simulate registration
                    sleep(Duration::from_millis(5)).await;
                    counts.insert(pod_key.clone(), 1);
                }));
            }
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify each pod was registered exactly once
        let counts = registration_counts.lock().await;
        assert_eq!(counts.len(), 10, "All 10 pods should be registered");
        for (pod_key, count) in counts.iter() {
            assert_eq!(*count, 1, "Pod {pod_key} should be registered exactly once");
        }

        // Verify all locks were cleaned up
        assert!(locks.is_empty(), "All locks should be cleaned up");
        assert_eq!(locks.active_locks(), 0);
    }

    #[tokio::test]
    async fn stress_test_lock_contention() {
        let locks = Arc::new(KeyedAsyncLock::<u32>::new());
        let success_count = Arc::new(AtomicUsize::new(0));
        let task_counter = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];

        // Create heavy contention: 100 tasks competing for 5 locks
        for _ in 0..100 {
            let locks = Arc::clone(&locks);
            let success_count = Arc::clone(&success_count);
            let task_counter = Arc::clone(&task_counter);

            handles.push(tokio::spawn(async move {
                let task_id = task_counter.fetch_add(1, Ordering::SeqCst);
                // Use task_id to deterministically pick a lock (5 different locks)
                let key = (task_id % 5) as u32;
                let _guard = locks.lock(&key).await;

                // Simulate work
                tokio::task::yield_now().await;
                success_count.fetch_add(1, Ordering::SeqCst);

                // Variable delay based on task_id
                if task_id % 3 == 0 {
                    sleep(Duration::from_micros(50)).await;
                }
            }));
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // All tasks should complete successfully
        assert_eq!(success_count.load(Ordering::SeqCst), 100);

        // All locks should be cleaned up
        assert!(locks.is_empty(), "All locks should be cleaned up");
        assert_eq!(locks.active_locks(), 0);
    }
}
