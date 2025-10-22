//! Simplified token manager implementation
//!
//! All kernels consume a fixed token cost (1.0), no workload prediction needed

use error_stack::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::traits::{ErlError, SharedStorage, TokenManager};

/// Simplified token manager with fixed cost per kernel
///
/// Run in limiter process, responsible for:
/// 1. Token bucket admission control (all kernels cost 1.0 token)
/// 2. Counting kernel launches for statistics
#[derive(Debug)]
pub struct SimpleTokenManager<K, S>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync,
    S: SharedStorage<K>,
{
    /// Shared storage
    storage: S,
    /// Kernel launch counter (for statistics)
    kernel_count: AtomicU64,
    /// Type marker
    _phantom: std::marker::PhantomData<K>,
}

impl<K, S> SimpleTokenManager<K, S>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync,
    S: SharedStorage<K>,
{
    /// Create new simple token manager
    pub fn new(storage: S) -> Self {
        Self {
            storage,
            kernel_count: AtomicU64::new(0),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get and reset kernel count (for periodic reporting)
    pub fn take_kernel_count(&self) -> u64 {
        self.kernel_count.swap(0, Ordering::Relaxed)
    }

    /// Execute token bucket refill logic
    fn refill_tokens(&self, key: &K) -> Result<f64, ErlError> {
        let (capacity, refill_rate) = self.storage.load_quota(key)?;
        let (mut tokens, last_timestamp) = self.storage.load_token_state(key)?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let delta_time = now - last_timestamp;
        let tokens_to_add = refill_rate * delta_time;
        tokens = (tokens + tokens_to_add).min(capacity);

        self.storage.save_token_state(key, tokens, now)?;
        Ok(tokens)
    }
}

impl<K, S> TokenManager<K> for SimpleTokenManager<K, S>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + std::fmt::Debug,
    S: SharedStorage<K>,
{
    type Storage = S;

    fn try_acquire_workload(
        &mut self,
        key: &K,
        _grid_count: u32,
        _block_count: u32,
    ) -> Result<(), ErlError> {
        // 1. Execute refill
        let current_tokens = self.refill_tokens(key)?;

        // 2. All kernels consume fixed 1.0 token
        const FIXED_COST: f64 = 1.0;

        // 3. Check and deduct tokens
        if current_tokens >= FIXED_COST {
            let new_tokens = current_tokens - FIXED_COST;
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();

            self.storage.save_token_state(key, new_tokens, now)?;

            // Increment kernel count in shared memory
            self.storage.increment_kernel_count(key)?;

            // Also keep local counter for compatibility
            self.kernel_count.fetch_add(1, Ordering::Relaxed);

            tracing::trace!(
                key = ?key,
                remaining_tokens = new_tokens,
                "Kernel launch admitted"
            );

            Ok(())
        } else {
            tracing::debug!(
                key = ?key,
                available_tokens = current_tokens,
                "Kernel launch denied: insufficient tokens"
            );

            Err(error_stack::report!(ErlError::AdmissionDenied {
                reason: format!(
                    "Insufficient tokens for {key:?}: need {FIXED_COST:.1}, have {current_tokens:.3}"
                )
            }))
        }
    }

    fn get_token_status(&self, key: &K) -> Result<(f64, f64, f64), ErlError> {
        let (capacity, refill_rate) = self.storage.load_quota(key)?;
        let current_tokens = self.refill_tokens(key)?;
        Ok((current_tokens, capacity, refill_rate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};

    // Simple in-memory storage for testing
    #[derive(Debug, Clone)]
    struct MockStorage {
        quotas: Arc<RwLock<HashMap<String, (f64, f64)>>>,
        token_states: Arc<RwLock<HashMap<String, (f64, f64)>>>,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                quotas: Arc::new(RwLock::new(HashMap::new())),
                token_states: Arc::new(RwLock::new(HashMap::new())),
            }
        }
    }

    impl SharedStorage<String> for MockStorage {
        fn load_token_state(&self, key: &String) -> Result<(f64, f64), ErlError> {
            self.token_states
                .read()
                .unwrap()
                .get(key)
                .copied()
                .ok_or_else(|| {
                    error_stack::report!(ErlError::MonitoringFailed {
                        reason: format!("Token state not found for {key}")
                    })
                })
        }

        fn save_token_state(
            &self,
            key: &String,
            tokens: f64,
            timestamp: f64,
        ) -> Result<(), ErlError> {
            self.token_states
                .write()
                .unwrap()
                .insert(key.clone(), (tokens, timestamp));
            Ok(())
        }

        fn load_quota(&self, key: &String) -> Result<(f64, f64), ErlError> {
            self.quotas
                .read()
                .unwrap()
                .get(key)
                .copied()
                .ok_or_else(|| {
                    error_stack::report!(ErlError::MonitoringFailed {
                        reason: format!("Quota not found for {key}")
                    })
                })
        }

        fn set_quota(&self, key: &String, capacity: f64, refill_rate: f64) -> Result<(), ErlError> {
            self.quotas
                .write()
                .unwrap()
                .insert(key.clone(), (capacity, refill_rate));
            Ok(())
        }

        fn load_avg_cost(&self, _key: &String) -> Result<f64, ErlError> {
            Ok(1.0)
        }

        fn save_avg_cost(&self, _key: &String, _avg_cost: f64) -> Result<(), ErlError> {
            Ok(())
        }

        fn increment_kernel_count(&self, _key: &String) -> Result<(), ErlError> {
            Ok(())
        }

        fn load_and_reset_kernel_count(&self, _key: &String) -> Result<u64, ErlError> {
            Ok(0)
        }
    }

    #[test]
    fn simple_token_manager_admits_when_tokens_available() {
        let storage = MockStorage::new();
        let key = "test-pod".to_string();

        storage
            .set_quota(&key, 10.0, 5.0)
            .expect("should set quota");
        storage
            .save_token_state(&key, 10.0, 0.0)
            .expect("should save token state");

        let mut manager = SimpleTokenManager::new(storage);

        let result = manager.try_acquire_workload(&key, 100, 256);
        assert!(result.is_ok(), "Should admit when tokens available");

        assert_eq!(manager.take_kernel_count(), 1, "Should count kernel launch");
    }

    #[test]
    fn simple_token_manager_denies_when_tokens_insufficient() {
        let storage = MockStorage::new();
        let key = "test-pod".to_string();

        storage
            .set_quota(&key, 10.0, 0.0) // Set refill_rate to 0 to prevent refill
            .expect("should set quota");

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        storage
            .save_token_state(&key, 0.5, now)
            .expect("should save token state");

        let mut manager = SimpleTokenManager::new(storage);

        let result = manager.try_acquire_workload(&key, 100, 256);
        assert!(
            result.is_err(),
            "Should deny when tokens insufficient (need 1.0, have 0.5)"
        );

        assert_eq!(
            manager.take_kernel_count(),
            0,
            "Should not count denied launch"
        );
    }

    #[test]
    fn simple_token_manager_ignores_kernel_size() {
        let storage = MockStorage::new();
        let key = "test-pod".to_string();

        storage
            .set_quota(&key, 10.0, 5.0)
            .expect("should set quota");
        storage
            .save_token_state(&key, 10.0, 0.0)
            .expect("should save token state");

        let mut manager = SimpleTokenManager::new(storage.clone());

        // Launch small kernel
        manager
            .try_acquire_workload(&key, 1, 32)
            .expect("should admit small kernel");

        let (tokens_after_small, _, _) = manager
            .get_token_status(&key)
            .expect("should get token status");

        // Reset tokens
        storage
            .save_token_state(&key, 10.0, 0.0)
            .expect("should save token state");

        // Launch large kernel
        manager
            .try_acquire_workload(&key, 1000, 1024)
            .expect("should admit large kernel");

        let (tokens_after_large, _, _) = manager
            .get_token_status(&key)
            .expect("should get token status");

        // Both should consume same amount
        assert!(
            (tokens_after_small - tokens_after_large).abs() < 0.01,
            "Small and large kernels should consume same tokens"
        );
    }
}
