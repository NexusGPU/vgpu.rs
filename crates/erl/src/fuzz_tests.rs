//! Fuzz test suite
//!
//! Test ERL system behavior under various random inputs and boundary conditions

use crate::token_manager::SimpleTokenManager;
use crate::traits::{ErlError, SharedStorage, TokenManager, UtilizationController};
use crate::utilization_controller::HypervisorUtilizationController;

use error_stack::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use proptest::prelude::*;

/// Simple memory storage implementation (only for testing)
#[derive(Debug, Clone)]
pub struct TestMemoryStorage<K>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + std::fmt::Debug,
{
    tokens: Arc<Mutex<HashMap<K, (f64, f64)>>>, // (tokens, timestamp)
    quotas: Arc<Mutex<HashMap<K, (f64, f64)>>>, // (capacity, refill_rate)
    avg_costs: Arc<Mutex<HashMap<K, f64>>>,     // refill_rate per device
}

impl<K> TestMemoryStorage<K>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + std::fmt::Debug,
{
    pub fn new() -> Self {
        Self {
            tokens: Arc::new(Mutex::new(HashMap::new())),
            quotas: Arc::new(Mutex::new(HashMap::new())),
            avg_costs: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl<K> SharedStorage<K> for TestMemoryStorage<K>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + std::fmt::Debug,
{
    fn load_token_state(&self, key: &K) -> Result<(f64, f64), ErlError> {
        let tokens = self.tokens.lock().map_err(|_| {
            error_stack::report!(ErlError::MonitoringFailed {
                reason: "Failed to acquire token lock".to_string()
            })
        })?;

        tokens.get(key).copied().ok_or_else(|| {
            error_stack::report!(ErlError::InvalidConfiguration {
                reason: format!("Token state not found for key: {key:?}")
            })
        })
    }

    fn save_token_state(&self, key: &K, tokens: f64, timestamp: f64) -> Result<(), ErlError> {
        let mut token_map = self.tokens.lock().map_err(|_| {
            error_stack::report!(ErlError::MonitoringFailed {
                reason: "Failed to acquire token lock".to_string()
            })
        })?;

        token_map.insert(key.clone(), (tokens, timestamp));
        Ok(())
    }

    fn load_quota(&self, key: &K) -> Result<(f64, f64), ErlError> {
        let quotas = self.quotas.lock().map_err(|_| {
            error_stack::report!(ErlError::MonitoringFailed {
                reason: "Failed to acquire quota lock".to_string()
            })
        })?;

        quotas.get(key).copied().ok_or_else(|| {
            error_stack::report!(ErlError::InvalidConfiguration {
                reason: format!("Quota not found for key: {key:?}")
            })
        })
    }

    fn set_quota(&self, key: &K, capacity: f64, refill_rate: f64) -> Result<(), ErlError> {
        let mut quotas = self.quotas.lock().map_err(|_| {
            error_stack::report!(ErlError::MonitoringFailed {
                reason: "Failed to acquire quota lock".to_string()
            })
        })?;

        quotas.insert(key.clone(), (capacity, refill_rate));

        // Initialize token state to full capacity
        let mut tokens = self.tokens.lock().map_err(|_| {
            error_stack::report!(ErlError::MonitoringFailed {
                reason: "Failed to acquire token lock".to_string()
            })
        })?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        tokens.insert(key.clone(), (capacity, now));

        Ok(())
    }

    fn load_avg_cost(&self, key: &K) -> Result<f64, ErlError> {
        let avg_costs = self.avg_costs.lock().map_err(|_| {
            error_stack::report!(ErlError::MonitoringFailed {
                reason: "Failed to acquire avg_cost lock".to_string()
            })
        })?;

        avg_costs.get(key).copied().ok_or_else(|| {
            error_stack::report!(ErlError::InvalidConfiguration {
                reason: format!("Avg cost not found for key: {key:?}")
            })
        })
    }

    fn save_avg_cost(&self, key: &K, avg_cost: f64) -> Result<(), ErlError> {
        let mut avg_costs = self.avg_costs.lock().map_err(|_| {
            error_stack::report!(ErlError::MonitoringFailed {
                reason: "Failed to acquire avg_cost lock".to_string()
            })
        })?;

        avg_costs.insert(key.clone(), avg_cost);
        Ok(())
    }

    fn increment_kernel_count(&self, _key: &K) -> Result<(), ErlError> {
        // No-op for testing
        Ok(())
    }

    fn load_and_reset_kernel_count(&self, _key: &K) -> Result<u64, ErlError> {
        // Return fixed count for testing
        Ok(100)
    }
}

/// Generate random workload parameters strategy
fn workload_strategy() -> impl Strategy<Value = (u32, u32)> {
    (
        1u32..=10000, // grid_count: 1 to 10,000
        1u32..=2048,  // block_count: 1 to 2,048 (GPU hardware limit)
    )
}

/// Generate random quota configuration strategy
fn quota_strategy() -> impl Strategy<Value = (f64, f64)> {
    (
        1.0..=10000.0, // capacity: 1 to 10,000
        0.1..=1000.0,  // refill_rate: 0.1 to 1,000 per second
    )
}

/// Operation sequence
#[derive(Debug, Clone)]
enum ErlOperation {
    Acquire { grid_count: u32, block_count: u32 },
    Noop,
}

/// Generate random operation sequence strategy
fn operation_sequence_strategy() -> impl Strategy<Value = Vec<ErlOperation>> {
    prop::collection::vec(
        prop_oneof![
            3 => workload_strategy().prop_map(|(g, b)| ErlOperation::Acquire {
                grid_count: g,
                block_count: b
            }),
            1 => Just(ErlOperation::Noop),
        ],
        1..20,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Property test: Token conservation
        #[test]
        fn property_token_conservation(
            operations in operation_sequence_strategy(),
            (capacity, refill_rate) in quota_strategy(),
        ) {
            let storage = TestMemoryStorage::new();
            let device_key = "test-device".to_string();

            storage.set_quota(&device_key, capacity, refill_rate).unwrap();

            let mut token_manager = SimpleTokenManager::new(storage.clone());

            for op in operations {
                match op {
                    ErlOperation::Acquire { grid_count, block_count } => {
                        let (tokens_before, _, _) = token_manager.get_token_status(&device_key).unwrap();

                        let result = token_manager.try_acquire_workload(&device_key, grid_count, block_count);

                        let (tokens_after, _, _) = token_manager.get_token_status(&device_key).unwrap();

                        if result.is_ok() {
                            // When acquire succeeds, net change should be approximately -1.0 (minus refill)
                            // Since refill can happen during the operation, we only check:
                            // 1. Tokens decreased (at least some cost was deducted)
                            // 2. The decrease is at most 1.0 (can be less due to refill)
                            let net_change = tokens_after - tokens_before;
                            prop_assert!(net_change <= 0.0,
                                "Tokens should decrease or stay same after acquire, got change of {net_change}");
                            prop_assert!(net_change >= -1.01,
                                "Tokens should decrease by at most 1.0 (with small tolerance), got change of {net_change}");
                        }

                        // tokens should never exceed capacity
                        prop_assert!(tokens_after <= capacity + 0.001);
                        prop_assert!(tokens_after >= 0.0);
                    }
                    ErlOperation::Noop => {}
                }
            }
        }

        /// Property test: Fixed cost per kernel
        #[test]
        fn property_fixed_cost(
            small_workload in (1u32..=100, 1u32..=100),
            large_workload in (500u32..=2000, 500u32..=1024),
            (capacity, refill_rate) in quota_strategy(),
        ) {
            let storage = TestMemoryStorage::new();
            let device_key = "test-device".to_string();

            storage.set_quota(&device_key, capacity, refill_rate).unwrap();

            let mut token_manager = SimpleTokenManager::new(storage.clone());

            let (small_grid, small_block) = small_workload;
            let (large_grid, large_block) = large_workload;

            // Try small workload
            let (tokens_before_small, _, _) = token_manager.get_token_status(&device_key).unwrap();
            let small_result = token_manager.try_acquire_workload(&device_key, small_grid, small_block);

            if small_result.is_ok() {
                let (tokens_after_small, _, _) = token_manager.get_token_status(&device_key).unwrap();
                let small_cost = tokens_before_small - tokens_after_small;

                // Reset to full capacity
                storage.set_quota(&device_key, capacity, refill_rate).unwrap();

                // Try large workload
                let (tokens_before_large, _, _) = token_manager.get_token_status(&device_key).unwrap();
                let large_result = token_manager.try_acquire_workload(&device_key, large_grid, large_block);

                if large_result.is_ok() {
                    let (tokens_after_large, _, _) = token_manager.get_token_status(&device_key).unwrap();
                    let large_cost = tokens_before_large - tokens_after_large;

                    // Both should cost approximately 1.0
                    // Allow tolerance for refill during operation
                    prop_assert!((0.99..=1.01).contains(&small_cost),
                        "Small kernel should cost ~1.0, got {small_cost}");
                    prop_assert!((0.99..=1.01).contains(&large_cost),
                        "Large kernel should cost ~1.0, got {large_cost}");
                }
            }
        }

        /// Property test: PI controller stability
        #[test]
        fn property_pi_stability(
            stable_utilization in 0.3f64..=0.9,
            update_count in 5usize..=15,
        ) {
            let storage = TestMemoryStorage::new();
            let device_key = "gpu-0".to_string();

            storage.set_quota(&device_key, 1000.0, 100.0).unwrap();

            let mut hypervisor = HypervisorUtilizationController::new(storage.clone());
            hypervisor.initialize_device_quota(&device_key, 1000.0, 10.0).unwrap();
            hypervisor.register_pod("test-pod".to_string(), device_key.clone(), 0.8, 10.0).unwrap();

            for _ in 0..update_count {
                // update_control now reads kernel counts from shared memory
                let _ = hypervisor.update_control(stable_utilization);
            }

            // System should remain stable
            let status = hypervisor.get_pods_status();
            prop_assert_eq!(status.len(), 1);

            let (_name, _target, rate, _count) = &status[0];
            prop_assert!(*rate > 0.1 && *rate < 1000.0, "Rate {rate} should be in reasonable range");
        }

        /// Property test: Concurrent safety
        #[test]
        fn property_concurrent_safety(
            operations in prop::collection::vec(operation_sequence_strategy(), 2..=3),
        ) {
            use std::sync::Arc;
            use std::thread;

            let storage = TestMemoryStorage::new();
            let device_key = "test-device".to_string();

            storage.set_quota(&device_key, 10000.0, 1000.0).unwrap();

            let token_manager = Arc::new(Mutex::new(SimpleTokenManager::new(storage.clone())));

            let mut handles = vec![];

            for ops in operations {
                let token_manager_clone = Arc::clone(&token_manager);
                let key = device_key.clone();

                let handle = thread::spawn(move || {
                    for op in ops {
                        match op {
                            ErlOperation::Acquire { grid_count, block_count } => {
                                if let Ok(mut mgr) = token_manager_clone.lock() {
                                    let _ = mgr.try_acquire_workload(&key, grid_count, block_count);
                                }
                            }
                            ErlOperation::Noop => {}
                        }
                    }
                });

                handles.push(handle);
            }

            for handle in handles {
                prop_assert!(handle.join().is_ok());
            }

            if let Ok(mgr) = token_manager.lock() {
                let (tokens, capacity, refill_rate) = mgr.get_token_status(&device_key).unwrap();
                prop_assert!(tokens >= 0.0);
                prop_assert!(tokens <= capacity + 0.001);
                prop_assert!(capacity == 10000.0);
                prop_assert!(refill_rate == 1000.0);
            }
        }
    }

    /// Regression test: zero workload
    #[test]
    fn regression_zero_workload() {
        let storage = TestMemoryStorage::new();
        let device_key = "test-device".to_string();

        storage.set_quota(&device_key, 100.0, 10.0).unwrap();
        let mut token_manager = SimpleTokenManager::new(storage);

        let result = token_manager.try_acquire_workload(&device_key, 0, 1);
        assert!(result.is_ok() || result.is_err());

        let result = token_manager.try_acquire_workload(&device_key, 1, 0);
        assert!(result.is_ok() || result.is_err());
    }

    /// Regression test: rapid utilization changes
    #[test]
    fn regression_rapid_utilization_changes() {
        let storage = TestMemoryStorage::new();
        let device_key = "gpu-0".to_string();

        storage.set_quota(&device_key, 1000.0, 100.0).unwrap();

        let mut hypervisor = HypervisorUtilizationController::new(storage.clone());
        let mut token_manager = SimpleTokenManager::new(storage);

        hypervisor
            .initialize_device_quota(&device_key, 1000.0, 10.0)
            .unwrap();

        hypervisor
            .register_pod("test-pod".to_string(), device_key.clone(), 0.8, 10.0)
            .unwrap();

        for i in 0..100 {
            let utilization = if i % 2 == 0 { 0.1 } else { 0.9 };

            let result = hypervisor.update_control(utilization);
            assert!(result.is_ok(), "Update {i} failed: {result:?}");
        }

        let result = token_manager.try_acquire_workload(&device_key, 32, 32);
        assert!(result.is_ok() || result.is_err());
    }

    /// Performance test
    #[test]
    fn performance_basic_operations() {
        let storage = TestMemoryStorage::new();
        let device_key = "gpu-0".to_string();

        storage.set_quota(&device_key, 10000.0, 1000.0).unwrap();

        let mut hypervisor = HypervisorUtilizationController::new(storage.clone());
        let mut token_manager = SimpleTokenManager::new(storage);

        hypervisor
            .initialize_device_quota(&device_key, 10000.0, 100.0)
            .unwrap();

        hypervisor
            .register_pod("test-pod".to_string(), device_key.clone(), 0.8, 100.0)
            .unwrap();

        let start = std::time::Instant::now();

        for i in 0..1000 {
            let _ = token_manager.try_acquire_workload(&device_key, 32, 32);
            if i % 10 == 0 {
                let _ = hypervisor.update_control(0.7);
            }
        }

        let duration = start.elapsed();

        assert!(
            duration.as_secs() < 5,
            "1000 operations took too long: {duration:?}"
        );
    }
}
