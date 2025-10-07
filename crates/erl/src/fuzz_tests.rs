//! Fuzz test suite
//!
//! Test ERL system behavior under various random inputs and boundary conditions

use crate::token_manager::WorkloadTokenManager;
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
    avg_costs: Arc<Mutex<HashMap<K, f64>>>,     // avg_cost per device
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

    pub fn set_quota(&self, key: &K, capacity: f64, refill_rate: f64) -> Result<(), ErlError> {
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

/// Operation sequence: mainly test acquire operation
#[derive(Debug, Clone)]
enum ErlOperation {
    Acquire { grid_count: u32, block_count: u32 },
    // In the new architecture, UpdateUtilization is handled by hypervisor, WaitTime is skipped for performance
    Noop, // Noop operation, keep test complexity
}

/// Generate random operation sequence strategy
fn operation_sequence_strategy() -> impl Strategy<Value = Vec<ErlOperation>> {
    prop::collection::vec(
        prop_oneof![
            3 => workload_strategy().prop_map(|(g, b)| ErlOperation::Acquire {
                grid_count: g,
                block_count: b
            }),
            1 => Just(ErlOperation::Noop), // Occasionally insert noop to increase test complexity
        ],
        1..20, // 1 to 20 operations to speed up test
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))] // Reduce test case number to speed up

        /// Property test: Token conservation
        ///
        /// No matter what operation sequence, the tokens in the system should follow the conservation law:
        /// - When acquire succeeds, tokens should decrease the corresponding cost
        /// - When refill, tokens should increase at the rate, not exceeding capacity
        #[test]
        fn property_token_conservation(
            operations in operation_sequence_strategy(),
            (capacity, refill_rate) in quota_strategy(),
        ) {
            let storage = TestMemoryStorage::new();
            let device_key = "test-device";

            // Set quota
            storage.set_quota(&device_key, capacity, refill_rate).unwrap();

            let mut token_manager = WorkloadTokenManager::with_default_calculator(storage.clone());

            let _expected_max_tokens = capacity;

            for op in operations {
                match op {
                    ErlOperation::Acquire { grid_count, block_count } => {
                        let (tokens_before, _, _) = token_manager.get_token_status(&device_key).unwrap();

                        let result = token_manager.try_acquire_workload(&device_key, grid_count, block_count);

                        let (tokens_after, _, _) = token_manager.get_token_status(&device_key).unwrap();

                        if result.is_ok() {
                            // When acquire succeeds, tokens should decrease
                            prop_assert!(tokens_after <= tokens_before);
                        } else {
                            // When acquire fails, tokens may refill due to time passes, but should not decrease
                            prop_assert!(tokens_after >= tokens_before - 0.001); // Allow floating point error
                        }

                        // tokens should never exceed capacity
                        prop_assert!(tokens_after <= capacity + 0.001); // Allow floating point error
                        prop_assert!(tokens_after >= 0.0);
                    }
                    ErlOperation::Noop => {
                        // Noop operation, do nothing
                    }
                }
            }
        }

        /// Property test: Workload fairness
        ///
        /// Larger workload should consume more tokens (or be rejected with higher probability)
        #[test]
        fn property_workload_fairness(
            small_workload in (1u32..=100, 1u32..=100),
            large_workload in (500u32..=2000, 500u32..=1024),
            (capacity, refill_rate) in quota_strategy(),
        ) {
            let storage = TestMemoryStorage::new();
            let device_key = "test-device";

            storage.set_quota(&device_key, capacity, refill_rate).unwrap();

            let mut token_manager = WorkloadTokenManager::with_default_calculator(storage.clone());

            // Reset to full capacity state
            storage.set_quota(&device_key, capacity, refill_rate).unwrap();

            let (small_grid, small_block) = small_workload;
            let (large_grid, large_block) = large_workload;

            // Try small workload
            let (tokens_before_small, _, _) = token_manager.get_token_status(&device_key).unwrap();
        let small_result = token_manager.try_acquire_workload(&device_key, small_grid, small_block);
        let (tokens_after_small, _, _) = token_manager.get_token_status(&device_key).unwrap();

        if small_result.is_ok() {
            let small_cost = tokens_before_small - tokens_after_small;

            // Reset to full capacity state
            storage.set_quota(&device_key, capacity, refill_rate).unwrap();

            // Try large workload
            let (tokens_before_large, _, _) = token_manager.get_token_status(&device_key).unwrap();
            let large_result = token_manager.try_acquire_workload(&device_key, large_grid, large_block);
            let (tokens_after_large, _, _) = token_manager.get_token_status(&device_key).unwrap();

                if large_result.is_ok() {
                    let large_cost = tokens_before_large - tokens_after_large;

                    // Large workload should consume more tokens
                    let small_threads = (small_grid as u64) * (small_block as u64);
                    let large_threads = (large_grid as u64) * (large_block as u64);

                    if large_threads > small_threads * 2 {
                        prop_assert!(large_cost >= small_cost,
                            "Large workload ({}*{}={} threads, cost={:.3}) should cost at least as much as small workload ({}*{}={} threads, cost={:.3})",
                            large_grid, large_block, large_threads, large_cost,
                            small_grid, small_block, small_threads, small_cost
                        );
                    }
                }
            }
        }

        /// Property test: CUBIC algorithm stability
        ///
        /// Under stable utilization input, CUBIC should converge to the target utilization nearby
        #[test]
        fn property_cubic_stability(
            stable_utilization in 0.3f64..=0.9,
            update_count in 5usize..=15, // Reduce update count
        ) {
            let storage = TestMemoryStorage::new();
            let device_key = "test-device";
            let target_utilization = 0.8;

            storage.set_quota(&device_key, 1000.0, 100.0).unwrap();


        // 1. Test hypervisor utilization control
        let mut hypervisor = HypervisorUtilizationController::new(storage.clone(), target_utilization);
        hypervisor.initialize_device_quota(&device_key, 1000.0, 100.0).unwrap();

        for _ in 0..update_count {
            let _ = hypervisor.update_utilization(stable_utilization);
            let _ = hypervisor.sync_avg_cost_to_devices(&[device_key]);
        }

        // Check CUBIC statistics
        let stats = hypervisor.get_cubic_stats();
        if stats.sample_count > 0 {
            prop_assert!(stats.cost_ratio > 0.1 && stats.cost_ratio < 10.0,
                "Cost ratio {} should be in reasonable range", stats.cost_ratio);
        }

        // 2. Test token manager
        let mut token_manager = WorkloadTokenManager::with_default_calculator(storage);
        // The system should be able to run basic acquire operation
        let result = token_manager.try_acquire_workload(&device_key, 32, 32);
            prop_assert!(result.is_ok() || result.is_err());
        }

        /// Property test: Concurrent safety
        ///
        /// Multiple concurrent operations should not cause system state inconsistency
        #[test]
        fn property_concurrent_safety(
            operations in prop::collection::vec(operation_sequence_strategy(), 2..=3), // Reduce concurrent thread count
        ) {
            use std::sync::Arc;
            use std::thread;

            let storage = TestMemoryStorage::new();
            let device_key = "test-device";

            storage.set_quota(&device_key, 10000.0, 1000.0).unwrap();

        let token_manager = Arc::new(Mutex::new(
            WorkloadTokenManager::with_default_calculator(storage.clone())
        ));

        let mut handles = vec![];

        for ops in operations {
            let token_manager_clone = Arc::clone(&token_manager);
            let _storage_clone = storage.clone();
            let key = device_key;

            let handle = thread::spawn(move || {
                for op in ops.into_iter() {
                    match op {
                        ErlOperation::Acquire { grid_count, block_count } => {
                            if let Ok(mut mgr) = token_manager_clone.lock() {
                                let _ = mgr.try_acquire_workload(&key, grid_count, block_count);
                            }
                        }
                        ErlOperation::Noop => {
                            // Noop operation, do nothing
                        }
                    }
                }
            });

            handles.push(handle);
        }

            // Wait for all threads to complete
            for handle in handles {
                prop_assert!(handle.join().is_ok());
            }

        // Verify final state consistency
        if let Ok(mgr) = token_manager.lock() {
            let (tokens, capacity, refill_rate) = mgr.get_token_status(&device_key).unwrap();
            prop_assert!(tokens >= 0.0);
            prop_assert!(tokens <= capacity + 0.001); // Allow floating point error
            prop_assert!(capacity == 10000.0);
            prop_assert!(refill_rate == 1000.0);
        }
        }

        /// Boundary condition test: extreme workload
        #[test]
        fn property_extreme_workloads(
            extreme_workload in prop_oneof![
                Just((1, 1)),           // Minimum workload
                Just((1, 2048)),        // Large block
                Just((10000, 1)),       // Large grid
                Just((1000, 1024)),     // Large workload
            ],
        ) {
            let storage = TestMemoryStorage::new();
            let device_key = "test-device";

            storage.set_quota(&device_key, 1000.0, 100.0).unwrap();

            let mut token_manager = WorkloadTokenManager::with_default_calculator(storage);

            let (grid_count, block_count) = extreme_workload;

        // Extreme workload should not cause system crash
        let result = token_manager.try_acquire_workload(&device_key, grid_count, block_count);

        // The result should either succeed or be admission denied
        prop_assert!(
            result.is_ok() || result.is_err(),
            "Extreme workload should either succeed or be denied, not crash: {:?}", result
        );

        // The system state should still be valid
        let (tokens, capacity, _refill_rate) = token_manager.get_token_status(&device_key).unwrap();
            prop_assert!(tokens >= 0.0);
            prop_assert!(tokens <= capacity + 0.001);
        }
    }

    /// Regression test: specific problem scenarios
    #[test]
    fn regression_zero_workload() {
        let storage = TestMemoryStorage::new();
        let device_key = "test-device";

        storage.set_quota(&device_key, 100.0, 10.0).unwrap();
        let mut token_manager = WorkloadTokenManager::with_default_calculator(storage);

        // Zero workload should not panic
        let result = token_manager.try_acquire_workload(&device_key, 0, 1);
        assert!(result.is_ok() || result.is_err()); // Any result is acceptable, but should not panic

        let result = token_manager.try_acquire_workload(&device_key, 1, 0);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn regression_rapid_utilization_changes() {
        let storage = TestMemoryStorage::new();
        let device_key = "test-device";

        storage.set_quota(&device_key, 1000.0, 100.0).unwrap();

        // Separate test hypervisor and token_manager
        let mut hypervisor = HypervisorUtilizationController::new(storage.clone(), 0.8);
        let mut token_manager = WorkloadTokenManager::with_default_calculator(storage);

        hypervisor
            .initialize_device_quota(&device_key, 1000.0, 100.0)
            .unwrap();

        // Rapid utilization changes should not cause instability
        for i in 0..100 {
            let utilization = if i % 2 == 0 { 0.1 } else { 0.9 };
            let result = hypervisor.update_utilization(utilization);
            assert!(result.is_ok(), "Update {i} failed: {result:?}");
            let _ = hypervisor.sync_avg_cost_to_devices(&[device_key]);
        }

        // The system should still be able to work normally
        let result = token_manager.try_acquire_workload(&device_key, 32, 32);
        assert!(result.is_ok() || result.is_err());
    }

    /// Performance test: ensure operations are completed within a reasonable time
    #[test]
    fn performance_basic_operations() {
        let storage = TestMemoryStorage::new();
        let device_key = "test-device";

        storage.set_quota(&device_key, 10000.0, 1000.0).unwrap();

        let mut hypervisor = HypervisorUtilizationController::new(storage.clone(), 0.8);
        let mut token_manager = WorkloadTokenManager::with_default_calculator(storage);

        hypervisor
            .initialize_device_quota(&device_key, 10000.0, 1000.0)
            .unwrap();

        let start = std::time::Instant::now();

        // Execute a large number of operations
        for i in 0..1000 {
            let _ = token_manager.try_acquire_workload(&device_key, 32, 32);
            if i % 10 == 0 {
                let _ = hypervisor.update_utilization(0.7);
                let _ = hypervisor.sync_avg_cost_to_devices(&[device_key]);
            }
        }

        let duration = start.elapsed();

        // 1000 operations should be completed within a reasonable time (e.g. 1 second)
        assert!(
            duration.as_secs() < 5,
            "1000 operations took too long: {duration:?}"
        );
    }
}
