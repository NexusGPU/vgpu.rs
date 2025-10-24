//! Per-Pod utilization controller implementation
//!
//! Each Pod has independent PI controller to converge to its own limit

use error_stack::Result;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::pi_controller::{PIController, PIParams};
use crate::traits::{CongestionController, ErlError, SharedStorage, UtilizationController};

/// Per-Pod controller state
#[derive(Debug)]
struct PodControllerState<K> {
    /// Pod's target utilization (limit)
    target_utilization: f64,
    /// PI controller for this Pod
    controller: PIController,
    /// Last kernel count
    last_kernel_count: u64,
    /// Device key this Pod is using
    device_key: K,
    /// Exponential moving average of estimated utilization (for smoothing)
    utilization_ema: f64,
    /// EMA smoothing factor (0.0-1.0, higher = more responsive)
    ema_alpha: f64,
}

/// Hypervisor utilization controller with per-Pod control
///
/// Run in hypervisor process, responsible for:
/// 1. Collect Pod metrics (kernel counts)
/// 2. Estimate each Pod's GPU utilization based on global utilization
/// 3. Run independent PI controller for each Pod
/// 4. Update each Pod's refill_rate in shared memory
#[derive(Debug)]
pub struct HypervisorUtilizationController<K, S>
where
    K: std::fmt::Display + std::hash::Hash + Eq + Clone + Send + Sync,
    S: SharedStorage<K>,
{
    /// Shared storage
    storage: S,
    /// Per-Pod controllers
    pod_controllers: HashMap<String, PodControllerState<K>>,
    /// Last update time
    last_update_time: f64,
    /// Global GPU utilization (from NVML)
    last_global_utilization: f64,
}

impl<K, S> HypervisorUtilizationController<K, S>
where
    K: std::fmt::Display + std::hash::Hash + Eq + Clone + Send + Sync,
    S: SharedStorage<K>,
{
    /// Create new utilization controller
    pub fn new(storage: S) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        Self {
            storage,
            pod_controllers: HashMap::new(),
            last_update_time: now,
            last_global_utilization: 0.0,
        }
    }

    /// Create new utilization controller with custom PI parameters
    pub fn with_pi_params(storage: S, _params: PIParams) -> Self {
        // For now, just use default
        Self::new(storage)
    }

    /// Register a new Pod for control
    ///
    /// # Arguments
    ///
    /// * `pod_name` - Pod name
    /// * `device_key` - Device this Pod is using
    /// * `target_utilization` - Pod's GPU limit (0.0 - 1.0)
    /// * `initial_rate` - Initial refill rate
    pub fn register_pod(
        &mut self,
        pod_name: String,
        device_key: K,
        target_utilization: f64,
        initial_rate: f64,
    ) -> Result<(), ErlError> {
        if !(0.0..=1.0).contains(&target_utilization) {
            return Err(error_stack::report!(ErlError::InvalidConfiguration {
                reason: format!(
                    "Target utilization must be between 0.0 and 1.0, got {target_utilization}"
                )
            }));
        }

        let controller = PIController::with_defaults(initial_rate);

        self.pod_controllers.insert(
            pod_name.clone(),
            PodControllerState {
                target_utilization,
                controller,
                last_kernel_count: 0,
                device_key,
                utilization_ema: target_utilization, // Initialize to target
                ema_alpha: 0.3, // 30% new data, 70% history (smooth but responsive)
            },
        );

        tracing::info!(
            pod_name = pod_name,
            target_utilization = target_utilization,
            initial_rate = initial_rate,
            "Pod registered for control"
        );

        Ok(())
    }

    /// Unregister a Pod
    pub fn unregister_pod(&mut self, pod_name: &str) {
        self.pod_controllers.remove(pod_name);
        tracing::info!(pod_name = pod_name, "Pod unregistered");
    }

    /// Update control loop with global utilization
    ///
    /// Reads kernel counts directly from shared memory for each registered Pod.
    ///
    /// # Arguments
    ///
    /// * `global_utilization` - Global GPU utilization from NVML (0.0 - 1.0)
    pub fn update_control(&mut self, global_utilization: f64) -> Result<(), ErlError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let delta_time = now - self.last_update_time;
        self.last_update_time = now;
        self.last_global_utilization = global_utilization;

        // Read kernel counts from shared memory for all registered Pods
        let mut total_kernel_count: u64 = 0;
        let mut pod_kernel_counts: Vec<(String, u64)> = Vec::new();

        for (pod_name, state) in &self.pod_controllers {
            match self.storage.load_and_reset_kernel_count(&state.device_key) {
                Ok(count) => {
                    total_kernel_count += count;
                    pod_kernel_counts.push((pod_name.clone(), count));
                }
                Err(e) => {
                    tracing::warn!(
                        pod_name = pod_name,
                        error = %e,
                        "Failed to load kernel count from shared memory"
                    );
                }
            }
        }

        if total_kernel_count == 0 {
            tracing::trace!("No kernels launched, skipping control update");
            return Ok(());
        }

        tracing::debug!(
            global_utilization = global_utilization,
            total_kernel_count = total_kernel_count,
            pod_count = pod_kernel_counts.len(),
            "Updating per-Pod controllers"
        );

        // Update each Pod's controller
        for (pod_name, kernel_count) in pod_kernel_counts {
            if let Some(state) = self.pod_controllers.get_mut(&pod_name) {
                // Estimate this Pod's utilization based on its kernel count ratio
                let kernel_ratio = (kernel_count as f64) / (total_kernel_count as f64);
                let raw_estimated_utilization = global_utilization * kernel_ratio;

                // Apply exponential moving average for smoothing
                // EMA(t) = alpha * new_value + (1 - alpha) * EMA(t-1)
                state.utilization_ema = state.ema_alpha * raw_estimated_utilization
                    + (1.0 - state.ema_alpha) * state.utilization_ema;

                let smoothed_utilization = state.utilization_ema;

                // Update PI controller with smoothed value
                let new_rate = state
                    .controller
                    .update(smoothed_utilization, state.target_utilization, delta_time)
                    .map_err(|e| {
                        error_stack::report!(ErlError::CongestionControlFailed {
                            reason: format!("PI controller update failed for {pod_name}: {e}")
                        })
                    })?;

                // Update refill rate in shared memory
                self.storage.save_avg_cost(&state.device_key, new_rate)?;

                tracing::info!(
                    pod_name = pod_name,
                    kernel_count = kernel_count,
                    kernel_ratio = kernel_ratio,
                    raw_utilization = raw_estimated_utilization,
                    smoothed_utilization = smoothed_utilization,
                    target_utilization = state.target_utilization,
                    new_rate = new_rate,
                    "Pod controller updated"
                );

                state.last_kernel_count = kernel_count;
            }
        }

        Ok(())
    }

    /// Get current status of all Pods
    pub fn get_pods_status(&self) -> Vec<(String, f64, f64, f64)> {
        self.pod_controllers
            .iter()
            .map(|(name, state)| {
                (
                    name.clone(),
                    state.target_utilization,
                    state.controller.current_avg_cost(),
                    state.last_kernel_count as f64,
                )
            })
            .collect()
    }
}

impl<K, S> UtilizationController<K> for HypervisorUtilizationController<K, S>
where
    K: std::fmt::Display + std::hash::Hash + Eq + Clone + Send + Sync,
    S: SharedStorage<K>,
{
    type Storage = S;

    fn target_utilization(&self) -> f64 {
        // Return average target across all Pods
        if self.pod_controllers.is_empty() {
            return 0.0;
        }

        let sum: f64 = self
            .pod_controllers
            .values()
            .map(|s| s.target_utilization)
            .sum();

        sum / (self.pod_controllers.len() as f64)
    }

    fn initialize_device_quota(
        &mut self,
        key: &K,
        capacity: f64,
        refill_rate: f64,
    ) -> Result<(), ErlError> {
        // Set device quota
        self.storage.set_quota(key, capacity, refill_rate)?;

        // Initialize token state to full capacity
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        self.storage.save_token_state(key, capacity, now)?;

        // Initialize refill rate
        self.storage.save_avg_cost(key, refill_rate)?;

        tracing::info!(
            capacity = capacity,
            refill_rate = refill_rate,
            "Device quota initialized"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, RwLock};

    #[derive(Debug, Clone)]
    struct MockStorage {
        quotas: Arc<RwLock<HashMap<String, (f64, f64)>>>,
        token_states: Arc<RwLock<HashMap<String, (f64, f64)>>>,
        avg_costs: Arc<RwLock<HashMap<String, f64>>>,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                quotas: Arc::new(RwLock::new(HashMap::new())),
                token_states: Arc::new(RwLock::new(HashMap::new())),
                avg_costs: Arc::new(RwLock::new(HashMap::new())),
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

        fn load_avg_cost(&self, key: &String) -> Result<f64, ErlError> {
            self.avg_costs
                .read()
                .unwrap()
                .get(key)
                .copied()
                .ok_or_else(|| {
                    error_stack::report!(ErlError::MonitoringFailed {
                        reason: format!("Avg cost not found for {key}")
                    })
                })
        }

        fn save_avg_cost(&self, key: &String, avg_cost: f64) -> Result<(), ErlError> {
            self.avg_costs
                .write()
                .unwrap()
                .insert(key.clone(), avg_cost);
            Ok(())
        }

        fn increment_kernel_count(&self, _key: &String) -> Result<(), ErlError> {
            Ok(())
        }

        fn load_and_reset_kernel_count(&self, _key: &String) -> Result<u64, ErlError> {
            Ok(100)
        }
    }

    #[test]
    fn register_and_control_single_pod() {
        let storage = MockStorage::new();
        let mut controller = HypervisorUtilizationController::new(storage.clone());

        controller
            .register_pod("pod-a".to_string(), "gpu-0".to_string(), 0.5, 10.0)
            .expect("should register pod");

        // Pod is under-utilizing (estimated 0.3, target 0.5)
        controller.update_control(0.3).expect("should update");

        // Check that rate increased
        let rate = storage
            .load_avg_cost(&"gpu-0".to_string())
            .expect("should load avg cost");
        assert!(rate > 10.0, "Rate should increase when under target");
    }

    #[test]
    fn control_multiple_pods_independently() {
        let storage = MockStorage::new();
        let mut controller = HypervisorUtilizationController::new(storage.clone());

        controller
            .register_pod("pod-a".to_string(), "gpu-0".to_string(), 0.5, 10.0)
            .expect("should register pod-a");

        controller
            .register_pod("pod-b".to_string(), "gpu-1".to_string(), 0.3, 10.0)
            .expect("should register pod-b");

        // Global utilization is 0.8
        // With mock returning 100 for each pod's kernel count:
        // pod-a gets 0.5 * 0.8 = 0.4 (under its target of 0.5)
        // pod-b gets 0.5 * 0.8 = 0.4 (over its target of 0.3)
        controller.update_control(0.8).expect("should update");

        let rate_a = storage
            .load_avg_cost(&"gpu-0".to_string())
            .expect("should load rate for pod-a");
        let rate_b = storage
            .load_avg_cost(&"gpu-1".to_string())
            .expect("should load rate for pod-b");

        // With equal kernel counts, both are at 0.4 utilization
        // pod-a target 0.5, so under target -> rate should increase
        // pod-b target 0.3, so over target -> rate should decrease
        assert!(
            rate_a > 10.0,
            "pod-a rate should increase (under target 0.5): {rate_a}"
        );
        assert!(
            rate_b < 10.0,
            "pod-b rate should decrease (over target 0.3): {rate_b}"
        );
    }
}
