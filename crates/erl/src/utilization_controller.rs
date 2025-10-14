//! Utilization controller implementation
//!
//! Execute utilization monitoring, target setting and congestion control in hypervisor process

use error_stack::Result;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cubic::{CubicParams, WorkloadAwareCubicController};
use crate::traits::{CongestionController, ErlError, SharedStorage, UtilizationController};
use crate::workload_calc::PowerWorkloadCalculator;

/// Hypervisor utilization controller
///
/// Run in hypervisor process, responsible for:
/// 1. Collect GPU utilization
/// 2. Run congestion control algorithm
/// 3. Write new avg_cost to shared memory
/// 4. Manage device quota
#[derive(Debug)]
pub struct HypervisorUtilizationController<K, S, C>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync,
    S: SharedStorage<K>,
    C: CongestionController,
{
    /// Shared storage
    storage: S,
    /// Congestion controller
    congestion_controller: C,
    /// Target utilization
    target_utilization: f64,
    /// Last update time
    last_update_time: f64,
    /// Type marker
    _phantom: std::marker::PhantomData<K>,
}

impl<K, S> HypervisorUtilizationController<K, S, WorkloadAwareCubicController>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync,
    S: SharedStorage<K>,
{
    /// Create new utilization controller (using default CUBIC algorithm)
    pub fn new(storage: S, target_utilization: f64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Use higher initial avg_cost to prevent over-launching
        // With lower values (e.g., 1.0), CUBIC may converge to a too-low cost,
        // causing utilization to exceed target. Starting higher and letting it
        // decrease is safer than starting low and trying to increase.
        let initial_avg_cost = 10.0;

        Self {
            storage,
            congestion_controller: WorkloadAwareCubicController::with_defaults(
                initial_avg_cost,                             // Initial avg_cost
                now,                                          // Current time
                Box::new(PowerWorkloadCalculator::default()), // Used for cost tracking
            ),
            target_utilization,
            last_update_time: now,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create new utilization controller with custom CUBIC parameters
    pub fn with_custom_params(
        storage: S,
        target_utilization: f64,
        initial_avg_cost: f64,
        cubic_params: CubicParams,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        Self {
            storage,
            congestion_controller: WorkloadAwareCubicController::new(
                initial_avg_cost,
                cubic_params,
                now,
                Box::new(PowerWorkloadCalculator::default()),
            ),
            target_utilization,
            last_update_time: now,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<K, S, C> HypervisorUtilizationController<K, S, C>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync,
    S: SharedStorage<K>,
    C: CongestionController,
{
    /// Create new utilization controller with custom congestion controller
    pub fn with_controller(storage: S, target_utilization: f64, congestion_controller: C) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        Self {
            storage,
            congestion_controller,
            target_utilization,
            last_update_time: now,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<K, S, C> UtilizationController<K> for HypervisorUtilizationController<K, S, C>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + std::fmt::Debug,
    S: SharedStorage<K>,
    C: CongestionController,
{
    type Storage = S;

    fn update_utilization(&mut self, utilization: f64) -> Result<(), ErlError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let delta_time = now - self.last_update_time;

        // Update congestion controller
        let new_avg_cost = self
            .congestion_controller
            .update(utilization, self.target_utilization, delta_time)
            .map_err(|e| {
                error_stack::report!(ErlError::CongestionControlFailed {
                    reason: format!("Congestion controller update failed: {e}")
                })
            })?;

        self.last_update_time = now;

        tracing::info!(
            utilization = utilization,
            target_utilization = self.target_utilization,
            new_avg_cost = new_avg_cost,
            delta_time = delta_time,
            "Utilization updated in hypervisor"
        );

        Ok(())
    }

    fn target_utilization(&self) -> f64 {
        self.target_utilization
    }

    fn set_target_utilization(&mut self, target: f64) -> Result<(), ErlError> {
        if !(0.0..=1.0).contains(&target) {
            return Err(error_stack::report!(ErlError::InvalidConfiguration {
                reason: format!("Target utilization must be between 0.0 and 1.0, got {target}")
            }));
        }

        self.target_utilization = target;

        tracing::info!(
            new_target = target,
            "Target utilization updated in hypervisor"
        );

        Ok(())
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

        // Initialize avg_cost
        let initial_avg_cost = self.congestion_controller.current_avg_cost();
        self.storage.save_avg_cost(key, initial_avg_cost)?;

        tracing::info!(
            key = ?key,
            capacity = capacity,
            refill_rate = refill_rate,
            initial_avg_cost = initial_avg_cost,
            "Device quota initialized in hypervisor"
        );

        Ok(())
    }

    fn get_devices_overview(&self) -> Result<Vec<(K, f64, f64, f64)>, ErlError> {
        // This implementation needs storage to support enumerating all devices
        // For now, return empty list, implementation depends on SharedStorage capabilities
        tracing::warn!(
            "get_devices_overview not yet implemented - requires device enumeration support in SharedStorage"
        );
        Ok(vec![])
    }
}

/// Helper function specifically for hypervisor
impl<K, S, C> HypervisorUtilizationController<K, S, C>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + std::fmt::Debug,
    S: SharedStorage<K>,
    C: CongestionController,
{
    /// Periodically update all devices' avg_cost to shared memory
    ///
    /// Hypervisor should periodically call this method to write the new avg_cost calculated by the congestion controller to shared memory
    /// so that limiter process can read the latest cost value
    pub fn sync_avg_cost_to_devices(&mut self, device_keys: &[K]) -> Result<(), ErlError> {
        let current_avg_cost = self.congestion_controller.current_avg_cost();

        for key in device_keys {
            self.storage.save_avg_cost(key, current_avg_cost)?;
        }

        tracing::debug!(
            avg_cost = current_avg_cost,
            device_count = device_keys.len(),
            "Synced avg_cost to devices"
        );

        Ok(())
    }
}

/// CUBIC congestion controller specific functionality
impl<K, S> HypervisorUtilizationController<K, S, WorkloadAwareCubicController>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + std::fmt::Debug,
    S: SharedStorage<K>,
{
    /// Get CUBIC controller statistics
    pub fn get_cubic_stats(&self) -> crate::cost_tracker::CostTrackerStats {
        self.congestion_controller.cost_tracker_stats()
    }
}
