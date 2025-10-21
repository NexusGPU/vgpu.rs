//! Support Workload-aware CUBIC congestion control algorithm
//!
//! Integrate CostTracker to correct feedback, keep CUBIC algorithm accurate

use crate::cost_tracker::CostTracker;
use crate::state::{CongestionState, StateMachineContext};
use crate::traits::{CongestionController, ErlError};
use crate::workload_calc::WorkloadCalculator;
use error_stack::Result;

/// CUBIC congestion control parameters
#[derive(Debug, Clone)]
pub struct CubicParams {
    /// CUBIC constant C, control the growth speed of the cubic function
    pub c: f64,
    /// Multiplicative decrease factor β, used to enter recovery quickly
    pub beta: f64,
    /// Growth factor for slow start
    pub slow_start_factor: f64,
    /// Minimum average cost
    pub min_avg_cost: f64,
    /// Maximum average cost
    pub max_avg_cost: f64,
    /// Conservative mode parameters (similar to traditional linear growth)
    pub conservative_mode: bool,
}

impl Default for CubicParams {
    fn default() -> Self {
        Self {
            c: 0.4,                  // Empirical value, balance convergence speed and stability
            beta: 1.5, // Quickly increase avg_cost (reduce launch rate) during recovery
            slow_start_factor: 1.2, // Multiplicative factor for slow start
            min_avg_cost: 0.001, // Allow very high throughput for workloads with small kernels
            max_avg_cost: 10.0, // Lower max to prevent excessive throttling
            conservative_mode: true, // Enable conservative mode
        }
    }
}

/// Workload-aware CUBIC controller
#[derive(Debug)]
pub struct WorkloadAwareCubicController {
    /// CUBIC parameters
    params: CubicParams,
    /// State machine context
    state_context: StateMachineContext,
    /// Current average cost
    avg_cost: f64,
    /// Maximum avg_cost before the last congestion event (W_max)
    w_max: f64,
    /// Congestion event time (used to calculate K)
    congestion_epoch: f64,
    /// Last update time
    last_update_time: f64,
    /// Conservative mode cost (used for mixed mode)
    conservative_cost: f64,
    /// Whether just recovered from congestion
    just_recovered: bool,
    /// workload calculator
    workload_calculator: Box<dyn WorkloadCalculator>,
    /// Cost tracker (used to correct feedback)
    cost_tracker: CostTracker,
    /// Congestion avoidance alpha (smoothing factor)
    congestion_alpha: f64,
    /// Adjustment threshold
    adjustment_threshold: f64,
    /// Adjustment coefficient
    adjustment_coefficient: f64,
}

impl WorkloadAwareCubicController {
    /// Create new workload-aware CUBIC controller
    ///
    /// # Arguments
    ///
    /// * `initial_avg_cost` - Initial average cost
    /// * `params` - CUBIC parameters
    /// * `current_time` - Current timestamp (seconds)
    /// * `workload_calculator` - Workload calculator
    pub fn new(
        initial_avg_cost: f64,
        params: CubicParams,
        current_time: f64,
        workload_calculator: Box<dyn WorkloadCalculator>,
    ) -> Self {
        let state_context = StateMachineContext::new(initial_avg_cost * 10.0, current_time);
        let min_cost = params.min_avg_cost;
        let max_cost = params.max_avg_cost;

        Self {
            params,
            state_context,
            avg_cost: initial_avg_cost.clamp(min_cost, max_cost),
            w_max: initial_avg_cost,
            congestion_epoch: current_time,
            last_update_time: current_time,
            conservative_cost: initial_avg_cost,
            just_recovered: false,
            workload_calculator,
            cost_tracker: CostTracker::with_defaults(),
            congestion_alpha: 0.5, // Default values
            adjustment_threshold: 0.01,
            adjustment_coefficient: 1.0,
        }
    }

    /// Create CUBIC controller with default parameters
    pub fn with_defaults(
        initial_avg_cost: f64,
        current_time: f64,
        workload_calculator: Box<dyn WorkloadCalculator>,
    ) -> Self {
        Self::new(
            initial_avg_cost,
            CubicParams::default(),
            current_time,
            workload_calculator,
        )
    }

    /// Create CUBIC controller with custom configuration
    pub fn with_config(
        initial_avg_cost: f64,
        params: CubicParams,
        current_time: f64,
        workload_calculator: Box<dyn WorkloadCalculator>,
        congestion_alpha: f64,
        adjustment_threshold: f64,
        adjustment_coefficient: f64,
    ) -> Self {
        let state_context = StateMachineContext::new(initial_avg_cost * 10.0, current_time);
        let min_cost = params.min_avg_cost;
        let max_cost = params.max_avg_cost;

        Self {
            params,
            state_context,
            avg_cost: initial_avg_cost.clamp(min_cost, max_cost),
            w_max: initial_avg_cost,
            congestion_epoch: current_time,
            last_update_time: current_time,
            conservative_cost: initial_avg_cost,
            just_recovered: false,
            workload_calculator,
            cost_tracker: CostTracker::with_defaults(),
            congestion_alpha,
            adjustment_threshold,
            adjustment_coefficient,
        }
    }

    /// Calculate dynamic cost for specific workload
    ///
    /// # Arguments
    ///
    /// * `grid_count` - Number of blocks in the grid
    /// * `block_count` - Number of threads in each block
    ///
    /// # Returns
    ///
    /// Return the cost value that should be deducted for this workload
    pub fn calculate_workload_cost(&mut self, grid_count: u32, block_count: u32) -> f64 {
        let workload_factor = self
            .workload_calculator
            .calculate_factor(grid_count, block_count);
        let dynamic_cost = self.avg_cost * workload_factor;

        // Record cost for correcting CUBIC feedback
        self.cost_tracker.record_cost(dynamic_cost, self.avg_cost);

        tracing::debug!(
            calculator = %self.workload_calculator.name(),
            grid_count = grid_count,
            block_count = block_count,
            workload_factor = workload_factor,
            avg_cost = self.avg_cost,
            dynamic_cost = dynamic_cost,
            "Calculated workload cost"
        );

        dynamic_cost
    }

    /// Get cost tracker statistics
    pub fn cost_tracker_stats(&self) -> crate::cost_tracker::CostTrackerStats {
        self.cost_tracker.stats()
    }

    /// CUBIC function calculation
    fn cubic_function(&self, current_time: f64) -> f64 {
        let time_since_congestion = current_time - self.congestion_epoch;
        let k = (self.w_max / self.params.c).powf(1.0 / 3.0);
        let cubic_cost = self.params.c * (time_since_congestion - k).powi(3) + self.w_max;
        cubic_cost.clamp(self.params.min_avg_cost, self.params.max_avg_cost)
    }

    /// Conservative mode calculation
    fn conservative_function(&self, current_time: f64, time_scale: f64) -> f64 {
        let time_since_congestion = current_time - self.congestion_epoch;
        let growth = time_since_congestion / (time_scale * self.conservative_cost);
        let new_cost = self.conservative_cost + growth;
        new_cost.clamp(self.params.min_avg_cost, self.params.max_avg_cost)
    }

    /// Slow start update
    fn slow_start_update(&mut self, utilization_error: f64) {
        if utilization_error < 0.0 {
            // More aggressive decrease when utilization is below target
            let decrease_factor = if utilization_error < -0.4 {
                // Zero or near-zero utilization: extremely aggressive decrease
                2.0
            } else if utilization_error < -0.2 {
                // Very low utilization: aggressive decrease
                1.5
            } else if utilization_error < -0.1 {
                // Low utilization: moderate decrease
                1.3
            } else {
                // Slightly below target: gentle decrease
                self.params.slow_start_factor
            };
            self.avg_cost /= decrease_factor;

            tracing::debug!(
                avg_cost = self.avg_cost,
                utilization_error = utilization_error,
                decrease_factor = decrease_factor,
                "Slow start: decreasing cost"
            );
        } else {
            self.avg_cost *= 1.0 + utilization_error * 0.2;
            tracing::debug!(
                avg_cost = self.avg_cost,
                utilization_error = utilization_error,
                "Slow start: increasing cost"
            );
        }

        self.avg_cost = self
            .avg_cost
            .clamp(self.params.min_avg_cost, self.params.max_avg_cost);
    }

    /// Congestion avoidance update
    fn congestion_avoidance_update(&mut self, current_time: f64, utilization_error: f64) {
        // When utilization is far below target, aggressively decrease cost
        // instead of following cubic function (which grows over time)
        if utilization_error < -0.2 {
            // Direct multiplicative decrease, similar to slow start
            let decrease_factor = if utilization_error < -0.3 {
                1.3 // Very low utilization, decrease aggressively
            } else {
                1.15 // Moderately low, decrease moderately
            };
            self.avg_cost /= decrease_factor;

            tracing::debug!(
                avg_cost = self.avg_cost,
                utilization_error = utilization_error,
                decrease_factor = decrease_factor,
                "Congestion avoidance: aggressive decrease due to low utilization"
            );
        } else {
            // Normal congestion avoidance: follow cubic function
            let cubic_cost = self.cubic_function(current_time);

            let target_cost = if self.params.conservative_mode {
                let time_scale = 0.5;
                let conservative_cost = self.conservative_function(current_time, time_scale);
                cubic_cost.max(conservative_cost)
            } else {
                cubic_cost
            };

            // Use configured alpha for faster response to utilization changes
            let alpha = self.congestion_alpha;
            self.avg_cost = self.avg_cost * (1.0 - alpha) + target_cost * alpha;

            // More aggressive adjustment when far from target
            if utilization_error.abs() > self.adjustment_threshold {
                let adjustment = 1.0 + utilization_error * self.adjustment_coefficient;
                self.avg_cost *= adjustment;
            }

            tracing::debug!(
                avg_cost = self.avg_cost,
                cubic_cost = cubic_cost,
                utilization_error = utilization_error,
                "Congestion avoidance update"
            );
        }

        self.avg_cost = self
            .avg_cost
            .clamp(self.params.min_avg_cost, self.params.max_avg_cost);
    }

    /// Recovery phase update
    fn recovery_update(&mut self, current_time: f64, utilization_error: f64) {
        if !self.just_recovered {
            self.w_max = self.avg_cost;
            self.avg_cost *= self.params.beta;
            self.congestion_epoch = current_time;
            self.just_recovered = true;

            tracing::warn!(
                new_avg_cost = self.avg_cost,
                w_max = self.w_max,
                "Entered recovery phase, multiplicative decrease applied"
            );
        } else if utilization_error > 0.01 {
            self.avg_cost *= 1.2;
        } else if utilization_error < -0.01 {
            self.avg_cost *= 0.9;
        }

        self.avg_cost = self
            .avg_cost
            .clamp(self.params.min_avg_cost, self.params.max_avg_cost);

        tracing::debug!(
            avg_cost = self.avg_cost,
            utilization_error = utilization_error,
            just_recovered = self.just_recovered,
            "Recovery update"
        );
    }

    /// Handle state transition
    fn handle_state_transition(&mut self, old_state: CongestionState, new_state: CongestionState) {
        match (old_state, new_state) {
            (CongestionState::Recovery, _) => {
                self.just_recovered = false;
                tracing::info!(
                    new_state = %new_state,
                    "Exited recovery phase"
                );
            }
            (_, CongestionState::Recovery) => {
                tracing::info!("Entering recovery phase");
            }
            _ => {}
        }
    }
}

impl CongestionController for WorkloadAwareCubicController {
    fn update(
        &mut self,
        current_utilization: f64,
        target_utilization: f64,
        delta_time: f64,
    ) -> Result<f64, ErlError> {
        let current_time = self.last_update_time + delta_time;

        let adjusted_utilization = if self.cost_tracker.has_sufficient_data() {
            self.cost_tracker
                .adjust_utilization(current_utilization, target_utilization)
        } else {
            current_utilization
        };

        let utilization_error = adjusted_utilization - target_utilization;

        // Update state machine
        let old_state = self.state_context.current_state;
        let state_changed =
            self.state_context
                .update(utilization_error, self.avg_cost, current_time);

        if state_changed {
            self.handle_state_transition(old_state, self.state_context.current_state);
        }

        // Execute corresponding update logic based on current state
        match self.state_context.current_state {
            CongestionState::SlowStart => {
                self.slow_start_update(utilization_error);
            }
            CongestionState::CongestionAvoidance => {
                self.congestion_avoidance_update(current_time, utilization_error);
            }
            CongestionState::Recovery => {
                self.recovery_update(current_time, utilization_error);
            }
        }

        // Update conservative mode cost
        if self.params.conservative_mode {
            self.conservative_cost = self.conservative_function(current_time, 0.5);
        }

        self.last_update_time = current_time;

        let cost_stats = self.cost_tracker.stats();

        tracing::info!(
            state = %self.state_context.current_state,
            avg_cost = self.avg_cost,
            raw_utilization = current_utilization,
            adjusted_utilization = adjusted_utilization,
            utilization_error = utilization_error,
            target_utilization = target_utilization,
            cost_tracker = %cost_stats,
            "CUBIC controller updated with workload feedback correction"
        );

        Ok(self.avg_cost)
    }

    fn current_avg_cost(&self) -> f64 {
        self.avg_cost
    }
}
