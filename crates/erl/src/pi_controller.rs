//! Simple PI (Proportional-Integral) controller for GPU utilization
//!
//! Provides a simple, robust control algorithm with only 2 parameters.

use crate::traits::{CongestionController, ErlError};
use error_stack::Result;

/// PI controller parameters
#[derive(Debug, Clone)]
pub struct PIParams {
    /// Proportional gain (how aggressively to respond to current error)
    pub kp: f64,
    /// Integral gain (how to respond to accumulated error over time)
    pub ki: f64,
    /// Minimum refill rate
    pub min_rate: f64,
    /// Maximum refill rate
    pub max_rate: f64,
    /// Bootstrap: aggressive growth multiplier when far from target
    pub bootstrap_aggressive_factor: f64,
    /// Bootstrap: conservative growth multiplier when near saturation
    pub bootstrap_conservative_factor: f64,
    /// Bootstrap: decrease multiplier when over target
    pub bootstrap_decrease_factor: f64,
    /// Bootstrap: utilization threshold to switch from aggressive to conservative
    pub bootstrap_saturation_threshold: f64,
}

impl Default for PIParams {
    fn default() -> Self {
        Self {
            // PI control parameters
            kp: 1.0, // Proportional gain: 1.0 = 100% response to current error
            ki: 0.2, // Integral gain: 0.2 = 20% response to accumulated error
            min_rate: 0.1,
            max_rate: 10000.0,

            // Bootstrap parameters
            bootstrap_aggressive_factor: 2.0, // Double rate when far from target
            bootstrap_conservative_factor: 1.2, // 20% increase when near saturation
            bootstrap_decrease_factor: 0.8,   // 20% decrease when over target
            bootstrap_saturation_threshold: 0.8, // 80% utilization = near saturation
        }
    }
}

/// Simple PI controller for per-pod utilization control
#[derive(Debug)]
pub struct PIController {
    /// Controller parameters
    params: PIParams,
    /// Current refill rate
    current_rate: f64,
    /// Accumulated integral error
    integral: f64,
    /// Calibration state: estimated rate needed for 100% utilization
    /// None during bootstrap phase
    calibrated_max_rate: Option<f64>,
    /// Bootstrap samples collected
    bootstrap_samples: Vec<(f64, f64)>, // (rate, utilization)
}

impl PIController {
    /// Create new PI controller
    pub fn new(initial_rate: f64, params: PIParams) -> Self {
        let min_rate = params.min_rate;
        let max_rate = params.max_rate;
        Self {
            params,
            current_rate: initial_rate.clamp(min_rate, max_rate),
            integral: 0.0,
            calibrated_max_rate: None,
            bootstrap_samples: Vec::new(),
        }
    }

    /// Create PI controller with default parameters
    pub fn with_defaults(initial_rate: f64) -> Self {
        Self::new(initial_rate, PIParams::default())
    }

    /// Reset integral term (useful when changing target)
    pub fn reset_integral(&mut self) {
        self.integral = 0.0;
    }

    /// Check if controller is in bootstrap phase
    pub fn is_bootstrapping(&self) -> bool {
        self.calibrated_max_rate.is_none()
    }

    /// Calibrate based on collected samples
    fn calibrate(&mut self) {
        if self.bootstrap_samples.len() < 3 {
            return;
        }

        // Find the highest utilization achieved and its corresponding rate
        let (max_util_rate, max_util) = self
            .bootstrap_samples
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied()
            .unwrap_or((self.current_rate, 0.01));

        if max_util < 0.05 {
            // Not enough data yet
            return;
        }

        // Estimate rate needed for 100% utilization using linear extrapolation
        // Example: if rate=500 gives utilization=0.25, then rate=2000 should give 1.0
        let estimated_full_rate = max_util_rate / max_util;

        self.calibrated_max_rate = Some(estimated_full_rate);

        tracing::info!(
            max_util = max_util,
            max_util_rate = max_util_rate,
            estimated_full_rate = estimated_full_rate,
            samples = self.bootstrap_samples.len(),
            "PI controller calibrated based on observed behavior"
        );
    }
}

impl CongestionController for PIController {
    fn update(
        &mut self,
        current_utilization: f64,
        target_utilization: f64,
        _delta_time: f64,
    ) -> Result<f64, ErlError> {
        // Bootstrap phase: aggressively increase rate to find operating range
        if self.is_bootstrapping() {
            // Record sample
            self.bootstrap_samples
                .push((self.current_rate, current_utilization));

            // Adjust rate based on utilization
            if current_utilization > target_utilization {
                // Already over target, decrease rate
                self.current_rate *= self.params.bootstrap_decrease_factor;
            } else if current_utilization < self.params.bootstrap_saturation_threshold {
                // Still have headroom, aggressive growth
                self.current_rate *= self.params.bootstrap_aggressive_factor;
            } else {
                // Getting close to saturation, conservative growth
                self.current_rate *= self.params.bootstrap_conservative_factor;
            }

            self.current_rate = self
                .current_rate
                .clamp(self.params.min_rate, self.params.max_rate);

            // Try to calibrate after collecting samples
            if self.bootstrap_samples.len() >= 5 {
                self.calibrate();
            }

            tracing::info!(
                current_utilization = current_utilization,
                target_utilization = target_utilization,
                new_rate = self.current_rate,
                samples = self.bootstrap_samples.len(),
                bootstrapping = true,
                "PI controller in bootstrap phase"
            );

            return Ok(self.current_rate);
        }

        // Normal PI control after calibration
        let error = target_utilization - current_utilization;
        self.integral += error;

        let adjustment = self.params.kp * error + self.params.ki * self.integral;

        // Use calibrated max rate if available
        let effective_max = self
            .calibrated_max_rate
            .unwrap_or(self.params.max_rate)
            .min(self.params.max_rate);

        // Calculate target rate based on calibration
        // If calibrated_max_rate = 2000 for 100% util, and we want 50%, target = 1000
        let target_rate = effective_max * target_utilization;

        // PI adjustment around target
        let multiplier = (1.0 + adjustment).clamp(0.5, 2.0);
        self.current_rate *= multiplier;

        // Clamp to valid range
        self.current_rate = self.current_rate.clamp(self.params.min_rate, effective_max);

        tracing::debug!(
            current_utilization = current_utilization,
            target_utilization = target_utilization,
            error = error,
            integral = self.integral,
            adjustment = adjustment,
            calibrated_max_rate = self.calibrated_max_rate,
            target_rate = target_rate,
            new_rate = self.current_rate,
            "PI controller updated"
        );

        Ok(self.current_rate)
    }

    fn current_avg_cost(&self) -> f64 {
        self.current_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pi_controller_increases_rate_when_under_target() {
        let mut controller = PIController::with_defaults(1.0);

        let initial_rate = controller.current_avg_cost();

        // Utilization is below target
        controller
            .update(0.3, 0.5, 1.0)
            .expect("should update successfully");

        let new_rate = controller.current_avg_cost();

        assert!(
            new_rate > initial_rate,
            "Rate should increase when utilization is below target"
        );
    }

    #[test]
    fn pi_controller_decreases_rate_when_over_target() {
        let mut controller = PIController::with_defaults(10.0);

        let initial_rate = controller.current_avg_cost();

        // Utilization is above target
        controller
            .update(0.8, 0.5, 1.0)
            .expect("should update successfully");

        let new_rate = controller.current_avg_cost();

        assert!(
            new_rate < initial_rate,
            "Rate should decrease when utilization is above target"
        );
    }

    #[test]
    fn pi_controller_respects_bounds() {
        let params = PIParams {
            kp: 0.5,
            ki: 0.1,
            min_rate: 1.0,
            max_rate: 10.0,
            ..Default::default()
        };

        let mut controller = PIController::new(5.0, params);

        // Try to drive it very high
        for _ in 0..100 {
            controller
                .update(0.1, 0.9, 1.0)
                .expect("should update successfully");
        }

        assert!(
            controller.current_avg_cost() <= 10.0,
            "Should respect max_rate"
        );

        // Try to drive it very low
        for _ in 0..100 {
            controller
                .update(0.9, 0.1, 1.0)
                .expect("should update successfully");
        }

        assert!(
            controller.current_avg_cost() >= 1.0,
            "Should respect min_rate"
        );
    }

    #[test]
    fn pi_controller_converges_near_target() {
        let mut controller = PIController::with_defaults(1.0);

        // Simulate steady state near target
        for _ in 0..10 {
            controller
                .update(0.49, 0.5, 1.0)
                .expect("should update successfully");
        }

        // Rate should stabilize (not oscillate wildly)
        let rate1 = controller.current_avg_cost();

        controller
            .update(0.49, 0.5, 1.0)
            .expect("should update successfully");

        let rate2 = controller.current_avg_cost();

        let change = ((rate2 - rate1) / rate1).abs();
        assert!(
            change < 0.1,
            "Rate should stabilize near target, change was {change}"
        );
    }

    #[test]
    fn bootstrap_converges_from_very_low_rate() {
        let mut controller = PIController::with_defaults(1.0);
        let target = 0.5;

        // Simulate a workload where rate=1.0 gives only 0.01 utilization
        // This means we need rate=50.0 for 50% utilization
        let actual_full_rate = 50.0;

        let mut utilizations = Vec::new();

        // Bootstrap phase: should aggressively increase rate
        for i in 0..20 {
            let current_rate = controller.current_avg_cost();
            // Simulate real utilization based on rate
            let sim_util = (current_rate / actual_full_rate).min(1.0);

            controller
                .update(sim_util, target, 1.0)
                .expect("should update successfully");

            utilizations.push(sim_util);

            tracing::info!(
                iteration = i,
                rate = current_rate,
                utilization = sim_util,
                calibrated = controller.calibrated_max_rate.is_some(),
                "Bootstrap test iteration"
            );

            // After calibration, should converge quickly
            if controller.calibrated_max_rate.is_some() && i > 10 {
                // Check convergence within 20% error
                assert!(
                    (sim_util - target).abs() < 0.2,
                    "Should converge to target after calibration, got {sim_util}"
                );
            }
        }

        // Final rate should be close to 50% of actual_full_rate
        let final_rate = controller.current_avg_cost();
        let expected_rate = actual_full_rate * target;
        let rate_error = (final_rate - expected_rate).abs() / expected_rate;

        assert!(
            rate_error < 0.3,
            "Final rate {final_rate} should be close to expected {expected_rate}, error: {rate_error}"
        );
    }

    #[test]
    fn bootstrap_handles_very_high_initial_rate() {
        let mut controller = PIController::with_defaults(1000.0);
        let target = 0.5;

        // Simulate a workload where rate=1000.0 gives 100% utilization
        // This means we need rate=500.0 for 50% utilization
        let actual_full_rate = 1000.0;

        // Bootstrap phase: should decrease rate quickly
        for i in 0..20 {
            let current_rate = controller.current_avg_cost();
            let sim_util = (current_rate / actual_full_rate).min(1.0);

            controller
                .update(sim_util, target, 1.0)
                .expect("should update successfully");

            tracing::info!(
                iteration = i,
                rate = current_rate,
                utilization = sim_util,
                "High initial rate test iteration"
            );

            // Should not stay at 100% utilization for long
            if i > 3 {
                assert!(
                    sim_util < 0.95,
                    "Should decrease from 100% utilization quickly, at iteration {i}, util={sim_util}"
                );
            }
        }

        // Final rate should converge
        let final_rate = controller.current_avg_cost();
        let expected_rate = actual_full_rate * target;
        let rate_error = (final_rate - expected_rate).abs() / expected_rate;

        assert!(
            rate_error < 0.3,
            "Final rate {final_rate} should converge to expected {expected_rate}"
        );
    }

    #[test]
    fn bootstrap_handles_different_workload_scales() {
        // Test with a very heavy workload (need high rates)
        let heavy_workload_rate = 10000.0;
        test_workload_convergence(100.0, heavy_workload_rate, 0.5);

        // Test with a light workload (need low rates)
        let light_workload_rate = 10.0;
        test_workload_convergence(100.0, light_workload_rate, 0.5);

        // Test with medium workload
        let medium_workload_rate = 500.0;
        test_workload_convergence(100.0, medium_workload_rate, 0.5);
    }

    fn test_workload_convergence(initial_rate: f64, actual_full_rate: f64, target: f64) {
        let mut controller = PIController::with_defaults(initial_rate);

        for i in 0..30 {
            let current_rate = controller.current_avg_cost();
            let sim_util = (current_rate / actual_full_rate).min(1.0);

            controller
                .update(sim_util, target, 1.0)
                .expect("should update successfully");

            // After calibration and some convergence iterations
            if controller.calibrated_max_rate.is_some() && i > 15 {
                // Should be reasonably close to target
                let error = (sim_util - target).abs();
                assert!(
                    error < 0.25,
                    "Workload scale {actual_full_rate}: Should converge to {target}, got {sim_util} at iteration {i}"
                );
            }
        }

        let final_rate = controller.current_avg_cost();
        let final_util = (final_rate / actual_full_rate).min(1.0);
        let util_error = (final_util - target).abs();

        assert!(
            util_error < 0.2,
            "Workload scale {actual_full_rate}: Final utilization {final_util} should be close to target {target}"
        );
    }

    #[test]
    fn bootstrap_with_noisy_utilization() {
        let mut controller = PIController::with_defaults(10.0);
        let target = 0.5;
        let actual_full_rate = 100.0;

        // Simulate noisy measurements (±20% variation)
        let noise_factors = [1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.85, 1.15];

        for i in 0..30 {
            let current_rate = controller.current_avg_cost();
            let base_util = (current_rate / actual_full_rate).min(1.0);
            // Add noise
            let noise = noise_factors[i % noise_factors.len()];
            let sim_util = (base_util * noise).min(1.0);

            controller
                .update(sim_util, target, 1.0)
                .expect("should update successfully");

            // Even with noise, should eventually converge
            if i > 20 {
                // Average utilization should be close to target
                // (we're more lenient with noise)
                assert!(
                    (base_util - target).abs() < 0.3,
                    "Should handle noisy measurements and converge, iteration {i}, base_util={base_util}"
                );
            }
        }
    }

    #[test]
    fn bootstrap_with_changing_workload() {
        let mut controller = PIController::with_defaults(10.0);
        let target = 0.5;

        // Workload changes halfway through
        let initial_full_rate = 50.0;
        let changed_full_rate = 200.0;

        for i in 0..40 {
            let current_rate = controller.current_avg_cost();

            // Workload changes at iteration 20
            let actual_full_rate = if i < 20 {
                initial_full_rate
            } else {
                changed_full_rate
            };

            let sim_util = (current_rate / actual_full_rate).min(1.0);

            controller
                .update(sim_util, target, 1.0)
                .expect("should update successfully");

            tracing::info!(
                iteration = i,
                rate = current_rate,
                utilization = sim_util,
                workload_rate = actual_full_rate,
                "Changing workload test"
            );
        }

        // Should adapt to new workload
        let final_rate = controller.current_avg_cost();
        let final_util = (final_rate / changed_full_rate).min(1.0);

        // May not be perfectly converged due to workload change, but should be reasonable
        assert!(
            (final_util - target).abs() < 0.4,
            "Should adapt to changed workload, final_util={final_util}"
        );
    }

    #[test]
    fn multiple_targets_all_converge() {
        let targets = [0.3, 0.5, 0.7, 0.9];
        let actual_full_rate = 1000.0;

        for target in targets {
            let mut controller = PIController::with_defaults(50.0);

            for i in 0..30 {
                let current_rate = controller.current_avg_cost();
                let sim_util = (current_rate / actual_full_rate).min(1.0);

                controller
                    .update(sim_util, target, 1.0)
                    .expect("should update successfully");

                if controller.calibrated_max_rate.is_some() && i > 15 {
                    assert!(
                        (sim_util - target).abs() < 0.25,
                        "Target {target}: Should converge, got {sim_util} at iteration {i}"
                    );
                }
            }

            let final_rate = controller.current_avg_cost();
            let final_util = (final_rate / actual_full_rate).min(1.0);
            let error = (final_util - target).abs();

            assert!(
                error < 0.15,
                "Target {target}: Final utilization {final_util} should match target, error={error}"
            );
        }
    }
}
