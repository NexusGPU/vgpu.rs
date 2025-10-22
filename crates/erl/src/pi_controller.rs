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
}

impl Default for PIParams {
    fn default() -> Self {
        Self {
            kp: 0.5,
            ki: 0.1,
            min_rate: 0.1,
            max_rate: 100.0,
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
}

impl CongestionController for PIController {
    fn update(
        &mut self,
        current_utilization: f64,
        target_utilization: f64,
        _delta_time: f64,
    ) -> Result<f64, ErlError> {
        // Calculate error (positive = need more rate, negative = need less rate)
        let error = target_utilization - current_utilization;

        // Update integral term
        self.integral += error;

        // PI control law
        let adjustment = self.params.kp * error + self.params.ki * self.integral;

        // Apply adjustment (multiplicative)
        self.current_rate *= (1.0 + adjustment).clamp(0.5, 2.0);

        // Clamp to valid range
        self.current_rate = self
            .current_rate
            .clamp(self.params.min_rate, self.params.max_rate);

        tracing::debug!(
            current_utilization = current_utilization,
            target_utilization = target_utilization,
            error = error,
            integral = self.integral,
            adjustment = adjustment,
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
}
