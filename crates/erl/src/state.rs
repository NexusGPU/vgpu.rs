//! Congestion control state machine
//!
//! Implements the three-state congestion control algorithm based on CUBIC.

use derive_more::Display;

/// Congestion control state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum CongestionState {
    /// Slow start phase: exponential growth, quickly increase launch rate
    #[display("SlowStart")]
    SlowStart,
    /// Congestion avoidance phase: cubic function growth, smooth approach to target
    #[display("CongestionAvoidance")]
    CongestionAvoidance,
    /// Recovery phase: when overload is detected, quickly decrease launch rate
    #[display("Recovery")]
    Recovery,
}

impl CongestionState {
    /// State transition logic
    ///
    /// # Arguments
    ///
    /// * `utilization_error` - Utilization error (actual - target)
    /// * `avg_cost` - Current average cost
    /// * `slow_start_threshold` - Slow start threshold
    ///
    /// # Returns
    ///
    /// New state
    pub fn transition(
        self,
        utilization_error: f64,
        avg_cost: f64,
        slow_start_threshold: f64,
    ) -> Self {
        match self {
            CongestionState::SlowStart => {
                if utilization_error > 0.0 {
                    // Utilization exceeds target, enter recovery phase
                    CongestionState::Recovery
                } else if avg_cost >= slow_start_threshold {
                    // Reached slow start threshold, enter congestion avoidance phase
                    CongestionState::CongestionAvoidance
                } else {
                    // Continue slow start
                    CongestionState::SlowStart
                }
            }
            CongestionState::CongestionAvoidance => {
                if utilization_error > 0.05 {
                    // Utilization significantly exceeds target (5% or more), enter recovery phase
                    CongestionState::Recovery
                } else {
                    // Continue congestion avoidance
                    CongestionState::CongestionAvoidance
                }
            }
            CongestionState::Recovery => {
                if utilization_error <= -0.1 && avg_cost < slow_start_threshold {
                    // Utilization significantly returns to below target and cost is low, re-enter slow start
                    CongestionState::SlowStart
                } else if utilization_error <= 0.01 && utilization_error >= -0.05 {
                    // Utilization basically returns to target vicinity, enter congestion avoidance
                    CongestionState::CongestionAvoidance
                } else {
                    // Continue recovery
                    CongestionState::Recovery
                }
            }
        }
    }
}

/// State machine context
///
/// Maintain historical information and parameters required for state transitions.
#[derive(Debug, Clone)]
pub struct StateMachineContext {
    /// Current state
    pub current_state: CongestionState,
    /// Slow start threshold
    pub slow_start_threshold: f64,
    /// Last state change timestamp (seconds)
    pub last_state_change: f64,
    /// Time spent in current state (seconds)
    pub time_in_state: f64,
    /// State change counter
    pub state_changes: u64,
    /// Maximum observed average cost (used for threshold setting after recovery)
    pub max_avg_cost: f64,
}

impl StateMachineContext {
    /// Create new state machine context
    ///
    /// # Arguments
    ///
    /// * `initial_threshold` - Initial slow start threshold
    /// * `current_time` - Current timestamp (seconds)
    pub fn new(initial_threshold: f64, current_time: f64) -> Self {
        Self {
            current_state: CongestionState::SlowStart,
            slow_start_threshold: initial_threshold,
            last_state_change: current_time,
            time_in_state: 0.0,
            state_changes: 0,
            max_avg_cost: initial_threshold,
        }
    }

    /// Update state machine
    ///
    /// # Arguments
    ///
    /// * `utilization_error` - Utilization error
    /// * `avg_cost` - Current average cost
    /// * `current_time` - Current timestamp (seconds)
    ///
    /// # Returns
    ///
    /// Whether a state transition occurred
    pub fn update(&mut self, utilization_error: f64, avg_cost: f64, current_time: f64) -> bool {
        // Update maximum average cost
        if avg_cost > self.max_avg_cost {
            self.max_avg_cost = avg_cost;
        }

        // Calculate the time spent in the current state
        self.time_in_state = current_time - self.last_state_change;

        // State transition
        let new_state =
            self.current_state
                .transition(utilization_error, avg_cost, self.slow_start_threshold);

        if new_state != self.current_state {
            self.handle_state_transition(new_state, avg_cost, current_time);
            true
        } else {
            false
        }
    }

    /// Handle state transition
    fn handle_state_transition(
        &mut self,
        new_state: CongestionState,
        avg_cost: f64,
        current_time: f64,
    ) {
        let old_state = self.current_state;

        // Special state transition handling
        match (old_state, new_state) {
            (_, CongestionState::Recovery) => {
                // When entering the recovery phase, update the slow start threshold to a certain proportion of the current cost
                self.slow_start_threshold =
                    (avg_cost * 0.75).max(0.1).min(self.slow_start_threshold);
            }
            (CongestionState::Recovery, CongestionState::SlowStart) => {
                // When returning from recovery to slow start, use a conservative threshold
                self.slow_start_threshold = (self.max_avg_cost * 0.5).max(0.1);
            }
            _ => {}
        }

        // Update state and time
        self.current_state = new_state;
        self.last_state_change = current_time;
        self.time_in_state = 0.0;
        self.state_changes += 1;

        tracing::info!(
            old_state = %old_state,
            new_state = %new_state,
            avg_cost = avg_cost,
            threshold = self.slow_start_threshold,
            state_changes = self.state_changes,
            "State transition occurred"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_transitions() {
        let initial_time = 100.0;
        let mut context = StateMachineContext::new(2.0, initial_time);

        // The initial state should be slow start
        assert_eq!(context.current_state, CongestionState::SlowStart);

        // Utilization is too high, should transition to recovery state
        let changed = context.update(0.1, 1.5, initial_time + 1.0);
        assert!(changed);
        assert_eq!(context.current_state, CongestionState::Recovery);

        // Utilization has decreased, but cost is still high, should continue recovery
        let changed = context.update(-0.1, 1.8, initial_time + 2.0);
        assert!(!changed);
        assert_eq!(context.current_state, CongestionState::Recovery);

        // Utilization has decreased slightly, enter congestion avoidance state
        let changed = context.update(-0.03, 1.2, initial_time + 3.0);
        assert!(changed);
        assert_eq!(context.current_state, CongestionState::CongestionAvoidance);

        // Utilization has decreased significantly but is still in congestion avoidance state
        let changed = context.update(-0.15, 0.5, initial_time + 4.0);
        assert!(!changed); // In congestion avoidance state, will not easily return to slow start
        assert_eq!(context.current_state, CongestionState::CongestionAvoidance);
    }

    #[test]
    fn slow_start_to_congestion_avoidance() {
        let initial_time = 100.0;
        let mut context = StateMachineContext::new(2.0, initial_time);

        // Average cost reaches threshold, should transition to congestion avoidance
        let changed = context.update(-0.05, 2.1, initial_time + 1.0);
        assert!(changed);
        assert_eq!(context.current_state, CongestionState::CongestionAvoidance);
    }

    #[test]
    fn threshold_updates() {
        let initial_time = 100.0;
        let mut context = StateMachineContext::new(2.0, initial_time);
        let initial_threshold = context.slow_start_threshold;

        // Entering recovery state should update threshold
        context.update(0.1, 3.0, initial_time + 1.0);
        assert_eq!(context.current_state, CongestionState::Recovery);
        // Threshold should be set to a smaller value (75% of current cost and minimum of original threshold)
        let expected_threshold = (3.0_f64 * 0.75).min(initial_threshold);
        assert!((context.slow_start_threshold - expected_threshold).abs() < 0.01);
    }

    #[test]
    fn debug_info() {
        let mut context = StateMachineContext::new(2.0, 100.0);

        // First update will cause state transition, time_in_state reset to 0
        context.update(0.1, 3.0, 101.0);
        assert_eq!(context.current_state, CongestionState::Recovery);
        assert_eq!(context.time_in_state, 0.0); // Just transitioned, time is 0

        // Second update, should have time accumulated
        context.update(0.1, 3.0, 102.0);
        assert!(context.time_in_state > 0.0); // Now should have time
    }
}
