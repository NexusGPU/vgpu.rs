//! Cost tracker
//!
//! Used to track actual cost vs expected cost, correct CUBIC feedback

use std::collections::VecDeque;

/// Cost tracker
///
/// Record actual cost vs expected cost, calculate correction ratio
/// To ensure the accuracy of the CUBIC feedback loop
#[derive(Debug, Clone)]
pub struct CostTracker {
    /// Recent cost records: (actual_cost, expected_cost)
    recent_costs: VecDeque<(f64, f64)>,
    /// Maximum number of records
    max_samples: usize,
    /// Total actual cost
    total_actual_cost: f64,
    /// Total expected cost
    total_expected_cost: f64,
}

impl CostTracker {
    /// Create a new cost tracker
    ///
    /// # Arguments
    ///
    /// * `max_samples` - Maximum number of records
    pub fn new(max_samples: usize) -> Self {
        Self {
            recent_costs: VecDeque::with_capacity(max_samples),
            max_samples,
            total_actual_cost: 0.0,
            total_expected_cost: 0.0,
        }
    }

    /// Create a new cost tracker with default parameters
    pub fn with_defaults() -> Self {
        Self::new(100) // Record last 100 costs
    }

    /// Record a cost event
    ///
    /// # Arguments
    ///
    /// * `actual_cost` - Actual cost (avg_cost × workload_factor)
    /// * `expected_cost` - Expected cost (avg_cost)
    pub fn record_cost(&mut self, actual_cost: f64, expected_cost: f64) {
        // Remove old records
        if self.recent_costs.len() >= self.max_samples {
            if let Some((old_actual, old_expected)) = self.recent_costs.pop_front() {
                self.total_actual_cost -= old_actual;
                self.total_expected_cost -= old_expected;
            }
        }

        // Add new record
        self.recent_costs.push_back((actual_cost, expected_cost));
        self.total_actual_cost += actual_cost;
        self.total_expected_cost += expected_cost;

        tracing::trace!(
            actual_cost = actual_cost,
            expected_cost = expected_cost,
            cost_ratio = self.get_cost_ratio(),
            samples = self.recent_costs.len(),
            "Recorded cost event"
        );
    }

    /// Get cost ratio
    ///
    /// # Returns
    ///
    /// Return the ratio of actual_cost / expected_cost
    /// - Ratio > 1.0: Actual cost is higher than expected, GPU utilization may be overestimated
    /// - Ratio < 1.0: Actual cost is lower than expected, GPU utilization may be underestimated
    /// - Ratio = 1.0: Actual cost is equal to expected, no correction needed
    pub fn get_cost_ratio(&self) -> f64 {
        if self.total_expected_cost > 0.0 {
            self.total_actual_cost / self.total_expected_cost
        } else {
            1.0 // No data to correct
        }
    }

    /// Get adjusted utilization
    ///
    /// # Arguments
    ///
    /// * `raw_utilization` - Original GPU utilization
    ///
    /// # Returns
    ///
    /// Return adjusted utilization, for CUBIC feedback
    pub fn adjust_utilization(&self, raw_utilization: f64) -> f64 {
        let cost_ratio = self.get_cost_ratio();

        // If actual cost is higher than expected, it means we actually "bought" more resources
        // Should reduce the utilization feedback to CUBIC, so CUBIC thinks resources are enough
        let adjusted_utilization = raw_utilization / cost_ratio;

        tracing::debug!(
            raw_utilization = raw_utilization,
            cost_ratio = cost_ratio,
            adjusted_utilization = adjusted_utilization,
            "Adjusted utilization for CUBIC feedback"
        );

        adjusted_utilization.clamp(0.0, 1.0)
    }

    /// Get statistics
    pub fn stats(&self) -> CostTrackerStats {
        CostTrackerStats {
            sample_count: self.recent_costs.len(),
            total_actual_cost: self.total_actual_cost,
            total_expected_cost: self.total_expected_cost,
            cost_ratio: self.get_cost_ratio(),
            avg_actual_cost: if self.recent_costs.is_empty() {
                0.0
            } else {
                self.total_actual_cost / (self.recent_costs.len() as f64)
            },
            avg_expected_cost: if self.recent_costs.is_empty() {
                0.0
            } else {
                self.total_expected_cost / (self.recent_costs.len() as f64)
            },
        }
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.recent_costs.clear();
        self.total_actual_cost = 0.0;
        self.total_expected_cost = 0.0;

        tracing::info!("Cost tracker reset");
    }

    /// Check if there is enough data to correct
    pub fn has_sufficient_data(&self) -> bool {
        self.recent_costs.len() >= 10 // At least 10 samples are needed
    }
}

/// Cost tracker statistics
#[derive(Debug, Clone)]
pub struct CostTrackerStats {
    /// Sample count
    pub sample_count: usize,
    /// Total actual cost
    pub total_actual_cost: f64,
    /// Total expected cost
    pub total_expected_cost: f64,
    /// Cost ratio
    pub cost_ratio: f64,
    /// Average actual cost
    pub avg_actual_cost: f64,
    /// Average expected cost
    pub avg_expected_cost: f64,
}

impl std::fmt::Display for CostTrackerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CostTracker(samples: {}, ratio: {:.3}, avg_actual: {:.3}, avg_expected: {:.3})",
            self.sample_count, self.cost_ratio, self.avg_actual_cost, self.avg_expected_cost
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_tracker_basic() {
        let mut tracker = CostTracker::new(5);

        // Initial state
        assert_eq!(tracker.get_cost_ratio(), 1.0);
        assert!(!tracker.has_sufficient_data());

        // Record some costs
        tracker.record_cost(2.0, 1.0); // Actual cost is 2 times expected
        tracker.record_cost(4.0, 2.0); // Actual cost is 2 times expected

        assert!((tracker.get_cost_ratio() - 2.0).abs() < 0.001);
    }

    #[test]
    fn cost_tracker_adjustment() {
        let mut tracker = CostTracker::new(10);

        // Record high actual cost cases
        for _ in 0..10 {
            tracker.record_cost(3.0, 1.0); // Actual cost is 3 times expected
        }

        // Original utilization 90%, adjusted utilization should be 30%
        let adjusted = tracker.adjust_utilization(0.9);
        assert!((adjusted - 0.3).abs() < 0.01);

        assert!(tracker.has_sufficient_data());
    }

    #[test]
    fn cost_tracker_window() {
        let mut tracker = CostTracker::new(3); // Only keep 3 samples

        // Add data exceeding window size
        tracker.record_cost(1.0, 1.0);
        tracker.record_cost(2.0, 1.0);
        tracker.record_cost(3.0, 1.0);
        tracker.record_cost(4.0, 1.0); // This will remove the first sample

        let stats = tracker.stats();
        assert_eq!(stats.sample_count, 3);
        assert_eq!(stats.total_actual_cost, 9.0); // 2+3+4
        assert_eq!(stats.total_expected_cost, 3.0); // 1+1+1
    }

    #[test]
    fn cost_tracker_edge_cases() {
        let mut tracker = CostTracker::new(10);

        // Zero expected cost
        tracker.record_cost(1.0, 0.0);
        assert_eq!(tracker.get_cost_ratio(), 1.0); // Should fallback to 1.0

        // Reset test
        tracker.record_cost(2.0, 1.0);
        tracker.reset();
        assert_eq!(tracker.get_cost_ratio(), 1.0);
        assert_eq!(tracker.stats().sample_count, 0);
    }

    #[test]
    fn cost_tracker_stats() {
        let mut tracker = CostTracker::new(5);

        tracker.record_cost(2.0, 1.0);
        tracker.record_cost(4.0, 2.0);

        let stats = tracker.stats();
        assert_eq!(stats.sample_count, 2);
        assert_eq!(stats.total_actual_cost, 6.0);
        assert_eq!(stats.total_expected_cost, 3.0);
        assert_eq!(stats.cost_ratio, 2.0);
        assert_eq!(stats.avg_actual_cost, 3.0);
        assert_eq!(stats.avg_expected_cost, 1.5);
    }
}
