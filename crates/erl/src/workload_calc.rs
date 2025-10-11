//! Workload calculator
//!
//! Provide workload factor calculation based on grid and block parameters

// Workload calculation does not depend on ERL error type, remove unused import

/// Workload calculator trait
pub trait WorkloadCalculator: Send + Sync + std::fmt::Debug {
    /// Calculate workload factor based on grid and block
    ///
    /// # Arguments
    ///
    /// * `grid_count` - Total number of blocks in the grid
    /// * `block_count` - Total number of threads in each block
    ///
    /// # Returns
    ///
    /// Return workload factor, used to adjust cost: actual_cost = avg_cost × factor
    fn calculate_factor(&self, grid_count: u32, block_count: u32) -> f64;

    /// Get calculator name (for logging and debugging)
    fn name(&self) -> &'static str;
}

/// Power function workload calculator (recommended)
///
/// Use power function to balance sensitivity and smoothness, formula:
/// factor = (total_threads / reference_threads)^power
#[derive(Debug, Clone)]
pub struct PowerWorkloadCalculator {
    /// Reference threads (baseline)
    pub reference_threads: u32,
    /// Power index (0.6 balance sensitivity and smoothness)
    pub power: f64,
    /// Minimum factor
    pub min_factor: f64,
    /// Maximum factor
    pub max_factor: f64,
}

impl Default for PowerWorkloadCalculator {
    fn default() -> Self {
        Self {
            // Modern ML kernels often launch 100K+ threads
            // Setting reference to 16K provides better scaling for large workloads
            reference_threads: 16384,
            power: 0.6,
            min_factor: 0.1,
            // Reduced max_factor to limit cost variability
            // With base_avg_cost in [0.1, 10.0] and max_factor=4.0,
            // dynamic_cost ranges [0.04, 40.0] which is sufficient for differentiation
            max_factor: 4.0,
        }
    }
}

impl WorkloadCalculator for PowerWorkloadCalculator {
    fn calculate_factor(&self, grid_count: u32, block_count: u32) -> f64 {
        let total_threads = (grid_count as u64) * (block_count as u64);
        let normalized = (total_threads as f64) / (self.reference_threads as f64);
        let capped_normalized = normalized.min(1000.0); // Prevent overflow

        capped_normalized
            .powf(self.power)
            .clamp(self.min_factor, self.max_factor)
    }

    fn name(&self) -> &'static str {
        "PowerWorkload"
    }
}

/// Linear workload calculator (simple solution)
#[derive(Debug, Clone)]
pub struct LinearWorkloadCalculator {
    /// Reference threads
    pub reference_threads: u32,
    /// Scale factor
    pub scale_factor: f64,
    /// Minimum factor
    pub min_factor: f64,
    /// Maximum factor
    pub max_factor: f64,
}

impl Default for LinearWorkloadCalculator {
    fn default() -> Self {
        Self {
            reference_threads: 16384,
            scale_factor: 1.0,
            min_factor: 0.1,
            max_factor: 4.0,
        }
    }
}

impl WorkloadCalculator for LinearWorkloadCalculator {
    fn calculate_factor(&self, grid_count: u32, block_count: u32) -> f64 {
        let total_threads = (grid_count as u64) * (block_count as u64);
        let ratio = (total_threads as f64) / (self.reference_threads as f64);

        (ratio * self.scale_factor).clamp(self.min_factor, self.max_factor)
    }

    fn name(&self) -> &'static str {
        "LinearWorkload"
    }
}

/// Fixed factor calculator (for testing or disabling workload awareness)
#[derive(Debug, Clone)]
pub struct FixedWorkloadCalculator {
    factor: f64,
}

impl FixedWorkloadCalculator {
    pub fn new(factor: f64) -> Self {
        Self { factor }
    }
}

impl Default for FixedWorkloadCalculator {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl WorkloadCalculator for FixedWorkloadCalculator {
    fn calculate_factor(&self, _grid_count: u32, _block_count: u32) -> f64 {
        self.factor
    }

    fn name(&self) -> &'static str {
        "FixedWorkload"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_workload_calculator() {
        let calc = PowerWorkloadCalculator::default();

        // Small workload (32 threads, well below reference)
        let small_factor = calc.calculate_factor(1, 32);
        assert!(
            small_factor < 1.0,
            "Small workload should have factor < 1.0"
        );
        assert!(small_factor >= 0.1, "Factor should be >= min_factor");

        // Baseline workload (reference_threads = 16384)
        let base_factor = calc.calculate_factor(16, 1024);
        assert!(
            (base_factor - 1.0).abs() < 0.1,
            "Base workload (16K threads) should have factor ≈ 1.0"
        );

        // Large workload (1M threads)
        let large_factor = calc.calculate_factor(1024, 1024);
        assert!(
            large_factor > 1.0,
            "Large workload should have factor > 1.0"
        );
        assert!(large_factor <= 4.0, "Factor should be <= max_factor (4.0)");

        // Monotonicity check
        assert!(small_factor < base_factor);
        assert!(base_factor < large_factor);
    }

    #[test]
    fn linear_workload_calculator() {
        let calc = LinearWorkloadCalculator::default();

        // With reference_threads = 16384
        let factor1 = calc.calculate_factor(1, 8192); // 8K threads: 0.5x reference
        let factor2 = calc.calculate_factor(16, 1024); // 16K threads: 1.0x reference
        let factor3 = calc.calculate_factor(32, 1024); // 32K threads: 2.0x reference

        // Linear growth
        assert!(
            factor1 < factor2,
            "factor1 ({factor1}) should be < factor2 ({factor2})"
        );
        assert!(
            factor2 < factor3,
            "factor2 ({factor2}) should be < factor3 ({factor3})"
        );

        // Baseline check (16K threads should give factor ≈ 1.0)
        assert!(
            (factor2 - 1.0).abs() < 0.1,
            "Baseline factor should be ≈ 1.0, got {factor2}"
        );
    }

    #[test]
    fn fixed_workload_calculator() {
        let calc = FixedWorkloadCalculator::new(2.5);

        assert_eq!(calc.calculate_factor(1, 1), 2.5);
        assert_eq!(calc.calculate_factor(1000, 1000), 2.5);
        assert_eq!(calc.calculate_factor(1, 1000000), 2.5);
    }

    #[test]
    fn workload_factor_bounds() {
        let calc = PowerWorkloadCalculator {
            reference_threads: 16384,
            power: 0.6,
            min_factor: 0.2,
            max_factor: 5.0,
        };

        // Test tiny value (1 thread)
        let tiny_factor = calc.calculate_factor(1, 1);
        assert!(tiny_factor >= 0.2, "Should respect min_factor");

        // Test huge value (100M threads)
        let huge_factor = calc.calculate_factor(10000, 10000);
        assert!(huge_factor <= 5.0, "Should respect max_factor");
    }
}
