//! lastic Rate Limiter, ERL

mod traits;

mod cost_tracker;
mod cubic;
mod state;
mod workload_calc;

mod token_manager;
mod utilization_controller;

#[cfg(test)]
mod fuzz_tests;

pub use cost_tracker::*;
pub use cubic::{CubicParams, WorkloadAwareCubicController};
pub use token_manager::*;
pub use traits::*;
pub use utilization_controller::*;
pub use workload_calc::*;

/// ERL dynamic configuration (subset of parameters that can be adjusted at runtime)
#[derive(Debug, Clone)]
pub struct ErlDynamicConfig {
    /// Initial average cost
    pub initial_avg_cost: f64,
    /// Minimum average cost
    pub min_avg_cost: f64,
    /// Maximum average cost
    pub max_avg_cost: f64,
    /// CUBIC C parameter
    pub cubic_c: f64,
    /// CUBIC beta
    pub cubic_beta: f64,
    /// CUBIC slow start factor
    pub cubic_slow_start_factor: f64,
    /// Congestion avoidance alpha
    pub congestion_alpha: f64,
    /// Adjustment threshold
    pub adjustment_threshold: f64,
    /// Adjustment coefficient
    pub adjustment_coefficient: f64,
}
