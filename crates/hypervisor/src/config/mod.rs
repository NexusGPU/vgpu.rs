pub mod cli;
pub mod daemon;
pub mod gpu;
pub mod shm;

/// ERL (Elastic Rate Limiter) configuration
#[derive(Debug, Clone)]
pub struct ErlConfig {
    /// Base token refill rate per second
    pub base_refill_rate: f64,
    /// Token bucket burst duration
    pub burst_duration: f64,
    /// Minimum token bucket capacity
    pub min_capacity: f64,
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

impl From<&daemon::DaemonArgs> for ErlConfig {
    fn from(args: &daemon::DaemonArgs) -> Self {
        Self {
            base_refill_rate: args.erl_base_refill_rate,
            burst_duration: args.erl_burst_duration,
            min_capacity: args.erl_min_capacity,
            initial_avg_cost: args.erl_initial_avg_cost,
            min_avg_cost: args.erl_min_avg_cost,
            max_avg_cost: args.erl_max_avg_cost,
            cubic_c: args.erl_cubic_c,
            cubic_beta: args.erl_cubic_beta,
            cubic_slow_start_factor: args.erl_cubic_slow_start_factor,
            congestion_alpha: args.erl_congestion_alpha,
            adjustment_threshold: args.erl_adjustment_threshold,
            adjustment_coefficient: args.erl_adjustment_coefficient,
        }
    }
}

impl From<&ErlConfig> for erl::ErlDynamicConfig {
    fn from(config: &ErlConfig) -> Self {
        Self {
            initial_avg_cost: config.initial_avg_cost,
            min_avg_cost: config.min_avg_cost,
            max_avg_cost: config.max_avg_cost,
            cubic_c: config.cubic_c,
            cubic_beta: config.cubic_beta,
            cubic_slow_start_factor: config.cubic_slow_start_factor,
            congestion_alpha: config.congestion_alpha,
            adjustment_threshold: config.adjustment_threshold,
            adjustment_coefficient: config.adjustment_coefficient,
        }
    }
}

pub use cli::*;
pub use daemon::*;
pub use gpu::*;
pub use shm::*;
