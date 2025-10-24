//! Elastic Rate Limiter (ERL) - Simplified Version
//!
//! A simple, robust GPU rate limiting system with per-Pod control.
//!
//! Key simplifications:
//! - All kernels consume fixed 1.0 token (no workload prediction)
//! - Simple PI controller (2 parameters: Kp, Ki)
//! - Per-Pod independent control
//! - Statistical utilization estimation (no complex event tracking)

mod traits;

mod pi_controller;
mod token_manager;
mod utilization_controller;

#[cfg(test)]
mod fuzz_tests;

pub use pi_controller::{PIController, PIParams};
pub use token_manager::*;
pub use traits::*;
pub use utilization_controller::*;

/// Simplified ERL configuration
#[derive(Debug, Clone)]
pub struct ErlConfig {
    /// PI controller proportional gain
    pub kp: f64,
    /// PI controller integral gain
    pub ki: f64,
    /// Minimum refill rate
    pub min_rate: f64,
    /// Maximum refill rate
    pub max_rate: f64,
}

impl Default for ErlConfig {
    fn default() -> Self {
        Self {
            kp: 0.5,
            ki: 0.1,
            min_rate: 0.1,
            max_rate: 100.0,
        }
    }
}

impl From<ErlConfig> for PIParams {
    fn from(config: ErlConfig) -> Self {
        PIParams {
            kp: config.kp,
            ki: config.ki,
            min_rate: config.min_rate,
            max_rate: config.max_rate,
            // Use default bootstrap parameters
            ..Default::default()
        }
    }
}
