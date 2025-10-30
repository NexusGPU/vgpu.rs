pub mod cli;
pub mod daemon;
pub mod gpu;
pub mod shm;

/// ERL (Elastic Rate Limiter) configuration passed from daemon CLI to the PID controller.
#[derive(Debug, Clone)]
pub struct ErlConfig {
    /// Controller update interval in milliseconds.
    pub update_interval_ms: u64,
    /// Minimum refill rate (tokens/second) - prevents rate from dropping to zero
    pub rate_min: f64,
    /// Maximum refill rate (tokens/second)
    pub rate_max: f64,
    /// PID proportional gain - how aggressively to respond to error
    pub kp: f64,
    /// PID integral gain - how quickly to eliminate steady-state error
    pub ki: f64,
    /// PID derivative gain - how much to dampen oscillations
    pub kd: f64,
    /// Low-pass filter coefficient for smoothing utilization (0.0 to 1.0)
    pub filter_alpha: f64,
    /// Burst window in seconds - capacity = refill_rate × burst_window
    pub burst_window: f64,
    /// Minimum capacity (tokens)
    pub capacity_min: f64,
    /// Maximum capacity (tokens) - prevents unbounded growth
    pub capacity_max: f64,
}

impl ErlConfig {
    /// Create a `DeviceControllerConfig` from this ERL config with the given target utilization.
    ///
    /// The `min_delta_time` field is set to a fixed default of 0.05 seconds.
    pub fn to_device_controller_config(
        &self,
        target_utilization: f64,
    ) -> erl::DeviceControllerConfig {
        erl::DeviceControllerConfig {
            target_utilization,
            rate_min: self.rate_min,
            rate_max: self.rate_max,
            kp: self.kp,
            ki: self.ki,
            kd: self.kd,
            filter_alpha: self.filter_alpha,
            burst_window: self.burst_window,
            capacity_min: self.capacity_min,
            capacity_max: self.capacity_max,
            ..Default::default()
        }
    }
}

impl From<&daemon::DaemonArgs> for ErlConfig {
    fn from(args: &daemon::DaemonArgs) -> Self {
        let erl_params = args
            .scheduling_config
            .as_ref()
            .map(|sc| &sc.elastic_rate_limit_parameters)
            .cloned()
            .unwrap_or_default();

        Self {
            update_interval_ms: args.erl_update_interval_ms,
            rate_min: erl_params.min_refill_rate,
            rate_max: erl_params.max_refill_rate,
            kp: erl_params.kp,
            ki: erl_params.ki,
            kd: erl_params.kd,
            filter_alpha: erl_params.filter_alpha,
            burst_window: erl_params.burst_window,
            capacity_min: erl_params.capacity_min,
            capacity_max: erl_params.capacity_max,
        }
    }
}

pub use cli::*;
pub use daemon::*;
pub use gpu::*;
pub use shm::*;
