pub mod cli;
pub mod daemon;
pub mod gpu;
pub mod shm;

/// ERL (Elastic Rate Limiter) configuration passed from daemon CLI to the PID controller.
#[derive(Debug, Clone)]
pub struct ErlConfig {
    /// Proportional gain for the PID loop.
    pub pid_kp: f64,
    /// Integral gain for the PID loop.
    pub pid_ki: f64,
    /// Derivative gain for the PID loop.
    pub pid_kd: f64,
    /// Lower bound of the token refill rate that hypervisor may write.
    pub min_refill_rate: f64,
    /// Upper bound of the token refill rate that hypervisor may write.
    pub max_refill_rate: f64,
    /// Initial refill rate when a controller is created.
    pub initial_refill_rate: f64,
    /// Burst allowance (seconds) used to compute token bucket capacity from refill rate.
    pub burst_seconds: f64,
    /// Minimal capacity allowed for the token bucket.
    pub capacity_floor: f64,
    /// Exponential smoothing factor for the derivative term.
    pub derivative_filter: f64,
    /// Clamp for the accumulated integral term (anti-windup).
    pub integral_limit: f64,
    /// Minimum delta time between PID updates to avoid division by zero.
    pub min_delta_time: f64,
}

impl From<&daemon::DaemonArgs> for ErlConfig {
    fn from(args: &daemon::DaemonArgs) -> Self {
        Self {
            pid_kp: args.erl_pid_kp,
            pid_ki: args.erl_pid_ki,
            pid_kd: args.erl_pid_kd,
            min_refill_rate: args.erl_min_refill_rate,
            max_refill_rate: args.erl_max_refill_rate,
            initial_refill_rate: args.erl_initial_refill_rate,
            burst_seconds: args.erl_burst_seconds,
            capacity_floor: args.erl_capacity_floor,
            derivative_filter: args.erl_derivative_filter,
            integral_limit: args.erl_integral_limit,
            min_delta_time: args.erl_min_delta_time,
        }
    }
}

pub use cli::*;
pub use daemon::*;
pub use gpu::*;
pub use shm::*;
