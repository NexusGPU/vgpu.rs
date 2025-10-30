pub mod cli;
pub mod daemon;
pub mod gpu;
pub mod shm;

/// ERL (Elastic Rate Limiter) configuration passed from daemon CLI to the PID controller.
#[derive(Debug, Clone)]
pub struct ErlConfig {
    /// Controller update interval in milliseconds.
    pub update_interval_ms: u64,
    /// System-wide maximum refill rate safety limit (tokens/sec).
    pub rate_limit: f64,
    /// Controller responsiveness multiplier (0.5 = slow, 1.0 = normal, 2.0 = fast).
    pub responsiveness: f64,
}

impl From<&daemon::DaemonArgs> for ErlConfig {
    fn from(args: &daemon::DaemonArgs) -> Self {
        Self {
            update_interval_ms: args.erl_update_interval_ms,
            rate_limit: args.erl_rate_limit,
            responsiveness: args.erl_responsiveness.max(0.1),
        }
    }
}

pub use cli::*;
pub use daemon::*;
pub use gpu::*;
pub use shm::*;
