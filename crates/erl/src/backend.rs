use crate::ErlError;
use error_stack::Report;
pub type Result<T, C> = core::result::Result<T, Report<C>>;

/// Snapshot of a token bucket.
#[derive(Debug, Clone, Copy)]
pub struct TokenState {
    pub tokens: f64,
    pub last_update: f64,
}

impl TokenState {
    pub fn new(tokens: f64, last_update: f64) -> Self {
        Self {
            tokens,
            last_update,
        }
    }
}

/// Capacity and refill rate of a token bucket.
#[derive(Debug, Clone, Copy)]
pub struct DeviceQuota {
    pub capacity: f64,
    pub refill_rate: f64,
}

impl DeviceQuota {
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            capacity,
            refill_rate,
        }
    }
}

/// Minimal interface the limiter and hypervisor expect from shared state.
pub trait DeviceBackend: Send + Sync {
    fn read_token_state(&self, device: usize) -> Result<TokenState, ErlError>;
    fn write_token_state(&self, device: usize, state: TokenState) -> Result<(), ErlError>;
    fn read_quota(&self, device: usize) -> Result<DeviceQuota, ErlError>;
    fn write_refill_rate(&self, device: usize, refill_rate: f64) -> Result<(), ErlError>;
    fn write_capacity(&self, device: usize, capacity: f64) -> Result<(), ErlError>;

    /// Atomically subtract tokens if available. Returns the tokens before subtraction.
    ///
    /// This operation should be atomic to prevent race conditions when multiple
    /// processes/threads try to consume tokens concurrently.
    fn fetch_sub_tokens(&self, device: usize, cost: f64) -> Result<f64, ErlError>;

    /// Atomically add tokens (for refilling by hypervisor).
    fn fetch_add_tokens(&self, device: usize, amount: f64) -> Result<f64, ErlError>;
}
