//! ERL abstract trait definitions
//!
//! Clearly separate the responsibilities of the limiter process and the hypervisor process

use error_stack::Result;

/// ERL error type
#[derive(Debug, derive_more::Display)]
pub enum ErlError {
    /// Admission control failed
    #[display("Admission denied: {reason}")]
    AdmissionDenied { reason: String },
    /// Resource monitoring failed
    #[display("Resource monitoring failed: {reason}")]
    MonitoringFailed { reason: String },
    /// Congestion control failed
    #[display("Congestion control update failed: {reason}")]
    CongestionControlFailed { reason: String },
    /// Invalid configuration
    #[display("Invalid configuration: {reason}")]
    InvalidConfiguration { reason: String },
}

impl core::error::Error for ErlError {}

/// Shared storage trait (both processes need)
///
/// Abstract underlying shared memory storage, supporting cross-process token and quota management
pub trait SharedStorage<K>: Send + Sync
where
    K: std::hash::Hash + Eq + Clone + Send + Sync,
{
    /// Load token bucket state (limiter and hypervisor need)
    fn load_token_state(&self, key: &K) -> Result<(f64, f64), ErlError>;

    /// Save token bucket state (mainly used by limiter)
    fn save_token_state(&self, key: &K, tokens: f64, timestamp: f64) -> Result<(), ErlError>;

    /// Load quota information (limiter read, hypervisor set)
    fn load_quota(&self, key: &K) -> Result<(f64, f64), ErlError>;

    /// Set quota information (only used by hypervisor)
    fn set_quota(&self, key: &K, capacity: f64, refill_rate: f64) -> Result<(), ErlError>;

    /// Load current average cost (limiter read, hypervisor update)
    fn load_avg_cost(&self, key: &K) -> Result<f64, ErlError>;

    /// Save current average cost (only used by hypervisor)
    fn save_avg_cost(&self, key: &K, avg_cost: f64) -> Result<(), ErlError>;
}

/// Token manager trait (limiter process专用）
///
/// Execute token consumption and workload-aware admission control in CUDA application process
pub trait TokenManager<K>: Send + Sync
where
    K: std::hash::Hash + Eq + Clone + Send + Sync,
{
    /// Associated storage type
    type Storage: SharedStorage<K>;

    /// Try to acquire execution permission (workload-aware version)
    ///
    /// Calculate dynamic cost based on kernel workload and execute admission control
    ///
    /// # Arguments
    ///
    /// * `key` - Token bucket identifier (usually device ID)
    /// * `grid_count` - Number of blocks in the grid
    /// * `block_count` - Number of threads in each block
    ///
    /// # Errors
    ///
    /// Return [`ErlError::AdmissionDenied`] when tokens are insufficient
    fn try_acquire_workload(
        &mut self,
        key: &K,
        grid_count: u32,
        block_count: u32,
    ) -> Result<(), ErlError>;

    /// Get token bucket status (for monitoring and debugging)
    fn get_token_status(&self, key: &K) -> Result<(f64, f64, f64), ErlError>;
}

/// Utilization controller trait (hypervisor process专用）
///
/// Execute utilization monitoring, target setting and congestion control in hypervisor process
pub trait UtilizationController<K>: Send + Sync
where
    K: std::hash::Hash + Eq + Clone + Send + Sync,
{
    /// Associated storage type
    type Storage: SharedStorage<K>;

    /// Update resource utilization feedback
    ///
    /// Hypervisor periodically collects GPU utilization and updates CUBIC algorithm state
    ///
    /// # Arguments
    ///
    /// * `utilization` - Current resource utilization (0.0 - 1.0)
    fn update_utilization(&mut self, utilization: f64) -> Result<(), ErlError>;

    /// Get target utilization
    fn target_utilization(&self) -> f64;

    /// Set target utilization
    ///
    /// # Arguments
    ///
    /// * `target` - New target utilization (0.0 - 1.0)
    fn set_target_utilization(&mut self, target: f64) -> Result<(), ErlError>;

    /// Initialize device quota (called by hypervisor at startup)
    ///
    /// # Arguments
    ///
    /// * `key` - Device identifier
    /// * `capacity` - Token bucket capacity
    /// * `refill_rate` - Token refill rate (tokens/second)
    fn initialize_device_quota(
        &mut self,
        key: &K,
        capacity: f64,
        refill_rate: f64,
    ) -> Result<(), ErlError>;

    /// Get all device status overview (for monitoring)
    fn get_devices_overview(&self) -> Result<Vec<(K, f64, f64, f64)>, ErlError>;
}

/// Congestion control algorithm trait
///
/// Used for CUBIC and other algorithms in hypervisor
pub trait CongestionController: Send + Sync {
    /// Update internal state based on resource utilization and return current base cost
    ///
    /// # Arguments
    ///
    /// * `current_utilization` - Current resource utilization (0.0 - 1.0)
    /// * `target_utilization` - Target utilization (0.0 - 1.0)
    /// * `delta_time` - Time interval since last update in seconds (typically ~0.1s for 100ms update interval)
    ///
    /// # Returns
    ///
    /// Return current task's base cost (will be adjusted by workload factor)
    fn update(
        &mut self,
        current_utilization: f64,
        target_utilization: f64,
        delta_time: f64,
    ) -> Result<f64, ErlError>;

    /// Get current base cost (used to store to shared memory)
    fn current_avg_cost(&self) -> f64;
}
