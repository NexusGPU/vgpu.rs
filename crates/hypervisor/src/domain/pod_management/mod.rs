//! Pod management module containing coordinator, manager, registry, and related components

pub mod coordinator;
pub mod device_info;
pub mod manager;
pub mod pod_state_store;
pub mod types;
pub mod utilization;

// Re-export commonly used types
pub use coordinator::LimiterCoordinator;
pub use manager::PodManager;
pub use pod_state_store::{PodStateStore, StoreStats};
pub use types::{
    DeviceUsage, PodManagementError, PodState, PodStatus, ProcessState, Result, SystemStats,
};

/// Concrete pod manager type
pub type PodManagerType = PodManager;
