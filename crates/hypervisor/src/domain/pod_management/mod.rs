//! Pod management module containing coordinator, manager, registry, and related components

pub mod coordinator;
pub mod device_info;
pub mod manager;
pub mod mock;
pub mod pod_state_store;
pub mod sampler;
pub mod traits;
pub mod types;
pub mod utilization;

// Re-export commonly used types
pub use manager::PodManager;
pub use pod_state_store::PodStateStore;
pub use types::{PodManagementError, PodState, PodStatus, Result, SystemStats};
