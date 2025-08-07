pub mod hypervisor;
pub mod pod_management;
pub mod process;
pub mod scheduler;

// Re-export concrete types for convenience
pub use hypervisor::HypervisorType;
pub use pod_management::PodManagerType;
