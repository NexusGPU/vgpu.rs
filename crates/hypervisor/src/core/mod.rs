//! Core business logic layer
//!
//! Contains pure domain logic independent of external systems:
//! - Hypervisor engine
//! - GPU scheduling algorithms
//! - Process management
//! - Pod lifecycle management
//! - Metrics collection logic

pub mod hypervisor;
pub mod pod;
pub mod process;
pub mod scheduler;

// Re-export main types
pub use hypervisor::{Hypervisor, HypervisorType};
pub use pod::{PodManager, PodState, PodStatus};
pub use process::{GpuProcess, GpuResources, ProcessState, Worker};
pub use scheduler::{GpuScheduler, Scheduler, SchedulingDecision};
