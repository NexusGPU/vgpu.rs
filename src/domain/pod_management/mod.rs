//! Pod management module with improved architecture
//! 
//! This module provides a clean, well-structured approach to managing pods,
//! containers, and worker processes with proper separation of concerns.

// Core functionality
pub mod core;
pub mod types;
pub mod services;

// Legacy modules (deprecated, will be removed)
mod coordinator;
mod manager;
mod registration;
mod registry;
mod resource_tracker;
mod utilization;

// Re-export main types and services
pub use core::{PodManager, PodRegistry, PodManagementError, Result};
pub use types::{Pod, PodId, Worker, DeviceAllocation, DeviceUsage};
pub use services::{DeviceService, WorkerService, ResourceMonitor};

// Legacy re-exports for backward compatibility (deprecated)
#[deprecated(note = "Use core::PodManager instead")]
pub use manager::PodManager as LegacyPodManager;

#[deprecated(note = "Use services::DeviceService instead")]
pub use coordinator::LimiterCoordinator;