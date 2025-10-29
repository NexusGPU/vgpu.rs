//! Elastic GPU rate limiter built on PID control.
//!
//! This crate provides two main components:
//! - [`KernelLimiter`]: Token bucket admission control for GPU kernels
//! - [`DeviceController`]: PID-based controller that adjusts token refill rates
//!
//! Both components operate on shared memory via the [`DeviceBackend`] trait,
//! allowing the limiter (running in user processes) and controller (running in
//! the hypervisor) to coordinate GPU resource allocation.

mod backend;
mod error;
mod hypervisor;
mod limiter;

pub use backend::{DeviceBackend, DeviceQuota, TokenState};
pub use error::ErlError;
pub use hypervisor::{DeviceController, DeviceControllerConfig, DeviceControllerState};
pub use limiter::{KernelLimiter, KernelLimiterConfig};

// Re-export pid crate for users who need to configure PID parameters
pub use pid;
