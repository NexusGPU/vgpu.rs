//! Elastic GPU rate limiter built on PID control.
//!
//! This crate provides two main components:
//! - [`KernelLimiter`]: Token bucket admission control for GPU kernels
//! - [`DeviceController`]: PID-based controller that adjusts token refill rates
//!
//! Both components operate on shared memory via the [`DeviceBackend`] trait,
//! allowing the limiter (running in user processes) and controller (running in
//! the hypervisor) to coordinate GPU resource allocation.

use error_stack::Report;

mod backend;
mod error;
mod hypervisor;
mod limiter;

/// Result type using error-stack for context-rich error reporting
pub type Result<T, C> = core::result::Result<T, Report<C>>;

pub use backend::{DeviceBackend, DeviceQuota, TokenState};
pub use error::ErlError;
pub use hypervisor::{DeviceController, DeviceControllerConfig, DeviceControllerState};
pub use limiter::{KernelLimiter, KernelLimiterConfig};
