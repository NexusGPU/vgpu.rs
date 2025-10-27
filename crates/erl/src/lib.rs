//! Elastic GPU rate limiter primitives built around a PID controller.
//!
//! The crate deliberately keeps the surface area small so the CUDA
//! limiter and the hypervisor can share a single definition of:
//! * how token buckets are stored in shared memory (`DeviceBackend`)
//! * how kernel launches are admitted (`KernelLimiter`)
//! * how utilization feedback updates refill rates (`DeviceController`)
//!
//! The limiter process only needs `KernelLimiter` with a backend that
//! can read/write token state. The hypervisor instantiates
//! `DeviceController` for each GPU it manages and feeds it the current
//! utilization samples from NVML (or any other source). Both sides
//! share the same `PidController` so tuning can be reasoned about in
//! one place.

mod backend;
mod error;
mod hypervisor;
mod limiter;
mod pid;

pub use backend::{DeviceBackend, DeviceQuota, TokenState};
pub use error::ErlError;
pub use hypervisor::{DeviceController, DeviceControllerConfig, DeviceControllerState};
pub use limiter::{KernelLimiter, KernelLimiterConfig};
pub use pid::{PidConfig, PidController, PidTuning};
