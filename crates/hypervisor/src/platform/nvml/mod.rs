//! NVIDIA Management Library (NVML) integration
//!
//! This module provides GPU device management using NVIDIA's NVML library:
//! - GPU device discovery and initialization
//! - Memory and utilization monitoring
//! - Device state management

pub mod gpu_device_state_watcher;
pub mod gpu_init;
pub mod gpu_observer;

pub use gpu_device_state_watcher::GpuDeviceStateWatcher;
pub use gpu_init::init_nvml;
pub use gpu_init::GpuSystem;
pub use gpu_observer::GpuObserver;
