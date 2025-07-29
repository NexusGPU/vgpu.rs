//! GPU Device State Watcher Module
//! 
//! This module provides functionality to watch GPU device state changes
//! and synchronize them with Kubernetes custom resources.

pub mod types;
pub mod watcher;
pub mod k8s_integration;

// Re-export main types and functions
pub use watcher::GpuDeviceStateWatcher;
pub use types::{KubeletDeviceState, PodDeviceEntry, GPU};