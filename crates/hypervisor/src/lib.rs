// Command layer - entry points
pub mod cmd;

// Core business logic layer
pub mod core;

// Configuration layer - configuration management
pub mod config;

// TUI layer - terminal user interface
pub mod tui;

// Controller layer - HTTP API
pub mod controller;

// Platform layer - technology-specific implementations
pub mod platform;

// Utility layer
pub mod util;

// Re-export commonly used types for backward compatibility
pub use cmd::{run_daemon, run_mount_shm, run_show_shm, run_show_tui_workers};
pub use config::{Cli, Commands, DaemonArgs};
pub use controller::{JwtAuthConfig, PodInfo, PodInfoResponse, ProcessInfo};
pub use core::{GpuProcess, GpuResources, ProcessState, Worker};
pub use core::{GpuScheduler, Hypervisor, HypervisorType, Scheduler, SchedulingDecision};
pub use core::{PodManager, PodState, PodStatus};
pub use process::WorkerHandle;

// Application builder for backward compatibility
pub mod app {
    pub use crate::util::{Application, ApplicationBuilder, ApplicationServices};
}

// Logging re-exports for backward compatibility
pub mod logging {
    pub use crate::util::logging::*;
}

// Infrastructure re-exports for backward compatibility
pub mod infrastructure {
    pub use crate::platform::*;
}

// Domain re-exports for backward compatibility
pub mod domain {
    pub use crate::core::*;
}

// Process management re-exports
pub mod process {
    pub use crate::core::process::*;
}

// Scheduler re-exports
pub mod scheduler {
    pub use crate::core::scheduler::*;
}

// Pod management re-exports
pub mod pod_management {
    pub use crate::core::pod::*;
}

// GPU-related re-exports
pub mod gpu_device_state_watcher {
    pub use crate::platform::nvml::gpu_device_state_watcher::*;
}

pub mod gpu_init {
    pub use crate::platform::nvml::gpu_init::*;
}

pub mod gpu_observer {
    pub use crate::platform::nvml::gpu_observer::*;
}

pub mod host_pid_probe {
    pub use crate::platform::host_pid_probe::*;
}

pub mod k8s {
    pub use crate::platform::k8s::*;
}

pub mod kube_client {
    pub use crate::platform::kube_client::*;
}

pub mod limiter_comm {
    pub use crate::platform::limiter_comm::*;
}
