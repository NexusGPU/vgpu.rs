//! Configuration module for hypervisor
//!
//! This module contains all configuration-related structs and functions,
//! organized into logical sub-modules.

pub mod cli;
pub mod daemon;
pub mod gpu;
pub mod shm;

// Re-export main CLI types
pub use cli::{Cli, Commands};
pub use daemon::DaemonArgs;
pub use gpu::{load_gpu_info, GPU_CAPACITY_MAP};
pub use shm::{MountShmArgs, ShowShmArgs, ShowTuiWorkersArgs};
