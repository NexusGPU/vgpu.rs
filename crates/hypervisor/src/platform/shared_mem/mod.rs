//! Shared memory platform implementation
//!
//! This module provides shared memory management for inter-process communication:
//! - Shared memory segment creation and management
//! - GPU state synchronization between processes
//! - Process discovery and monitoring

pub mod manager;
pub mod state_sync;

pub use manager::SharedMemoryManager;
pub use state_sync::StateSync;
