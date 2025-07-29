//! Core pod management functionality

pub mod registry;
pub mod manager;
pub mod error;

// Re-export main types
pub use registry::PodRegistry;
pub use manager::PodManager;
pub use error::{PodManagementError, Result};