//! Utility modules
//!
//! Common utilities used across the codebase:
//! - Configuration management
//! - Logging setup
//! - TUI components
//! - Application builder

pub mod builder;
pub mod core;
pub mod keyed_lock;
pub mod logging;
pub mod services;
pub mod tasks;

// Re-export configuration types
pub use builder::ApplicationBuilder;
pub use core::Application;
pub use services::ApplicationServices;
