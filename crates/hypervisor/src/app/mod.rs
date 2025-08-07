//! Application module
//!
//! This module contains the main application structure and lifecycle management,
//! organized into logical sub-modules.

pub mod builder;
pub mod core;
pub mod services;
pub mod tasks;

// Re-export main types
pub use builder::ApplicationBuilder;
pub use core::Application;
pub use services::ApplicationServices;
