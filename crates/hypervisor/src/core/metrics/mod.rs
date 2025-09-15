//! Core metrics collection logic
//!
//! This module contains the business logic for collecting and processing metrics

pub mod collector;

pub use collector::*;

// Re-export from platform layer
pub use crate::platform::metrics::{encoders, BytesWrapper};
