//! Weighted GPU scheduler implementation
//!
//! This module provides a comprehensive weighted scheduling algorithm for GPU processes,
//! with support for QoS levels, trap handling, and advanced scheduling decisions.

mod decision_engine;
mod queue_manager;
mod scheduler;
mod types;
mod weight_calculator;

// Re-export main types and implementation
pub use scheduler::WeightedScheduler;
pub use types::{Trap, WithTraps};
pub use weight_calculator::Weight;

// Internal modules for implementation
use decision_engine::DecisionEngine;
use queue_manager::QueueManager;
