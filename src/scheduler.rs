use anyhow::Result;
use std::sync::Arc;

use crate::process::GpuProcess;

pub mod fifo;

/// Scheduling decisions
#[derive(Debug, Clone)]
pub enum SchedulingDecision {
    /// Pause specified process
    Pause(String),
    /// Pause and release memory of specified process
    Release(String),
    /// Resume specified process
    Resume(String),
}

/// Trait for GPU scheduler
pub trait GpuScheduler: Send + Sync {
    /// Add a new process to the scheduler
    fn add_process(&mut self, process: Arc<dyn GpuProcess>) -> Result<()>;

    /// Remove a process from the scheduler
    fn remove_process(&mut self, process_id: &str) -> Result<()>;

    /// Get a process from the scheduler
    fn get_process(&self, process_id: &str) -> Option<Arc<dyn GpuProcess>>;

    /// Execute scheduling decisions
    /// Returns a series of scheduling operations to be executed
    fn schedule(&mut self) -> Result<Vec<SchedulingDecision>>;
}
