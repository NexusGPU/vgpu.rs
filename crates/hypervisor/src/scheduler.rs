use anyhow::Result;
use std::sync::Arc;

use crate::process::GpuProcess;

pub(crate) mod fifo;
pub(crate) mod random1;

/// Scheduling decisions
#[derive(Debug, Clone)]
pub(crate) enum SchedulingDecision {
    /// Pause specified process
    Pause(u32),
    /// Pause and release memory of specified process
    Release(u32),
    /// Resume specified process
    Resume(u32),
}

/// Trait for GPU scheduler
pub(crate) trait GpuScheduler: Send + Sync {
    /// Add a new process to the scheduler
    fn add_process(&mut self, process: Arc<dyn GpuProcess>);

    /// Remove a process from the scheduler
    fn remove_process(&mut self, process_id: u32);

    /// Get a process from the scheduler
    fn get_process(&self, process_id: u32) -> Option<Arc<dyn GpuProcess>>;

    /// Execute scheduling decisions
    /// Returns a series of scheduling operations to be executed
    fn schedule(&mut self) -> Result<Vec<SchedulingDecision>>;
}
