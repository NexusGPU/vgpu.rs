use std::fmt;

use anyhow::Result;
use std::sync::Arc;

use crate::core::process::{GpuProcess, WorkerHandle};

pub mod weighted;

// Re-export weighted scheduler
pub use weighted::WeightedScheduler;

/// Concrete scheduler type
pub type Scheduler = WeightedScheduler<WorkerHandle>;

/// Scheduling decisions
pub enum SchedulingDecision {
    /// Pause specified process
    #[allow(dead_code)]
    Pause(u32),
    /// Pause and release memory of specified process
    Release(u32),
    /// Resume specified process
    Resume(u32),
    /// Wake up a process
    Wake(Box<dyn trap::Waker>, u64, trap::TrapAction),
}

impl fmt::Debug for SchedulingDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pause(pid) => f.debug_tuple("Pause").field(pid).finish(),
            Self::Release(pid) => f.debug_tuple("Release").field(pid).finish(),
            Self::Resume(pid) => f.debug_tuple("Resume").field(pid).finish(),
            Self::Wake(_, trap_id, action) => f
                .debug_tuple("Wake")
                .field(&"<waker>")
                .field(trap_id)
                .field(action)
                .finish(),
        }
    }
}

/// Trait for GPU scheduler
#[async_trait::async_trait]
pub trait GpuScheduler<Proc: GpuProcess> {
    /// Register a new process with the scheduler
    fn register_process(&mut self, process: Proc);

    /// Remove a process from the scheduler
    fn remove_process(&mut self, process_id: u32);

    /// Get a process from the scheduler
    fn get_process(&self, process_id: u32) -> Option<&Proc>;

    /// Execute scheduling decisions
    /// Returns a series of scheduling operations to be executed
    async fn schedule(&mut self) -> Result<Vec<SchedulingDecision>>;

    fn done_decision(&mut self, decision: &SchedulingDecision);

    /// Handle a trap event for a process
    async fn on_trap(
        &mut self,
        process_id: u32,
        trap_id: u64,
        frame: Arc<trap::TrapFrame>,
        waker: Box<dyn trap::Waker>,
    );
}
