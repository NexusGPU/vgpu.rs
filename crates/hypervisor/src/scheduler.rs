use anyhow::Result;

use crate::process::GpuProcess;

pub(crate) mod weighted;

/// Scheduling decisions
#[derive(Debug)]
pub(crate) enum SchedulingDecision {
    /// Pause specified process
    #[allow(dead_code)]
    Pause(u32),
    /// Pause and release memory of specified process
    Release(u32),
    /// Resume specified process
    Resume(u32),
    /// Wake up a process
    Wake(trap::Waker, u64, trap::TrapAction),
}

/// Trait for GPU scheduler
pub(crate) trait GpuScheduler<Proc: GpuProcess> {
    /// Add a new process to the scheduler
    fn add_process(&mut self, process: Proc);

    /// Remove a process from the scheduler
    fn remove_process(&mut self, process_id: u32);

    /// Get a process from the scheduler
    fn get_process(&self, process_id: u32) -> Option<&Proc>;

    /// Execute scheduling decisions
    /// Returns a series of scheduling operations to be executed
    fn schedule(&mut self) -> Result<Vec<SchedulingDecision>>;

    fn done_decision(&mut self, decision: &SchedulingDecision);

    /// Handle a trap event for a process
    fn on_trap(
        &mut self,
        process_id: u32,
        trap_id: u64,
        frame: &trap::TrapFrame,
        waker: trap::Waker,
    );
}
