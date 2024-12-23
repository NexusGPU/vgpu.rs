use anyhow::Result;
use std::sync::Arc;

use crate::process::{GpuProcess, GpuResources};

use super::{GpuScheduler, SchedulingDecision};

/// Simple FIFO scheduler implementation
pub struct FifoScheduler {
    processes: Vec<Arc<dyn GpuProcess>>,
    gpu_limit: GpuResources,
}

impl FifoScheduler {
    pub fn new(gpu_limit: GpuResources) -> Self {
        Self {
            processes: Vec::new(),
            gpu_limit,
        }
    }
}

impl GpuScheduler for FifoScheduler {
    fn add_process(&mut self, process: Arc<dyn GpuProcess>) -> Result<()> {
        self.processes.push(process);
        Ok(())
    }

    fn remove_process(&mut self, process_id: &str) -> Result<()> {
        self.processes.retain(|p| p.id() != process_id);
        Ok(())
    }

    fn schedule(&mut self) -> Result<Vec<SchedulingDecision>> {
        let mut decisions = Vec::new();

        // Simple FIFO strategy: if resources are insufficient, pause later processes
        let mut available_memory = self.gpu_limit.memory_bytes;
        let mut available_compute = self.gpu_limit.compute_percentage;

        for process in &self.processes {
            let required = process.requested_resources();

            if required.memory_bytes <= available_memory
                && required.compute_percentage <= available_compute
            {
                if process.state() != crate::process::ProcessState::Running {
                    decisions.push(SchedulingDecision::Resume(process.id()));
                }
                available_memory -= required.memory_bytes;
                available_compute -= required.compute_percentage;
            } else {
                if process.state() == crate::process::ProcessState::Running {
                    decisions.push(SchedulingDecision::Release(process.id()));
                }
            }
        }

        Ok(decisions)
    }
}
