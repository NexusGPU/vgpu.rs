use anyhow::Result;
use std::{collections::HashMap, sync::Arc};

use crate::process::{GpuProcess, GpuResources, ProcessState};

use super::{GpuScheduler, SchedulingDecision};

/// Simple FIFO scheduler implementation
pub struct FifoScheduler {
    processes: HashMap<String, Arc<dyn GpuProcess>>,
    gpu_limit: GpuResources,
}

impl FifoScheduler {
    pub fn new(gpu_limit: GpuResources) -> Self {
        Self {
            processes: HashMap::new(),
            gpu_limit,
        }
    }
}

impl GpuScheduler for FifoScheduler {
    fn add_process(&mut self, process: Arc<dyn GpuProcess>) -> Result<()> {
        self.processes.insert(process.id(), process);
        Ok(())
    }

    fn remove_process(&mut self, process_id: &str) -> Result<()> {
        self.processes.remove(process_id);
        Ok(())
    }

    fn get_process(&self, process_id: &str) -> Option<Arc<dyn GpuProcess>> {
        self.processes.get(process_id).cloned()
    }

    fn schedule(&mut self) -> Result<Vec<SchedulingDecision>> {
        let mut decisions = Vec::new();

        // Simple FIFO strategy: if resources are insufficient, pause later processes
        let mut available_memory = self.gpu_limit.memory_bytes;
        let mut available_compute = self.gpu_limit.compute_percentage;

        for (_, process) in self.processes.iter() {
            let current = process.current_resources()?;
            if current.memory_bytes <= available_memory
                && current.compute_percentage <= available_compute
            {
                match process.state() {
                    ProcessState::Released | ProcessState::Paused => {
                        decisions.push(SchedulingDecision::Resume(process.id()));
                    }
                    ProcessState::Running => {}
                }
                available_memory -= current.memory_bytes;
                available_compute -= current.compute_percentage;
            } else {
                // Only pause if we have enough memory but not enough compute
                if current.memory_bytes <= available_memory
                    && current.compute_percentage > available_compute
                    && process.state() == ProcessState::Running
                {
                    decisions.push(SchedulingDecision::Pause(process.id()));
                }
                // Release if we don't have enough memory
                else if current.memory_bytes > available_memory
                    && (process.state() == ProcessState::Running
                        || process.state() == ProcessState::Paused)
                {
                    decisions.push(SchedulingDecision::Release(process.id()));
                }
            }
        }

        Ok(decisions)
    }
}
