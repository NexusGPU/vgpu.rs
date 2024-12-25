use anyhow::Result;
use std::{collections::HashMap, sync::Arc};

use crate::process::{GpuProcess, GpuResources, ProcessState};

use super::{GpuScheduler, SchedulingDecision};

/// Simple FIFO scheduler implementation
pub struct FifoScheduler {
    processes: HashMap<u32, Arc<dyn GpuProcess>>,
    gpu_limits: HashMap<String, GpuResources>,
}

impl FifoScheduler {
    pub fn new(gpu_limits: HashMap<String, GpuResources>) -> Self {
        Self {
            processes: HashMap::new(),
            gpu_limits,
        }
    }
}

impl GpuScheduler for FifoScheduler {
    fn add_process(&mut self, process: Arc<dyn GpuProcess>) {
        self.processes.insert(process.id(), process);
    }

    fn remove_process(&mut self, process_id: u32) {
        self.processes.remove(&process_id);
    }

    fn get_process(&self, process_id: u32) -> Option<Arc<dyn GpuProcess>> {
        self.processes.get(&process_id).cloned()
    }

    fn schedule(&mut self) -> Result<Vec<SchedulingDecision>> {
        let mut decisions = Vec::new();

        // Track available resources per GPU
        let mut available_resources: HashMap<String, GpuResources> = self.gpu_limits.clone();

        // Simple FIFO strategy: if resources are insufficient, pause later processes
        for (_, process) in self.processes.iter() {
            let current = process.current_resources()?;
            let gpu_uuid = process.gpu_uuid();

            // Skip if GPU not found in limits
            let available = match available_resources.get_mut(gpu_uuid) {
                Some(res) => res,
                None => {
                    tracing::warn!("GPU {} not found in limits", gpu_uuid);
                    continue;
                }
            };

            if current.memory_bytes <= available.memory_bytes
                && current.compute_percentage <= available.compute_percentage
            {
                match process.state() {
                    ProcessState::Released | ProcessState::Paused => {
                        decisions.push(SchedulingDecision::Resume(process.id()));
                    }
                    ProcessState::Running => {}
                }
                available.memory_bytes -= current.memory_bytes;
                available.compute_percentage -= current.compute_percentage;
            } else {
                // Only pause if we have enough memory but not enough compute
                if current.memory_bytes <= available.memory_bytes
                    && current.compute_percentage > available.compute_percentage
                    && process.state() == ProcessState::Running
                {
                    decisions.push(SchedulingDecision::Pause(process.id()));
                }
                // Release if we don't have enough memory
                else if current.memory_bytes > available.memory_bytes
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
