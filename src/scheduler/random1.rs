use anyhow::Result;
use rand::seq::IteratorRandom;
use std::{collections::HashMap, sync::Arc};

use crate::process::{GpuProcess, ProcessState};

use super::{GpuScheduler, SchedulingDecision};

/// Random1Scheduler implementation
pub struct Random1Scheduler {
    processes: HashMap<u32, Arc<dyn GpuProcess>>,
}

impl Random1Scheduler {
    pub fn new() -> Self {
        Self {
            processes: HashMap::new(),
        }
    }
}

impl GpuScheduler for Random1Scheduler {
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
        for (_, process) in self.processes.iter() {
            match process.state() {
                ProcessState::Released | ProcessState::Paused => {}
                ProcessState::Running => {
                    decisions.push(SchedulingDecision::Release(process.id()));
                }
            }
        }

        let random1 = self.processes.iter().choose(&mut rand::thread_rng());

        if let Some((_, p)) = random1 {
            decisions.push(SchedulingDecision::Resume(p.id()));
        }
        Ok(decisions)
    }
}
