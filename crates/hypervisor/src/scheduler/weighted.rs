use std::ops::{Deref, DerefMut};

use crate::process::GpuProcess;
use fnv::FnvHashMap;
use priority_queue::PriorityQueue;

use super::GpuScheduler;

struct WithTraps<Proc> {
    pub process: Proc,
    pub traps: Vec<(
        trap::TrapFrame,
        Box<dyn FnOnce(Result<trap::TrapAction, trap::TrapError>) + Send + Sync>,
    )>,
}

impl<Proc> Deref for WithTraps<Proc> {
    type Target = Proc;
    fn deref(&self) -> &Self::Target {
        &self.process
    }
}

impl<Proc> DerefMut for WithTraps<Proc> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.process
    }
}

pub(crate) struct WeightedScheduler<Proc> {
    // pid -> process
    processes: FnvHashMap<u32, WithTraps<Proc>>,

    // PriorityQueue(pid, weight)
    running_queue: PriorityQueue<u32, u32>,

    sleep_queue: PriorityQueue<u32, u32>,
    trap_wait_queue: PriorityQueue<u32, u32>,
}

trait Weight {
    fn weight(&self) -> u32;
}

impl<T: GpuProcess> Weight for WithTraps<T> {
    fn weight(&self) -> u32 {
        todo!()
    }
}

impl<Proc: GpuProcess> GpuScheduler<Proc> for WeightedScheduler<Proc> {
    fn add_process(&mut self, process: Proc) {
        let pid = process.id();
        let process = WithTraps {
            process: process,
            traps: Vec::new(),
        };
        let weight = process.weight();
        self.processes.insert(pid, process);
        self.running_queue.push(pid, weight);
    }

    fn remove_process(&mut self, process_id: u32) {
        if let Some(_) = self.processes.remove(&process_id) {
            // Remove from all queues
            self.running_queue.remove(&process_id);
            self.sleep_queue.remove(&process_id);
            self.trap_wait_queue.remove(&process_id);
        }
    }

    fn get_process(&self, process_id: u32) -> Option<&Proc> {
        self.processes.get(&process_id).map(|p| &p.process)
    }

    fn schedule(&mut self) -> anyhow::Result<Vec<super::SchedulingDecision>> {
        let mut decisions = Vec::new();
        loop {
            let (waiting_pid, waiting_weight) = match self.trap_wait_queue.peek() {
                Some((pid, weight)) => (*pid, *weight),
                None => break,
            };
            let min_weight_in_running_queue = self
                .running_queue
                .peek()
                .map(|(_, weight)| *weight)
                .unwrap_or(0);

            if waiting_weight > min_weight_in_running_queue {
                let mut process = self
                    .processes
                    .remove(&waiting_pid)
                    .expect("process not found");

                if waiting_weight > min_weight_in_running_queue {
                    while let Some((trap_frame, waker)) = process.traps.pop() {
                        match trap_frame {
                            trap::TrapFrame::OutOfMemory { requested_bytes } => {
                                let (release_decisions, success) = self
                                    .release_memory_from_running(requested_bytes, waiting_weight);
                                decisions.extend(release_decisions);
                                if success {
                                    decisions.push(super::SchedulingDecision::Wake(waker, Ok(trap::TrapAction::Resume)));
                                } else {
                                    // insufficient memory, push back the trap
                                    process.traps.push((trap_frame, waker));
                                    break;
                                }
                            }
                        }
                    }
                    continue;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        Ok(decisions)
    }

    /// Handle a trap event for a process
    fn on_trap(
        &mut self,
        process_id: u32,
        frame: &trap::TrapFrame,
        waker: Box<dyn FnOnce(Result<trap::TrapAction, trap::TrapError>) + Send + Sync>,
    ) {
        if let Some(process) = self.processes.get_mut(&process_id) {
            let weight = process.weight();
            process.traps.push((frame.clone(), waker));
            // Move the process to the trap wait queue
            self.running_queue.remove(&process_id);
            self.sleep_queue.remove(&process_id);

            self.trap_wait_queue.push(process_id, weight);
        } else {
            // Process not found
            tracing::warn!("process {} not found for trap", process_id);
        }
    }
}

impl<Proc: GpuProcess> WeightedScheduler<Proc> {
    pub(crate) fn new() -> Self {
        Self {
            processes: FnvHashMap::default(),
            running_queue: PriorityQueue::new(),
            sleep_queue: PriorityQueue::new(),
            trap_wait_queue: PriorityQueue::new(),
        }
    }
    fn release_memory_from_running(
        &mut self,
        requested_bytes: u64,
        min_weight: u32,
    ) -> (Vec<super::SchedulingDecision>, bool) {
        let mut decisions = Vec::new();
        let mut available_resources = 0;
        while let Some((pid, weight)) = self.running_queue.peek() {
            let current_pid = *pid;
            let current_weight = *weight;
            if current_weight > min_weight {
                return (decisions, false);
            }
            let _ = self.running_queue.pop();
            self.sleep_queue.push(current_pid, current_weight);
            decisions.push(super::SchedulingDecision::Release(current_pid));

            if let Some(process) = self.processes.get(&current_pid) {
                let current_resources = process.current_resources();
                available_resources += current_resources.memory_bytes;
                if available_resources >= requested_bytes {
                    return (decisions, true);
                }
            }
        }
        (decisions, false)
    }
}
