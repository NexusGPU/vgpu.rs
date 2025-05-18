use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use crate::process::GpuProcess;
use priority_queue::PriorityQueue;
use trap::{self};

use super::GpuScheduler;

struct Trap {
    pub frame: trap::TrapFrame,
    pub waker: trap::Waker,
    pub round: u32,
}
struct WithTraps<Proc> {
    pub process: Proc,
    pub traps: Vec<Trap>,
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
    processes: HashMap<u32, WithTraps<Proc>>,

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
        let qos = match self.process.qos_level() {
            crate::process::QosLevel::Low => 1,
            crate::process::QosLevel::Medium => 2,
            crate::process::QosLevel::High => 3,
            crate::process::QosLevel::Critical => 4,
        };

        self.traps
            .iter()
            .fold(0, |acc, trap| acc + (trap.round * qos))
            + qos * 10
    }
}

impl<Proc: GpuProcess> GpuScheduler<Proc> for WeightedScheduler<Proc> {
    fn add_process(&mut self, process: Proc) {
        let pid = process.id();
        let process = WithTraps {
            process,
            traps: Vec::new(),
        };
        let weight = process.weight();
        self.processes.insert(pid, process);
        self.running_queue.push(pid, weight);
    }

    fn remove_process(&mut self, process_id: u32) {
        if self.processes.remove(&process_id).is_some() {
            // Remove from all queues
            self.running_queue.remove(&process_id);
            self.sleep_queue.remove(&process_id);
            self.trap_wait_queue.remove(&process_id);
        }
    }

    fn get_process(&self, process_id: u32) -> Option<&Proc> {
        self.processes.get(&process_id).map(|p| &p.process)
    }

    /// Schedule processes in trap wait queue based on their weights
    /// ```plaintext
    /// Process Flow:
    /// ┌─────────────────┐
    /// │ Trap Wait Queue │
    /// └────────┬────────┘
    ///          │ weight > min_running_weight?
    ///          ▼
    /// ┌─────────────────┐
    /// │ Remove Process  │←─────────────┐
    /// └────────┬────────┘              │
    ///          │                       │
    ///          ▼                       │
    /// ┌─────────────────┐     No       │
    /// │ Process Traps   │──────────────┘
    /// └────────┬────────┘   Retry
    ///          │ Success
    ///          ▼
    /// ┌─────────────────┐
    /// │ Update Queue:   │
    /// │ - Running Queue │
    /// │ - Trap Queue    │
    /// └─────────────────┘
    /// ```
    fn schedule(&mut self) -> anyhow::Result<Vec<super::SchedulingDecision>> {
        let mut decisions = Vec::new();
        loop {
            // First peek at the trap wait queue without borrowing self
            let (pid, weight) = match self.trap_wait_queue.peek() {
                Some((pid, weight)) => (*pid, *weight),
                None => break,
            };

            let min_weight_in_running_queue =
                self.running_queue.peek().map(|(_, w)| *w).unwrap_or(0);

            if weight > min_weight_in_running_queue {
                // Remove from queues and get process
                self.trap_wait_queue.remove(&pid);
                let mut process = self.processes.remove(&pid).expect("process not found");

                // Process each trap
                let mut should_retry = false;
                while let Some(Trap {
                    round,
                    frame: trap_frame,
                    waker,
                }) = process.traps.pop()
                {
                    match trap_frame {
                        trap::TrapFrame::OutOfMemory { requested_bytes } => {
                            let (release_decisions, success) =
                                self.release_memory_from_running(requested_bytes, weight);
                            decisions.extend(release_decisions);

                            if success {
                                decisions.push(super::SchedulingDecision::Wake(
                                    waker,
                                    trap::TrapAction::Resume,
                                ));
                            } else {
                                // insufficient memory, push back the trap
                                process.traps.push(Trap {
                                    round: round + 1,
                                    frame: trap_frame,
                                    waker,
                                });
                                should_retry = true;
                                break;
                            }
                        }
                    }
                }

                // Put process back in appropriate queue
                if should_retry {
                    self.trap_wait_queue.push(pid, process.weight());
                } else if process.traps.is_empty() {
                    self.running_queue.push(pid, process.weight());
                }

                // Put process back
                self.processes.insert(pid, process);
            } else {
                break;
            }
        }
        Ok(decisions)
    }

    /// Handle a trap event for a process
    fn on_trap(&mut self, process_id: u32, frame: &trap::TrapFrame, waker: trap::Waker) {
        if let Some(process) = self.processes.get_mut(&process_id) {
            process.traps.push(Trap {
                frame: frame.clone(),
                waker,
                round: 1,
            });
            let weight = process.weight();
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
            processes: HashMap::default(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::tests::MockGpuProcess;
    use crate::scheduler::SchedulingDecision;
    use ipc_channel::ipc;

    fn create_test_waker() -> trap::Waker {
        let (sender, _receiver) = ipc::channel().unwrap();
        sender
    }

    #[test]
    fn test_weighted_scheduler_basic() {
        let mut scheduler: WeightedScheduler<MockGpuProcess> = WeightedScheduler::new();

        // Create a mock process
        let process = MockGpuProcess::new(1, 2048, 75);
        scheduler.add_process(process);

        // Verify the process is added
        assert!(scheduler.get_process(1).is_some());
        assert!(scheduler.running_queue.peek().is_some());

        // Remove the process
        scheduler.remove_process(1);

        // Verify the process is removed
        assert!(scheduler.get_process(1).is_none());
        assert!(scheduler.running_queue.is_empty());
    }

    #[test]
    fn test_process_weight_calculation() {
        let mut scheduler: WeightedScheduler<MockGpuProcess> = WeightedScheduler::new();

        // Create processes with different QoS levels
        let process_low = MockGpuProcess::new_with_qos(1, 1024, 50, crate::process::QosLevel::Low);
        let process_high =
            MockGpuProcess::new_with_qos(2, 1024, 50, crate::process::QosLevel::High);

        scheduler.add_process(process_low);
        scheduler.add_process(process_high);

        // High priority process should have higher weight
        let (_, weight_low) = scheduler.running_queue.get(&1).unwrap();
        let (_, weight_high) = scheduler.running_queue.get(&2).unwrap();

        assert!(
            weight_high > weight_low,
            "High QoS process should have higher weight"
        );
    }

    #[test]
    fn test_trap_weight_increase() {
        let mut scheduler: WeightedScheduler<MockGpuProcess> = WeightedScheduler::new();

        // Add a process
        let process = MockGpuProcess::new(1, 2048, 75);
        scheduler.add_process(process);

        let initial_weight = *scheduler.running_queue.get(&1).unwrap().1;

        // Trigger a trap
        scheduler.on_trap(
            1,
            &trap::TrapFrame::OutOfMemory {
                requested_bytes: 1024,
            },
            create_test_waker(),
        );

        // Verify weight increase
        assert!(!scheduler.trap_wait_queue.is_empty());
        let trap_weight = scheduler.trap_wait_queue.peek().unwrap().1;
        assert!(
            trap_weight > &initial_weight,
            "Weight should increase after trap"
        );
    }

    #[test]
    fn test_memory_release_scheduling() {
        let mut scheduler: WeightedScheduler<MockGpuProcess> = WeightedScheduler::new();

        // Add three processes to simulate memory pressure scenario
        let process1 = MockGpuProcess::new(1, 2048, 50); // 2GB memory
        let process2 = MockGpuProcess::new(2, 3072, 30); // 3GB memory
        let process3 = MockGpuProcess::new(3, 4096, 20); // 4GB memory

        scheduler.add_process(process1);
        scheduler.add_process(process2);
        scheduler.add_process(process3);

        // Trigger memory trap
        scheduler.on_trap(
            1,
            &trap::TrapFrame::OutOfMemory {
                requested_bytes: 5120,
            }, // Request 5GB memory
            create_test_waker(),
        );

        // Execute scheduling
        let decisions = scheduler.schedule().unwrap();

        // Verify low weight processes are released
        assert!(decisions.iter().any(|d| matches!(
            d,
            SchedulingDecision::Release(2) | SchedulingDecision::Release(3)
        )));
    }

    #[test]
    fn test_schedule_with_no_processes() {
        let mut scheduler: WeightedScheduler<MockGpuProcess> = WeightedScheduler::new();
        let decisions = scheduler.schedule().unwrap();
        assert!(
            decisions.is_empty(),
            "Empty scheduler should produce no decisions"
        );
    }

    #[test]
    fn test_multiple_traps() {
        let mut scheduler: WeightedScheduler<MockGpuProcess> = WeightedScheduler::new();
        let process = MockGpuProcess::new(1, 1024, 50);
        scheduler.add_process(process);

        // Trigger multiple traps
        for _ in 0..3 {
            scheduler.on_trap(
                1,
                &trap::TrapFrame::OutOfMemory {
                    requested_bytes: 512,
                },
                create_test_waker(),
            );
        }

        // Verify traps are correctly accumulated
        let process = scheduler.processes.get(&1).unwrap();
        assert_eq!(process.traps.len(), 3, "All traps should be queued");
    }
}
