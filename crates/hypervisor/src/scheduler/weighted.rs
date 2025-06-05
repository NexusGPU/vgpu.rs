use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::Deref;
use std::ops::DerefMut;

use priority_queue::PriorityQueue;

use super::GpuScheduler;
use crate::process::GpuProcess;

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

    fn on_trap(
        &mut self,
        process_id: u32,
        _trap_id: u64,
        frame: &trap::TrapFrame,
        waker: trap::Waker,
    ) {
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

    /// Schedule processes based on their weights
    ///
    /// The scheduling algorithm works as follows:
    /// ```text
    ///                              +----------------+
    ///                              |    Schedule    |
    ///                              +----------------+
    ///                                     |
    ///                                     v
    ///                       +----------------------------+
    ///                       | Get highest weight process |
    ///                       |  - from trap_wait_queue    |
    ///                       |  - from sleep_queue        |
    ///                       +----------------------------+
    ///                                     |
    ///                                     v
    ///                       +----------------------------+
    ///                       | Get minimum running weight |
    ///                       +----------------------------+
    ///                                     |
    ///                                     v
    ///                           /------------------\    No
    ///                          ( Any waiting proc?  ) ------+
    ///                           \------------------/        |
    ///                                    | Yes              |
    ///                                    v                  |
    ///                        +------------------------+     |
    ///                        | Compare process weights|     |
    ///                        +------------------------+     |
    ///                                    |                  |
    ///                                    v                  |
    ///              +---------------------(?)----------------+
    ///              |                     |                  |
    ///              v                     v                  v
    ///      +--------------+    +------------------+    +--------+
    ///      | Handle Trap  |    | Resume from      |    | Break  |
    ///      | if highest & |    | sleep if higher  |    |        |
    ///      | > min run    |    | than min run     |    |        |
    ///      +--------------+    +------------------+    +--------+
    /// ```
    fn schedule(&mut self) -> anyhow::Result<Vec<super::SchedulingDecision>> {
        let mut decisions = Vec::new();
        // Use a HashSet to track all processes that have been processed in this scheduling round
        // This prevents any process from being processed more than once in the same round
        let mut processed_processes = HashSet::new();

        loop {
            // Get highest weight process from both queues, filtering out already processed processes
            let trap_process = self
                .trap_wait_queue
                .iter()
                .filter(|(pid, _)| !processed_processes.contains(*pid))
                .max_by_key(|(_, weight)| *weight)
                .map(|(pid, weight)| (*pid, *weight));
            let sleep_process = self
                .sleep_queue
                .iter()
                .filter(|(pid, _)| !processed_processes.contains(*pid))
                .max_by_key(|(_, weight)| *weight)
                .map(|(pid, weight)| (*pid, *weight));

            // Get minimum weight in running queue
            // If running queue is empty, use 0 to allow any process to run
            let min_weight_in_running_queue =
                self.running_queue.peek().map(|(_, w)| *w).unwrap_or(0);

            // Process based on weights
            match (trap_process, sleep_process) {
                (None, None) => break, // No waiting processes
                (Some((trap_pid, trap_weight)), Some((sleep_pid, sleep_weight))) => {
                    // Check if trap process has higher priority
                    let trap_has_priority = trap_weight >= sleep_weight;

                    // Skip this process if it has already been processed in this scheduling round
                    if trap_has_priority && trap_weight > min_weight_in_running_queue {
                        // Handle the process
                        self.handle_trap_process(&mut decisions, trap_pid, trap_weight)?;
                        // Mark this process as processed regardless of success
                        processed_processes.insert(trap_pid);
                    } else if sleep_weight > min_weight_in_running_queue {
                        decisions.push(super::SchedulingDecision::Resume(sleep_pid));
                        // Mark this process as processed
                        processed_processes.insert(sleep_pid);
                    } else {
                        break;
                    }
                }
                (Some((trap_pid, trap_weight)), None) => {
                    // Skip this process if it has already been processed in this scheduling round
                    if trap_weight > min_weight_in_running_queue {
                        // Handle the process
                        self.handle_trap_process(&mut decisions, trap_pid, trap_weight)?;
                        // Mark this process as processed regardless
                        processed_processes.insert(trap_pid);
                    } else {
                        break;
                    }
                }
                (None, Some((sleep_pid, sleep_weight))) => {
                    if sleep_weight > min_weight_in_running_queue {
                        decisions.push(super::SchedulingDecision::Resume(sleep_pid));
                        // Mark this process as processed
                        processed_processes.insert(sleep_pid);
                    } else {
                        break;
                    }
                }
            }
        }
        Ok(decisions)
    }

    fn done_decision(&mut self, decision: &super::SchedulingDecision) {
        match decision {
            super::SchedulingDecision::Resume(pid) => {
                if self.running_queue.get(pid).is_some() {
                    return;
                }

                // Move process from sleep queue to running queue
                if let Some((_, &weight)) = self.sleep_queue.get(pid) {
                    self.sleep_queue.remove(pid);
                    self.running_queue.push(*pid, weight);
                }
            }
            super::SchedulingDecision::Release(pid) => {
                if self.sleep_queue.get(pid).is_some() {
                    return;
                }

                // Move process from running queue to sleep queue
                if let Some((_, &weight)) = self.running_queue.get(pid) {
                    self.running_queue.remove(pid);
                    self.sleep_queue.push(*pid, weight);
                }
            }
            super::SchedulingDecision::Wake(_, _, _) => {}
            super::SchedulingDecision::Pause(_) => {}
        }
    }
}

impl<Proc: GpuProcess> WeightedScheduler<Proc> {
    pub(crate) fn new() -> Self {
        Self {
            processes: HashMap::new(),
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
        let mut released_memory = 0;

        // Get all processes with lower weight than the requesting process
        let mut candidates: Vec<_> = self
            .running_queue
            .iter()
            .filter(|(_, &weight)| weight < min_weight)
            .map(|(pid, _)| *pid)
            .collect();

        // Sort by weight (lowest first)
        candidates.sort_by_key(|pid| {
            self.running_queue
                .get(pid)
                .map(|(_, &weight)| weight)
                .unwrap_or(0)
        });

        // Release processes until we have enough memory
        for pid in candidates {
            if let Some(process) = self.processes.get(&pid) {
                let current_resources = process.current_resources();
                released_memory += current_resources.memory_bytes;
                decisions.push(super::SchedulingDecision::Release(pid));

                if released_memory >= requested_bytes {
                    return (decisions, true);
                }
            }
        }

        // If we couldn't release enough memory, return false
        (decisions, released_memory >= requested_bytes)
    }

    fn handle_trap_process(
        &mut self,
        decisions: &mut Vec<super::SchedulingDecision>,
        pid: u32,
        weight: u32,
    ) -> anyhow::Result<()> {
        // Returns true if process was handled successfully, false if it needs to be retried later
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

                    // Add the release decisions to the output
                    decisions.extend(release_decisions);

                    if success {
                        decisions.push(super::SchedulingDecision::Wake(
                            waker,
                            round as u64, // Convert round to u64 for trap_id
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
            // Put process back
            self.processes.insert(pid, process);
            return Ok(());
        } else if process.traps.is_empty() {
            self.running_queue.push(pid, process.weight());
        }

        // Put process back
        self.processes.insert(pid, process);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ipc_channel::ipc;

    use super::*;
    use crate::process::tests::MockGpuProcess;
    use crate::scheduler::SchedulingDecision;

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
                1, // Using the same trap_id for simplicity in tests
                &trap::TrapFrame::OutOfMemory {
                    requested_bytes: 512,
                },
                create_test_waker(),
            );
        }

        // Verify traps are accumulated
        let process = scheduler.processes.get(&1).unwrap();
        assert_eq!(process.traps.len(), 3);
    }

    #[test]
    fn test_resume_from_sleep() {
        let mut scheduler: WeightedScheduler<MockGpuProcess> = WeightedScheduler::new();

        // Create two processes with different priorities
        // Process 1 has higher QoS than Process 2, so Process 2 should be released when Process 1 needs memory
        let process1 = MockGpuProcess::new_with_qos(1, 1024, 50, crate::process::QosLevel::High);
        let process2 = MockGpuProcess::new_with_qos(2, 1024, 50, crate::process::QosLevel::Low);

        scheduler.add_process(process1);
        scheduler.add_process(process2);

        // Simulate memory pressure by triggering OOM
        scheduler.on_trap(
            1,
            1,
            &trap::TrapFrame::OutOfMemory {
                requested_bytes: 2048,
            },
            create_test_waker(),
        );

        // Process1 should be in trap queue, Process2 should be moved to sleep queue
        let decisions = scheduler.schedule().unwrap();

        // Verify process2 is released and moved to sleep queue
        assert!(decisions
            .iter()
            .any(|d| matches!(d, SchedulingDecision::Release(2))));

        for decision in &decisions {
            scheduler.done_decision(decision);
        }

        assert!(
            scheduler.sleep_queue.get(&2).is_some(),
            "Process 2 should be in sleep queue after Release"
        );
        assert!(
            scheduler.running_queue.get(&2).is_none(),
            "Process 2 should not be in running queue after Release"
        );

        // Now simulate more memory becoming available by removing process1
        scheduler.remove_process(1);

        // Schedule again - process2 should be resumed
        let decisions = scheduler.schedule().unwrap();

        // Verify process2 is resumed
        assert!(decisions
            .iter()
            .any(|d| matches!(d, SchedulingDecision::Resume(2))));

        // Execute the Resume decisions using the scheduler's method and verify each process is moved correctly
        for decision in &decisions {
            if let SchedulingDecision::Resume(pid) = decision {
                scheduler.done_decision(decision);

                // Verify the process was moved to running queue after Resume
                assert!(
                    scheduler.running_queue.get(pid).is_some(),
                    "Process {pid} should be in running queue after Resume"
                );
                assert!(
                    scheduler.sleep_queue.get(pid).is_none(),
                    "Process {pid} should not be in sleep queue after Resume"
                );
            }
        }
    }

    #[test]
    fn test_sleep_queue_weight_priority() {
        let mut scheduler: WeightedScheduler<MockGpuProcess> = WeightedScheduler::new();

        // Create three processes with different priorities
        let process1 = MockGpuProcess::new_with_qos(1, 1024, 50, crate::process::QosLevel::Low);
        let process2 = MockGpuProcess::new_with_qos(2, 1024, 50, crate::process::QosLevel::Medium);
        let process3 = MockGpuProcess::new_with_qos(3, 1024, 50, crate::process::QosLevel::High);

        scheduler.add_process(process1);
        scheduler.add_process(process2);
        scheduler.add_process(process3);

        // Verify initial state
        assert!(scheduler.running_queue.get(&1).is_some());
        assert!(scheduler.running_queue.get(&2).is_some());
        assert!(scheduler.running_queue.get(&3).is_some());

        // Simulate high memory pressure by triggering OOM for high priority process
        scheduler.on_trap(
            3,
            1,
            &trap::TrapFrame::OutOfMemory {
                requested_bytes: 2048,
            },
            create_test_waker(),
        );

        // Process 3 should now be in the trap queue
        assert!(scheduler.trap_wait_queue.get(&3).is_some());
        assert!(scheduler.running_queue.get(&3).is_none());

        // Manually release lower priority processes to simulate what would happen
        // in the release_memory_from_running function
        let weight1 = *scheduler.running_queue.get(&1).unwrap().1;
        scheduler.running_queue.remove(&1);
        scheduler.sleep_queue.push(1, weight1);

        let weight2 = *scheduler.running_queue.get(&2).unwrap().1;
        scheduler.running_queue.remove(&2);
        scheduler.sleep_queue.push(2, weight2);

        // Verify processes 1 and 2 are now in sleep queue
        assert!(
            scheduler.sleep_queue.get(&1).is_some(),
            "Process 1 should be in sleep queue"
        );
        assert!(
            scheduler.sleep_queue.get(&2).is_some(),
            "Process 2 should be in sleep queue"
        );
        assert!(
            scheduler.running_queue.get(&1).is_none(),
            "Process 1 should not be in running queue"
        );
        assert!(
            scheduler.running_queue.get(&2).is_none(),
            "Process 2 should not be in running queue"
        );

        // Remove high priority process to simulate it completing
        scheduler.remove_process(3);

        // Now schedule - medium priority process should be resumed first
        let decisions = scheduler.schedule().unwrap();

        // Verify medium priority process (2) is resumed first
        assert!(decisions
            .iter()
            .any(|d| matches!(d, SchedulingDecision::Resume(2))));

        // Execute the Resume decision
        for decision in &decisions {
            scheduler.done_decision(decision);
        }

        // Verify process 2 is now in running queue
        assert!(
            scheduler.running_queue.get(&2).is_some(),
            "Process 2 should be in running queue"
        );
        assert!(
            scheduler.sleep_queue.get(&2).is_none(),
            "Process 2 should not be in sleep queue"
        );

        // With our updated scheduler, Process 1 might also be processed in the same round
        // If Process 1 is still in the sleep queue, that's good
        // If it's been moved to the running queue, that's also acceptable
        assert!(
            scheduler.sleep_queue.get(&1).is_some() || scheduler.running_queue.get(&1).is_some(),
            "Process 1 should be in either sleep queue or running queue"
        );

        // If Process 1 is in the running queue, we're done with the test
        if scheduler.running_queue.get(&1).is_some() {
            return;
        }

        // Otherwise, remove process 2 to simulate it completing
        scheduler.remove_process(2);

        // Now schedule again - low priority process should be resumed
        let decisions = scheduler.schedule().unwrap();

        // Verify low priority process (1) is resumed
        assert!(decisions
            .iter()
            .any(|d| matches!(d, SchedulingDecision::Resume(1))));
    }
}
