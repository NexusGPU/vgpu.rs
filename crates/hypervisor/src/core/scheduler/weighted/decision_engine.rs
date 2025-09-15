//! Decision engine for scheduling logic

use super::super::SchedulingDecision;
use super::types::{Trap, WithTraps};
use crate::core::process::GpuProcess;
use priority_queue::PriorityQueue;
use std::collections::HashMap;
use trap::{TrapAction, TrapError, Waker};

/// Engine for making scheduling decisions based on process states and priorities
pub struct DecisionEngine;

impl DecisionEngine {
    /// Generate scheduling decisions based on current queue states
    pub fn make_decisions<Proc: GpuProcess>(
        processes: &HashMap<u32, WithTraps<Proc>>,
        running_queue: &PriorityQueue<u32, u32>,
        sleep_queue: &PriorityQueue<u32, u32>,
        trap_wait_queue: &PriorityQueue<u32, u32>,
    ) -> anyhow::Result<Vec<SchedulingDecision>> {
        let mut decisions = Vec::new();

        // Handle trap completions first (highest priority)
        if let Some((pid, _)) = trap_wait_queue.peek() {
            if let Some(process) = processes.get(pid) {
                // Check if any traps can be resolved
                for trap in &process.traps {
                    // This is a simplified example - in reality, you'd check specific trap conditions
                    if Self::should_wake_trap(trap) {
                        // Create a dummy waker for now - in real implementation you'd store the original waker
                        let dummy_waker = Box::new(DummyWaker);
                        decisions.push(SchedulingDecision::Wake(
                            dummy_waker,
                            0, // trap_id would come from the trap
                            TrapAction::Resume,
                        ));
                        break;
                    }
                }
            }
        }

        // Resume high-priority sleeping processes
        if let Some((pid, weight)) = sleep_queue.peek() {
            if Self::should_resume(*weight, running_queue) {
                decisions.push(SchedulingDecision::Resume(*pid));
            }
        }

        // Consider releasing memory from low-priority running processes
        for (pid, weight) in running_queue.iter() {
            if let Some(process) = processes.get(pid) {
                if Self::should_release(*weight, process) {
                    decisions.push(SchedulingDecision::Release(*pid));
                    break; // Only release one at a time for stability
                }
            }
        }

        Ok(decisions)
    }

    /// Check if a trap should be woken up
    fn should_wake_trap(trap: &Trap) -> bool {
        // Simplified logic - in practice, this would check memory availability,
        // other resource states, etc.
        trap.round > 5 // Wake up traps that have been waiting too long
    }

    /// Check if a sleeping process should be resumed
    fn should_resume(weight: u32, running_queue: &PriorityQueue<u32, u32>) -> bool {
        // Resume if we have capacity or if the sleeping process has higher priority
        if running_queue.len() < 4 {
            return true;
        }

        // Check if this process has higher priority than the lowest running process
        if let Some((_, min_weight)) = running_queue.iter().min_by_key(|(_, w)| *w) {
            weight > *min_weight
        } else {
            true
        }
    }

    /// Check if a running process should be released to free up memory
    fn should_release<Proc: GpuProcess>(weight: u32, process: &WithTraps<Proc>) -> bool {
        // Release low-priority processes under memory pressure
        // This is simplified - real implementation would check actual memory usage
        weight < 15 && process.traps.is_empty() // Only release if no active traps
    }
}

/// Dummy waker implementation for compilation - in real implementation you'd store the original waker
struct DummyWaker;

#[async_trait::async_trait]
impl Waker for DummyWaker {
    async fn send(&self, _trap_id: u64, _action: TrapAction) -> Result<(), TrapError> {
        Ok(())
    }
}
