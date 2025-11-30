//! Main weighted scheduler implementation

use anyhow::Result;
use influxdb_line_protocol::LineProtocolBuilder;
use std::collections::HashMap;
use std::sync::Arc;

use super::super::{GpuScheduler, SchedulingDecision};
use super::decision_engine::DecisionEngine;
use super::queue_manager::{QueueManager, QueueType};
use super::types::{Trap, WithTraps};
use super::weight_calculator::Weight;
use crate::core::process::GpuProcess;
use crate::platform::metrics::{current_time, BytesWrapper};
use trap::{TrapFrame, Waker};

/// Weighted scheduler implementation with modular architecture
pub struct WeightedScheduler<Proc> {
    /// Process registry: pid -> process with traps
    processes: HashMap<u32, WithTraps<Proc>>,
    /// Queue manager for different process states
    queues: QueueManager,
}

impl<Proc: GpuProcess> Default for WeightedScheduler<Proc> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Proc: GpuProcess> WeightedScheduler<Proc> {
    /// Create a new weighted scheduler
    pub fn new() -> Self {
        Self {
            processes: HashMap::new(),
            queues: QueueManager::new(),
        }
    }

    /// Get process statistics
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            total_processes: self.processes.len(),
            running_processes: self.queues.running_queue.len(),
            sleeping_processes: self.queues.sleep_queue.len(),
            trap_waiting_processes: self.queues.trap_wait_queue.len(),
        }
    }

    /// Check if scheduler has any processes
    pub fn is_empty(&self) -> bool {
        self.processes.is_empty()
    }

    /// Get all process IDs
    pub fn process_ids(&self) -> Vec<u32> {
        self.processes.keys().copied().collect()
    }

    /// Update trap rounds (called periodically)
    pub fn update_trap_rounds(&mut self) {
        for process in self.processes.values_mut() {
            for trap in &mut process.traps {
                trap.round += 1;
            }
        }
    }
}

#[async_trait::async_trait]
impl<Proc: GpuProcess> GpuScheduler<Proc> for WeightedScheduler<Proc> {
    fn register_process(&mut self, process: Proc) {
        let pid = process.pid();
        let qos = process.qos_level();
        let wrapped_process = WithTraps {
            process,
            traps: Vec::new(),
        };
        let weight = wrapped_process.weight();

        self.processes.insert(pid, wrapped_process);
        self.queues.running_queue.push(pid, weight);

        // Log metrics
        let lp = LineProtocolBuilder::new()
            .measurement("tf_scheduler_event")
            .tag("event_type", "register_process")
            .tag("pid", &pid.to_string())
            .tag("qos", &qos.to_string())
            .field("weight", weight as i64)
            .timestamp(current_time())
            .close_line()
            .build();
        let lp_str = BytesWrapper::from(lp).to_string();
        tracing::info!(target: "metrics", msg = %lp_str);
    }

    fn remove_process(&mut self, process_id: u32) {
        if self.processes.remove(&process_id).is_some() {
            self.queues.remove_all(process_id);

            // Log metrics
            let lp = LineProtocolBuilder::new()
                .measurement("tf_scheduler_event")
                .tag("event_type", "remove_process")
                .tag("pid", &process_id.to_string())
                .field("value", 1i64)
                .timestamp(current_time())
                .close_line()
                .build();
            let lp_str = BytesWrapper::from(lp).to_string();
            tracing::info!(target: "metrics", msg = %lp_str);
        }
    }

    fn get_process(&self, process_id: u32) -> Option<&Proc> {
        self.processes.get(&process_id).map(|p| &p.process)
    }

    async fn schedule(&mut self) -> Result<Vec<SchedulingDecision>> {
        DecisionEngine::make_decisions(
            &self.processes,
            &self.queues.running_queue,
            &self.queues.sleep_queue,
            &self.queues.trap_wait_queue,
        )
    }

    fn done_decision(&mut self, decision: &SchedulingDecision) {
        match decision {
            SchedulingDecision::Resume(pid) => {
                if let Some(process) = self.processes.get(pid) {
                    let weight = process.weight();
                    self.queues
                        .move_process(*pid, QueueType::Sleep, QueueType::Running, weight);
                }
            }
            SchedulingDecision::Release(pid) => {
                if let Some(process) = self.processes.get(pid) {
                    let weight = process.weight();
                    self.queues
                        .move_process(*pid, QueueType::Running, QueueType::Sleep, weight);
                }
            }
            SchedulingDecision::Wake(_, _, _) => {
                // Handle trap wake-up logic
                // This would involve moving from trap_wait_queue to running_queue
                // and cleaning up resolved traps
            }
            SchedulingDecision::Pause(_) => {
                // Handle pause logic if needed
            }
        }
    }

    async fn on_trap(
        &mut self,
        process_id: u32,
        _trap_id: u64,
        frame: Arc<TrapFrame>,
        waker: Box<dyn Waker>,
    ) {
        if let Some(process) = self.processes.get_mut(&process_id) {
            let frame_debug = format!("{frame:?}");

            process.traps.push(Trap {
                frame,
                waker,
                round: 1,
            });
            let weight = process.weight();

            // Move to trap wait queue
            self.queues.remove_all(process_id);
            self.queues.trap_wait_queue.push(process_id, weight);

            // Log metrics
            let lp = LineProtocolBuilder::new()
                .measurement("tf_scheduler_event")
                .tag("event_type", "on_trap")
                .tag("pid", &process_id.to_string())
                .field("new_weight", weight as i64)
                .field("trap_details", frame_debug.as_str())
                .timestamp(current_time())
                .close_line()
                .build();
            let lp_str = BytesWrapper::from(lp).to_string();
            tracing::info!(target: "metrics", msg = %lp_str);
        }
    }
}

/// Statistics about the scheduler state
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub total_processes: usize,
    pub running_processes: usize,
    pub sleeping_processes: usize,
    pub trap_waiting_processes: usize,
}
