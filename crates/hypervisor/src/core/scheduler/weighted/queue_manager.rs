//! Queue management for the weighted scheduler

use priority_queue::PriorityQueue;

/// Manages multiple priority queues for different process states
pub struct QueueManager {
    pub running_queue: PriorityQueue<u32, u32>,
    pub sleep_queue: PriorityQueue<u32, u32>,
    pub trap_wait_queue: PriorityQueue<u32, u32>,
}

impl Default for QueueManager {
    fn default() -> Self {
        Self::new()
    }
}

impl QueueManager {
    pub fn new() -> Self {
        Self {
            running_queue: PriorityQueue::new(),
            sleep_queue: PriorityQueue::new(),
            trap_wait_queue: PriorityQueue::new(),
        }
    }

    /// Move a process from one queue to another
    pub fn move_process(&mut self, pid: u32, from: QueueType, to: QueueType, weight: u32) {
        // Remove from source queue
        match from {
            QueueType::Running => {
                self.running_queue.remove(&pid);
            }
            QueueType::Sleep => {
                self.sleep_queue.remove(&pid);
            }
            QueueType::TrapWait => {
                self.trap_wait_queue.remove(&pid);
            }
        }

        // Add to destination queue
        match to {
            QueueType::Running => {
                self.running_queue.push(pid, weight);
            }
            QueueType::Sleep => {
                self.sleep_queue.push(pid, weight);
            }
            QueueType::TrapWait => {
                self.trap_wait_queue.push(pid, weight);
            }
        }
    }

    /// Remove a process from all queues
    pub fn remove_all(&mut self, pid: u32) {
        self.running_queue.remove(&pid);
        self.sleep_queue.remove(&pid);
        self.trap_wait_queue.remove(&pid);
    }
}

#[derive(Debug, Clone, Copy)]
pub enum QueueType {
    Running,
    Sleep,
    #[allow(dead_code)]
    TrapWait,
}
