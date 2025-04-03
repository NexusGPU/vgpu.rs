use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    thread,
    time::Duration,
};

use crate::process::GpuProcess;
use crate::scheduler::{GpuScheduler, SchedulingDecision};

pub(crate) struct Hypervisor {
    scheduler: Box<RwLock<dyn GpuScheduler>>,
    scheduling_interval: Duration,
    pub(crate) worker_pid_mapping: RwLock<HashMap<u32, String>>,
}

impl Hypervisor {
    pub(crate) fn new(
        scheduler: Box<RwLock<dyn GpuScheduler>>,
        scheduling_interval: Duration,
    ) -> Self {
        Self {
            scheduler,
            scheduling_interval,
            worker_pid_mapping: Default::default(),
        }
    }

    /// Add a new process to hypervisor
    pub(crate) fn add_process(&self, worker_name: String, process: Arc<dyn GpuProcess>) {
        self.worker_pid_mapping
            .write()
            .expect("poisoned")
            .insert(process.id(), worker_name);
        self.scheduler
            .write()
            .expect("poisoned")
            .add_process(process);
    }

    /// Remove a process from hypervisor
    pub(crate) fn remove_process(&self, process_id: u32) {
        self.worker_pid_mapping
            .write()
            .expect("poisoned")
            .remove(&process_id);
        self.scheduler
            .write()
            .expect("poisoned")
            .remove_process(process_id);
    }

    /// Get a process by id
    pub(crate) fn get_process(&self, process_id: u32) -> Option<Arc<dyn GpuProcess>> {
        self.scheduler
            .read()
            .expect("poisoned")
            .get_process(process_id)
    }

    pub(crate) fn schedule_once(&self) {
        // Execute scheduling decisions
        let decisions = match self.scheduler.write().expect("poisoned").schedule() {
            Ok(d) => d,
            Err(e) => {
                tracing::error!("scheduling error: {}", e);
                return;
            }
        };

        // Apply scheduling decisions
        for decision in decisions {
            match decision {
                SchedulingDecision::Pause(id) => {
                    tracing::info!("pausing process {}", id);
                    if let Some(process) = self.scheduler.read().expect("poisoned").get_process(id)
                    {
                        if let Err(e) = process.pause() {
                            tracing::warn!("failed to pause process {}: {}", id, e);
                        }
                    }
                }
                SchedulingDecision::Release(id) => {
                    tracing::info!("releasing process {}", id);
                    if let Some(process) = self.scheduler.read().expect("poisoned").get_process(id)
                    {
                        if let Err(e) = process.release() {
                            tracing::warn!("failed to release process {}: {}", id, e);
                        }
                    }
                }
                SchedulingDecision::Resume(id) => {
                    tracing::info!("resuming process {}", id);
                    if let Some(process) = self.scheduler.read().expect("poisoned").get_process(id)
                    {
                        if let Err(e) = process.resume() {
                            tracing::warn!("failed to resume process {}: {}", id, e);
                        }
                    }
                }
            }
        }
    }

    /// Start the scheduling loop
    pub(crate) fn run(&self) {
        loop {
            self.schedule_once();
            // Sleep for the scheduling interval
            thread::sleep(self.scheduling_interval);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::tests::MockGpuProcess as MockProcess;
    use anyhow::Result;
    use std::sync::mpsc;
    use std::{collections::HashMap, sync::Mutex};

    // Mock scheduler for testing
    struct MockScheduler {
        processes: HashMap<u32, Arc<dyn GpuProcess>>,
        schedule_calls: Arc<Mutex<u32>>,
        next_decisions: Arc<Mutex<Vec<Vec<SchedulingDecision>>>>,
        control_rx: Arc<Mutex<Option<mpsc::Receiver<()>>>>,
    }

    impl MockScheduler {
        fn new() -> Self {
            Self {
                processes: HashMap::new(),
                schedule_calls: Arc::new(Mutex::new(0)),
                next_decisions: Arc::new(Mutex::new(Vec::new())),
                control_rx: Arc::new(Mutex::new(None)),
            }
        }

        fn set_next_decisions(&self, decisions: Vec<SchedulingDecision>) {
            self.next_decisions.lock().unwrap().push(decisions);
        }

        fn set_control_channel(&self, rx: mpsc::Receiver<()>) {
            *self.control_rx.lock().unwrap() = Some(rx);
        }
    }

    impl GpuScheduler for MockScheduler {
        fn add_process(&mut self, process: Arc<dyn GpuProcess>) {
            self.processes.insert(process.id(), process.clone());
        }

        fn remove_process(&mut self, process_id: u32) {
            self.processes.remove(&process_id);
        }

        fn schedule(&mut self) -> Result<Vec<SchedulingDecision>> {
            let mut calls = self.schedule_calls.lock().unwrap();
            *calls += 1;

            // Check if we should stop
            if let Some(rx) = self.control_rx.lock().unwrap().as_ref() {
                if rx.try_recv().is_ok() {
                    return Ok(vec![]);
                }
            }

            let mut decisions = self.next_decisions.lock().unwrap();
            if decisions.is_empty() {
                Ok(vec![])
            } else {
                Ok(decisions.remove(0))
            }
        }

        fn get_process(&self, process_id: u32) -> Option<Arc<dyn GpuProcess>> {
            self.processes.get(&process_id).cloned()
        }
    }

    #[test]
    fn test_hypervisor_process_management() {
        let scheduler = MockScheduler::new();
        let hypervisor =
            Hypervisor::new(Box::new(RwLock::new(scheduler)), Duration::from_millis(100));

        // Test adding process
        let process = Arc::new(MockProcess::new(1, 2048, 75));
        hypervisor.add_process("process".to_string(), process.clone());

        // Test removing process
        hypervisor.remove_process(process.id());
    }

    #[test]
    fn test_hypervisor_scheduling_decisions() {
        let scheduler = MockScheduler::new();
        let schedule_calls = scheduler.schedule_calls.clone();

        // Create test decisions
        let decisions = vec![
            SchedulingDecision::Pause(1),
            SchedulingDecision::Release(2),
            SchedulingDecision::Resume(3),
        ];
        scheduler.set_next_decisions(decisions.clone());

        // Setup control channel
        let (tx, rx) = mpsc::channel();
        scheduler.set_control_channel(rx);

        let hypervisor =
            Hypervisor::new(Box::new(RwLock::new(scheduler)), Duration::from_millis(10));

        // Add some test processes
        let process1 = Arc::new(MockProcess::new(1, 2048, 75));
        let process2 = Arc::new(MockProcess::new(2, 1024, 50));
        let process3 = Arc::new(MockProcess::new(3, 4096, 90));

        hypervisor.add_process("process1".to_string(), process1.clone());
        hypervisor.add_process("process2".to_string(), process2.clone());
        hypervisor.add_process("process3".to_string(), process3.clone());

        hypervisor.schedule_once();
        // Signal to stop
        tx.send(()).unwrap();

        // Verify scheduler was called
        let calls = *schedule_calls.lock().unwrap();
        assert!(calls > 0, "Scheduler should have been called");
    }
}
