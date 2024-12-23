use anyhow::Result;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use crate::process::GpuProcess;
use crate::scheduler::{GpuScheduler, SchedulingDecision};

pub struct Hypervisor {
    scheduler: Box<dyn GpuScheduler>,
    scheduling_interval: Duration,
    running: Arc<AtomicBool>,
}

impl Hypervisor {
    pub fn new(scheduler: Box<dyn GpuScheduler>, scheduling_interval: Duration) -> Self {
        Self {
            scheduler,
            scheduling_interval,
            running: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Add a new process to hypervisor
    pub fn add_process(&mut self, process: Arc<dyn GpuProcess>) -> Result<()> {
        self.scheduler.add_process(process)
    }

    /// Remove a process from hypervisor
    pub fn remove_process(&mut self, process_id: u32) -> Result<()> {
        self.scheduler.remove_process(process_id)
    }

    /// Get the running flag for stopping the scheduler
    pub fn running(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    /// Start the scheduling loop
    pub fn run(&mut self) -> Result<()> {
        while self.running.load(Ordering::SeqCst) {
            // Execute scheduling decisions
            let decisions = self.scheduler.schedule()?;

            // Apply scheduling decisions
            for decision in decisions {
                match decision {
                    SchedulingDecision::Pause(id) => {
                        tracing::info!("pausing process {}", id);
                        if let Some(process) = self.scheduler.get_process(id) {
                            process.pause()?;
                        }
                    }
                    SchedulingDecision::Release(id) => {
                        tracing::info!("releasing process {}", id);
                        if let Some(process) = self.scheduler.get_process(id) {
                            process.release()?;
                        }
                    }
                    SchedulingDecision::Resume(id) => {
                        tracing::info!("resuming process {}", id);
                        if let Some(process) = self.scheduler.get_process(id) {
                            process.resume()?;
                        }
                    }
                }
            }

            // Sleep for the scheduling interval
            thread::sleep(self.scheduling_interval);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::tests::MockGpuProcess as MockProcess;
    use std::{collections::HashMap, sync::Mutex};

    // Mock scheduler for testing
    struct MockScheduler {
        processes: HashMap<u32, Arc<dyn GpuProcess>>,
        schedule_calls: Arc<Mutex<u32>>,
        next_decisions: Arc<Mutex<Vec<Vec<SchedulingDecision>>>>,
    }

    impl MockScheduler {
        fn new() -> Self {
            Self {
                processes: HashMap::new(),
                schedule_calls: Arc::new(Mutex::new(0)),
                next_decisions: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn set_next_decisions(&self, decisions: Vec<SchedulingDecision>) {
            self.next_decisions.lock().unwrap().push(decisions);
        }
    }

    impl GpuScheduler for MockScheduler {
        fn add_process(&mut self, process: Arc<dyn GpuProcess>) -> Result<()> {
            self.processes.insert(process.id(), process.clone());
            Ok(())
        }

        fn remove_process(&mut self, process_id: u32) -> Result<()> {
            self.processes.remove(&process_id);
            Ok(())
        }

        fn schedule(&mut self) -> Result<Vec<SchedulingDecision>> {
            let mut calls = self.schedule_calls.lock().unwrap();
            *calls += 1;

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
    fn test_hypervisor_process_management() -> Result<()> {
        let scheduler = MockScheduler::new();
        let mut hypervisor = Hypervisor::new(Box::new(scheduler), Duration::from_millis(100));

        // Test adding process
        let process = Arc::new(MockProcess::new(1, 2048, 75));
        hypervisor.add_process(process.clone())?;

        // Test removing process
        hypervisor.remove_process(process.id())?;

        Ok(())
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

        let mut hypervisor = Hypervisor::new(Box::new(scheduler), Duration::from_millis(10));

        let running = hypervisor.running();

        // Run hypervisor for a short duration to test scheduling
        let _handle = thread::spawn(move || {
            hypervisor.run().unwrap();
        });

        // Give some time for the scheduler to run
        thread::sleep(Duration::from_millis(50));
        running.store(false, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(20)); // Wait for thread to finish

        // Verify scheduler was called
        let calls = *schedule_calls.lock().unwrap();
        assert!(calls > 0, "Scheduler should have been called");
    }

    #[test]
    fn test_hypervisor_scheduling_interval() {
        let scheduler = MockScheduler::new();
        let schedule_calls = scheduler.schedule_calls.clone();

        let interval = Duration::from_millis(10);
        let mut hypervisor = Hypervisor::new(Box::new(scheduler), interval);

        let running = hypervisor.running();

        // Run hypervisor for multiple intervals
        let _handle = thread::spawn(move || {
            hypervisor.run().unwrap();
        });

        // Wait for multiple scheduling intervals
        thread::sleep(interval * 5);
        running.store(false, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(20)); // Wait for thread to finish

        // Verify multiple scheduling calls were made
        let calls = *schedule_calls.lock().unwrap();
        assert!(
            calls >= 2,
            "Scheduler should have been called multiple times"
        );
    }
}
