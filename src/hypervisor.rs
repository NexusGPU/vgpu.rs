use anyhow::Result;
use std::{sync::{Arc, atomic::{AtomicBool, Ordering}}, thread, time::Duration};

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
    pub fn remove_process(&mut self, process_id: &str) -> Result<()> {
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
                    SchedulingDecision::Continue => continue,
                    SchedulingDecision::Pause(id) => {
                        tracing::info!("Pausing process {}", id);
                        // Actual pause logic to be implemented here
                    }
                    SchedulingDecision::Release(id) => {
                        tracing::info!("Releasing process {}", id);
                        // Actual release logic to be implemented here
                    }
                    SchedulingDecision::Resume(id) => {
                        tracing::info!("Resuming process {}", id);
                        // Actual resume logic to be implemented here
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
    use std::sync::Mutex;
    use crate::process::tests::MockGpuProcess as MockProcess;

    // Mock scheduler for testing
    struct MockScheduler {
        processes: Vec<Arc<dyn GpuProcess>>,
        schedule_calls: Arc<Mutex<u32>>,
        next_decisions: Arc<Mutex<Vec<Vec<SchedulingDecision>>>>,
    }

    impl MockScheduler {
        fn new() -> Self {
            Self {
                processes: Vec::new(),
                schedule_calls: Arc::new(Mutex::new(0)),
                next_decisions: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn get_schedule_calls(&self) -> u32 {
            *self.schedule_calls.lock().unwrap()
        }

        fn set_next_decisions(&self, decisions: Vec<SchedulingDecision>) {
            self.next_decisions.lock().unwrap().push(decisions);
        }
    }

    impl GpuScheduler for MockScheduler {
        fn add_process(&mut self, process: Arc<dyn GpuProcess>) -> Result<()> {
            self.processes.push(process);
            Ok(())
        }

        fn remove_process(&mut self, process_id: &str) -> Result<()> {
            self.processes.retain(|p| p.id() != process_id);
            Ok(())
        }

        fn schedule(&mut self) -> Result<Vec<SchedulingDecision>> {
            let mut calls = self.schedule_calls.lock().unwrap();
            *calls += 1;
            
            let mut decisions = self.next_decisions.lock().unwrap();
            if decisions.is_empty() {
                Ok(vec![SchedulingDecision::Continue])
            } else {
                Ok(decisions.remove(0))
            }
        }
    }

    #[test]
    fn test_hypervisor_process_management() -> Result<()> {
        let scheduler = MockScheduler::new();
        let mut hypervisor = Hypervisor::new(
            Box::new(scheduler),
            Duration::from_millis(100),
        );

        // Test adding process
        let process = Arc::new(MockProcess::new("test1", 2048, 75));
        hypervisor.add_process(process.clone())?;

        // Test removing process
        hypervisor.remove_process(&process.id())?;

        Ok(())
    }

    #[test]
    fn test_hypervisor_scheduling_decisions() {
        let scheduler = MockScheduler::new();
        let schedule_calls = scheduler.schedule_calls.clone();
        
        // Create test decisions
        let decisions = vec![
            SchedulingDecision::Pause("test1".to_string()),
            SchedulingDecision::Release("test2".to_string()),
            SchedulingDecision::Resume("test3".to_string()),
        ];
        scheduler.set_next_decisions(decisions.clone());

        let mut hypervisor = Hypervisor::new(
            Box::new(scheduler),
            Duration::from_millis(10),
        );

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
        let mut hypervisor = Hypervisor::new(
            Box::new(scheduler),
            interval,
        );

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
        assert!(calls >= 2, "Scheduler should have been called multiple times");
    }
}
