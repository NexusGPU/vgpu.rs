use std::{
    marker::PhantomData,
    sync::{Arc, RwLock},
    thread,
    time::Duration,
};

use fnv::FnvHashMap;

use crate::process::GpuProcess;
use crate::scheduler::{GpuScheduler, SchedulingDecision};

pub(crate) struct Hypervisor<Proc: GpuProcess, Sched: GpuScheduler<Proc>> {
    scheduler: Sched,
    scheduling_interval: Duration,
    pub(crate) worker_pid_mapping: FnvHashMap<u32, String>,
    _marker: PhantomData<Proc>,
}

impl<Proc: GpuProcess, Sched: GpuScheduler<Proc>> Hypervisor<Proc, Sched> {
    pub(crate) fn new(scheduler: Sched, scheduling_interval: Duration) -> Self {
        Self {
            scheduler,
            scheduling_interval,
            worker_pid_mapping: Default::default(),
            _marker: PhantomData,
        }
    }

    /// Add a new process to hypervisor
    pub(crate) fn add_process(&mut self, worker_name: String, process: Proc) {
        self.worker_pid_mapping.insert(process.id(), worker_name);
        self.scheduler.add_process(process);
    }

    /// Remove a process from hypervisor
    pub(crate) fn remove_process(&mut self, process_id: u32) {
        self.worker_pid_mapping.remove(&process_id);
        self.scheduler.remove_process(process_id);
    }

    /// Get a process by id
    pub(crate) fn get_process(&self, process_id: u32) -> Option<&Proc> {
        self.scheduler.get_process(process_id)
    }

    pub(crate) fn schedule_once(&mut self) {
        // Execute scheduling decisions
        let decisions = match self.scheduler.schedule() {
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
                    if let Some(process) = self.scheduler.get_process(id) {
                        if let Err(e) = process.pause() {
                            tracing::warn!("failed to pause process {}: {}", id, e);
                        }
                    }
                }
                SchedulingDecision::Release(id) => {
                    tracing::info!("releasing process {}", id);
                    if let Some(process) = self.scheduler.get_process(id) {
                        if let Err(e) = process.release() {
                            tracing::warn!("failed to release process {}: {}", id, e);
                        }
                    }
                }
                SchedulingDecision::Resume(id) => {
                    tracing::info!("resuming process {}", id);
                    if let Some(process) = self.scheduler.get_process(id) {
                        if let Err(e) = process.resume() {
                            tracing::warn!("failed to resume process {}: {}", id, e);
                        }
                    }
                }
            }
        }
    }

    /// Start the scheduling loop
    pub(crate) fn run(hypervisor: Arc<RwLock<Self>>) {
        let scheduling_interval = hypervisor.read().expect("poisoned").scheduling_interval;
        loop {
            hypervisor.write().expect("poisoned").schedule_once();
            // Sleep for the scheduling interval
            thread::sleep(scheduling_interval);
        }
    }
}
