use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use crate::process::GpuProcess;
use crate::scheduler::{GpuScheduler, SchedulingDecision};

pub(crate) struct Hypervisor<Proc: GpuProcess, Sched: GpuScheduler<Proc>> {
    scheduler: Arc<Mutex<Sched>>,
    scheduling_interval: Duration,
    _marker: PhantomData<Proc>,
}

impl<Proc: GpuProcess, Sched: GpuScheduler<Proc>> Hypervisor<Proc, Sched> {
    pub(crate) fn new(scheduler: Sched, scheduling_interval: Duration) -> Self {
        Self {
            scheduler: Arc::new(Mutex::new(scheduler)),
            scheduling_interval,
            _marker: PhantomData,
        }
    }

    /// Add a new process to hypervisor
    pub(crate) fn add_process(&self, process: Proc) {
        self.scheduler
            .lock()
            .expect("poisoned")
            .add_process(process);
    }

    /// Remove a process from hypervisor
    pub(crate) fn remove_process(&self, process_id: u32) {
        self.scheduler
            .lock()
            .expect("poisoned")
            .remove_process(process_id);
    }

    /// Get a process by id
    pub(crate) fn process_exists(&self, process_id: u32) -> bool {
        self.scheduler
            .lock()
            .expect("poisoned")
            .get_process(process_id)
            .is_some()
    }

    pub(crate) fn schedule_once(&self) {
        let mut scheduler = self.scheduler.lock().expect("poisoned");
        // Execute scheduling decisions
        let decisions = match scheduler.schedule() {
            Ok(d) => d,
            Err(e) => {
                tracing::error!("scheduling error: {}", e);
                return;
            }
        };

        // Apply scheduling decisions
        for decision in decisions.iter() {
            match decision {
                SchedulingDecision::Pause(id) => {
                    tracing::info!("pausing process {}", id);
                    if let Some(process) = scheduler.get_process(*id) {
                        if let Err(e) = process.pause() {
                            tracing::warn!("failed to pause process {}: {}", id, e);
                            continue;
                        }
                    }
                }
                SchedulingDecision::Release(id) => {
                    tracing::info!("releasing process {}", id);
                    if let Some(process) = scheduler.get_process(*id) {
                        if let Err(e) = process.release() {
                            tracing::warn!("failed to release process {}: {}", id, e);
                            continue;
                        }
                    }
                }
                SchedulingDecision::Resume(id) => {
                    tracing::info!("resuming process {}", id);
                    if let Some(process) = scheduler.get_process(*id) {
                        if let Err(e) = process.resume() {
                            tracing::warn!("failed to resume process {}: {}", id, e);
                            continue;
                        }
                    }
                }
                SchedulingDecision::Wake(waker, arg) => {
                    tracing::info!("waking up tapped process");
                    let _ = waker.send(arg.clone());
                }
            }
            scheduler.done_decision(decision);
        }
    }

    /// Start the scheduling loop
    pub(crate) fn run(&self) {
        let scheduling_interval = self.scheduling_interval;
        // let trap_server = IpcTrapHandler::create_server(handler).un;

        loop {
            self.schedule_once();
            // Sleep for the scheduling interval
            thread::sleep(scheduling_interval);
        }
    }
}

impl<Proc: GpuProcess, Sched: GpuScheduler<Proc>> trap::TrapHandler for Hypervisor<Proc, Sched> {
    fn handle_trap(&self, pid: u32, frame: &trap::TrapFrame, waker: trap::Waker) {
        // Handle the trap event
        self.scheduler
            .lock()
            .expect("poisoned")
            .on_trap(pid, frame, waker);
    }
}
