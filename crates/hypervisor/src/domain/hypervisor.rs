use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::process::GpuProcess;
use crate::scheduler::GpuScheduler;
use crate::scheduler::SchedulingDecision;

pub struct Hypervisor<Proc: GpuProcess, Sched: GpuScheduler<Proc>> {
    scheduler: Arc<Mutex<Sched>>,
    scheduling_interval: Duration,
    _marker: PhantomData<Proc>,
}

impl<Proc: GpuProcess, Sched: GpuScheduler<Proc>> Hypervisor<Proc, Sched> {
    pub fn new(scheduler: Sched, scheduling_interval: Duration) -> Self {
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
                SchedulingDecision::Wake(waker, trap_id, arg) => {
                    tracing::info!("waking up trapped process");
                    let _ = waker.send(*trap_id, arg.clone());
                }
            }
            scheduler.done_decision(decision);
        }
    }

    /// Start the scheduling loop asynchronously
    pub(crate) async fn run(&self, cancellation_token: CancellationToken) {
        let scheduling_interval = self.scheduling_interval;

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    tracing::info!("Hypervisor shutdown requested");
                    break;
                }
                _ = async {
                    self.schedule_once();
                    // Sleep for the scheduling interval
                    tokio::time::sleep(scheduling_interval).await;
                } => {
                    // Continue the loop
                }
            }
        }
    }
}

impl<Proc: GpuProcess, Sched: GpuScheduler<Proc>> trap::TrapHandler for Hypervisor<Proc, Sched> {
    fn handle_trap(
        &self,
        pid: u32,
        trap_id: u64,
        frame: &trap::TrapFrame,
        waker: Box<dyn trap::Waker>,
    ) {
        // Handle the trap event
        self.scheduler
            .lock()
            .expect("poisoned")
            .on_trap(pid, trap_id, frame, waker);
    }
}
