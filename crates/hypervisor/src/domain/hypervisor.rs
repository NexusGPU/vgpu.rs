use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::process::{GpuProcess, Worker};
use crate::scheduler::SchedulingDecision;
use crate::scheduler::{GpuScheduler, Scheduler};
use trap::{TrapFrame, TrapHandler, Waker};

/// Concrete hypervisor type used throughout the application
pub type HypervisorType = Hypervisor<Worker, Scheduler>;

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
    pub(crate) async fn add_process(&self, process: Proc) {
        self.scheduler.lock().await.add_process(process);
    }

    /// Remove a process from hypervisor
    pub(crate) async fn remove_process(&self, process_id: u32) {
        self.scheduler.lock().await.remove_process(process_id);
    }

    /// Get a process by id
    pub(crate) async fn process_exists(&self, process_id: u32) -> bool {
        self.scheduler
            .lock()
            .await
            .get_process(process_id)
            .is_some()
    }

    pub(crate) async fn schedule_once(&self) {
        let mut scheduler = self.scheduler.lock().await;
        // Execute scheduling decisions
        let decisions = match scheduler.schedule().await {
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
                        if let Err(e) = process.pause().await {
                            tracing::warn!("failed to pause process {}: {}", id, e);
                            continue;
                        }
                    }
                }
                SchedulingDecision::Release(id) => {
                    tracing::info!("releasing process {}", id);
                    if let Some(process) = scheduler.get_process(*id) {
                        if let Err(e) = process.release().await {
                            tracing::warn!("failed to release process {}: {}", id, e);
                            continue;
                        }
                    }
                }
                SchedulingDecision::Resume(id) => {
                    tracing::info!("resuming process {}", id);
                    if let Some(process) = scheduler.get_process(*id) {
                        if let Err(e) = process.resume().await {
                            tracing::warn!("failed to resume process {}: {}", id, e);
                            continue;
                        }
                    }
                }
                SchedulingDecision::Wake(waker, trap_id, arg) => {
                    tracing::info!("waking up trapped process");
                    if let Err(e) = waker.send(*trap_id, arg.clone()).await {
                        tracing::warn!("failed to wake trapped process: {}", e);
                    }
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
                    self.schedule_once().await;
                    // Sleep for the scheduling interval
                    tokio::time::sleep(scheduling_interval).await;
                } => {
                    // Continue the loop
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl<Proc: GpuProcess, Sched: GpuScheduler<Proc> + Send + 'static> TrapHandler
    for Hypervisor<Proc, Sched>
{
    async fn handle_trap(&self, pid: u32, trap_id: u64, frame: &TrapFrame, waker: Box<dyn Waker>) {
        // Handle the trap event - spawn async task for lock acquisition
        // Clone frame to avoid lifetime issues
        let frame_clone = frame.clone();
        let scheduler = self.scheduler.clone();
        scheduler
            .lock()
            .await
            .on_trap(pid, trap_id, &frame_clone, waker)
            .await;
    }
}
