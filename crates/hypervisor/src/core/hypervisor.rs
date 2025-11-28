use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::core::process::{GpuProcess, Worker};
use crate::core::scheduler::{GpuScheduler, Scheduler, SchedulingDecision};
use trap::{TrapAction, TrapError, TrapFrame, TrapHandler, Waker};

pub type HypervisorType = Hypervisor<Worker, Scheduler>;

pub struct Hypervisor<Proc: GpuProcess + Clone, Sched: GpuScheduler<Proc>> {
    scheduler: Arc<Mutex<Sched>>,
    scheduling_interval: Duration,
    _marker: PhantomData<Proc>,
}

impl<Proc: GpuProcess + Clone, Sched: GpuScheduler<Proc>> Hypervisor<Proc, Sched> {
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
        let scheduled_actions = {
            let mut scheduler = self.scheduler.lock().await;
            let decisions = match scheduler.schedule().await {
                Ok(d) => d,
                Err(e) => {
                    tracing::error!("scheduling error: {}", e);
                    return;
                }
            };

            decisions
                .into_iter()
                .map(|decision| {
                    let process = match &decision {
                        SchedulingDecision::Pause(pid)
                        | SchedulingDecision::Release(pid)
                        | SchedulingDecision::Resume(pid) => scheduler.get_process(*pid).cloned(),
                        SchedulingDecision::Wake(_, _, _) => None,
                    };

                    let done_decision = match &decision {
                        SchedulingDecision::Pause(pid) => SchedulingDecision::Pause(*pid),
                        SchedulingDecision::Release(pid) => SchedulingDecision::Release(*pid),
                        SchedulingDecision::Resume(pid) => SchedulingDecision::Resume(*pid),
                        SchedulingDecision::Wake(_, trap_id, action) => {
                            SchedulingDecision::Wake(Box::new(NoopWaker), *trap_id, action.clone())
                        }
                    };

                    (decision, done_decision, process)
                })
                .collect::<Vec<_>>()
        };

        for (decision, done_decision, process) in scheduled_actions {
            match (decision, process) {
                (SchedulingDecision::Pause(id), Some(proc)) => {
                    tracing::info!("pausing process {}", id);
                    if let Err(e) = proc.pause().await {
                        tracing::warn!("failed to pause process {}: {}", id, e);
                    }
                }
                (SchedulingDecision::Release(id), Some(proc)) => {
                    tracing::info!("releasing process {}", id);
                    if let Err(e) = proc.release().await {
                        tracing::warn!("failed to release process {}: {}", id, e);
                    }
                }
                (SchedulingDecision::Resume(id), Some(proc)) => {
                    tracing::info!("resuming process {}", id);
                    if let Err(e) = proc.resume().await {
                        tracing::warn!("failed to resume process {}: {}", id, e);
                    }
                }
                (SchedulingDecision::Pause(id), None)
                | (SchedulingDecision::Release(id), None)
                | (SchedulingDecision::Resume(id), None) => {
                    tracing::warn!("scheduler decision references unknown process {}", id);
                }
                (SchedulingDecision::Wake(waker, trap_id, action), _) => {
                    tracing::info!("waking up trapped process");
                    if let Err(e) = waker.send(trap_id, action).await {
                        tracing::warn!("failed to wake trapped process: {}", e);
                    }
                }
            }

            self.scheduler.lock().await.done_decision(&done_decision);
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
impl<Proc: GpuProcess + Clone, Sched: GpuScheduler<Proc> + Send + 'static> TrapHandler
    for Hypervisor<Proc, Sched>
{
    async fn handle_trap(&self, pid: u32, trap_id: u64, frame: &TrapFrame, waker: Box<dyn Waker>) {
        // Handle the trap event - spawn async task for lock acquisition
        // Create Arc for frame to avoid expensive cloning
        let frame_arc = Arc::new(frame.clone());
        let scheduler = self.scheduler.clone();
        scheduler
            .lock()
            .await
            .on_trap(pid, trap_id, frame_arc, waker)
            .await;
    }
}

struct NoopWaker;

#[async_trait::async_trait]
impl Waker for NoopWaker {
    async fn send(&self, _: u64, _: TrapAction) -> Result<(), TrapError> {
        Ok(())
    }
}
