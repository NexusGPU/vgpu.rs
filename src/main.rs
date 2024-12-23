mod hypervisor;
mod process;
mod scheduler;

use anyhow::Result;
use hypervisor::Hypervisor;
use process::GpuResources;
use scheduler::fifo::FifoScheduler;
use std::time::Duration;

fn main() -> Result<()> {
    // Create a FIFO scheduler with GPU resource limits
    let scheduler = FifoScheduler::new(GpuResources {
        memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB memory
        compute_percentage: 100,
    });

    // Create hypervisor with 1-second scheduling interval
    let mut hypervisor = Hypervisor::new(Box::new(scheduler), Duration::from_secs(1));

    // Start scheduling loop
    hypervisor.run()?;

    Ok(())
}
