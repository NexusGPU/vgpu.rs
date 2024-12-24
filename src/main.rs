mod hypervisor;
mod logging;
mod process;
mod scheduler;
mod worker_watcher;

use anyhow::Result;
use clap::{command, Parser};
use hypervisor::Hypervisor;
use nvml_wrapper::Nvml;
use process::GpuResources;
use scheduler::fifo::FifoScheduler;
use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Duration,
};
use worker_watcher::WorkerWatcher;

#[derive(Parser)]
#[command(about, long_about)]
struct Cli {
    #[arg(long, value_hint = clap::ValueHint::DirPath)]
    sock_path: PathBuf,
}

fn main() -> Result<()> {
    logging::init();

    let cli = Cli::parse();

    let nvml = Arc::new(Nvml::init()?);
    let device = nvml.device_by_index(0)?;
    let memory_info = device.memory_info()?;

    // Create a FIFO scheduler with GPU resource limits
    let scheduler = FifoScheduler::new(GpuResources {
        memory_bytes: memory_info.total,
        compute_percentage: 100,
    });

    // Create hypervisor with 1-second scheduling interval
    let hypervisor = Arc::new(Hypervisor::new(
        Box::new(RwLock::new(scheduler)),
        Duration::from_secs(1),
    ));

    let _ = std::thread::Builder::new()
        .name("worker watcher".into())
        .spawn({
            let hypervisor = hypervisor.clone();
            || {
                let watcher =
                    WorkerWatcher::new(cli.sock_path, hypervisor).expect("new worker watcher");
                watcher.run(nvml);
            }
        });

    // Start scheduling loop
    hypervisor.run()?;

    Ok(())
}
