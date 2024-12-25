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
    collections::HashMap,
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

    // Create a FIFO scheduler with GPU resource limits for all available GPUs
    let mut gpu_limits = HashMap::new();
    let device_count = nvml.device_count()?;

    for i in 0..device_count {
        let device = nvml.device_by_index(i)?;
        let memory_info = device.memory_info()?;
        let uuid = device.uuid()?;

        tracing::info!("Found GPU {}: {}", i, uuid);

        gpu_limits.insert(
            uuid,
            GpuResources {
                memory_bytes: memory_info.total,
                compute_percentage: 100,
            },
        );
    }

    let scheduler = FifoScheduler::new(gpu_limits);

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
    hypervisor.run();

    Ok(())
}
