mod gpu_observer;
mod hypervisor;
mod logging;
mod metrics;
mod process;
mod scheduler;
mod worker_watcher;

use anyhow::Result;
use clap::{command, Parser};
use gpu_observer::GpuObserver;
use hypervisor::Hypervisor;
use nvml_wrapper::Nvml;
use process::GpuResources;
use scheduler::weighted::WeightedScheduler;
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

    #[arg(long, value_hint = clap::ValueHint::FilePath)]
    gpu_metrics_file: Option<PathBuf>,

    #[arg(long, default_value = "10")]
    metrics_batch_size: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let _guard = logging::init(cli.gpu_metrics_file);

    let nvml = Arc::new(match Nvml::init() {
        Ok(nvml) => Ok(nvml),
        Err(_) => Nvml::builder()
            .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
            .init(),
    }?);

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

    let scheduler = WeightedScheduler::new();
    // Create hypervisor with 1-second scheduling interval
    let hypervisor = Arc::new(Hypervisor::new(scheduler, Duration::from_secs(1)));

    let gpu_observer = GpuObserver::create(nvml.clone(), Duration::from_secs(1));
    let worker_pid_mapping = Arc::new(RwLock::new(HashMap::new()));
    metrics::output_metrics(
        gpu_observer.clone(),
        worker_pid_mapping.clone(),
        cli.metrics_batch_size,
    );

    // Ensure socket directory exists
    std::fs::create_dir_all(&cli.sock_path)?;

    let _ = std::thread::Builder::new()
        .name("worker watcher".into())
        .spawn({
            let hypervisor = hypervisor.clone();
            || {
                let watcher = WorkerWatcher::new(cli.sock_path, hypervisor, worker_pid_mapping)
                    .expect("new worker watcher");
                watcher.run(gpu_observer);
            }
        });

    let _ = std::thread::Builder::new()
        .name("trap server".into())
        .spawn({
            let hypervisor = hypervisor.clone();
            move || {
                let mut trap_server = trap::ipc::IpcTrapServer::new(hypervisor.clone())?;
                trap_server.run()
            }
        });

    // Start scheduling loop
    hypervisor.run();

    Ok(())
}
