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
    // Set up global panic hook to print detailed information and propagate to the main thread when a thread panics
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        // Call the default panic hook to print detailed information
        default_hook(panic_info);
        // Log the panic location and reason
        tracing::error!("Thread panicked: {}", panic_info);
    }));

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

    let gpu_observer = GpuObserver::create(nvml.clone());
    let worker_pid_mapping = Arc::new(RwLock::new(HashMap::new()));

    // Ensure socket directory exists
    std::fs::create_dir_all(&cli.sock_path)?;

    // Use crossbeam scoped threads to ensure all threads are joined before the main function ends
    // If any thread panics, the panic will propagate to the main thread
    crossbeam::thread::scope(|s| {
        // Start GPU observer thread
        let gpu_observer_handle = s.spawn({
            let gpu_observer = gpu_observer.clone();
            move |_| {
                tracing::info!("Starting GPU observer thread");
                gpu_observer.run(Duration::from_secs(1));
            }
        });

        // Start metrics collection thread
        let metrics_handle = s.spawn({
            let gpu_observer = gpu_observer.clone();
            let worker_pid_mapping = worker_pid_mapping.clone();
            let metrics_batch_size = cli.metrics_batch_size;
            move |_| {
                tracing::info!("Starting metrics collection thread");
                metrics::run_metrics(gpu_observer, worker_pid_mapping, metrics_batch_size);
            }
        });

        // Create the worker watcher instance to be shared by both threads
        let sock_path = cli.sock_path.clone();
        let watcher = Arc::new(
            WorkerWatcher::new(&sock_path, hypervisor.clone(), worker_pid_mapping.clone())
                .expect("new worker watcher"),
        );

        // Start worker watcher loop thread (directory polling)
        let watcher_loop_handle = s.spawn({
            let watcher = watcher.clone();
            let sock_path = sock_path.clone();
            move |_| {
                tracing::info!("Starting worker watcher loop thread");
                watcher.run_watcher_loop(sock_path);
            }
        });

        // Start worker watcher event handling thread
        let worker_handle = s.spawn({
            let watcher = watcher.clone();
            let gpu_observer = gpu_observer.clone();
            move |_| {
                tracing::info!("Starting worker watcher event handler thread");
                watcher.run(gpu_observer);
            }
        });

        // Start trap server thread
        let trap_handle = s.spawn({
            let hypervisor = hypervisor.clone();
            move |_| {
                tracing::info!("Starting trap server thread");
                let mut trap_server = trap::ipc::IpcTrapServer::new(hypervisor)
                    .expect("failed to create trap server");
                trap_server.run().expect("trap server failed");
            }
        });

        // Start hypervisor in a separate thread
        let hypervisor_handle = s.spawn({
            let hypervisor = hypervisor.clone();
            move |_| {
                tracing::info!("Starting hypervisor thread");
                hypervisor.run();
            }
        });

        // Join threads to catch any panics
        // Note: Since all these threads run in infinite loops, these join calls will only complete
        // if a thread panics or if the program receives a termination signal (e.g., Ctrl+C)
        gpu_observer_handle
            .join()
            .expect("GPU observer thread panicked");
        metrics_handle
            .join()
            .expect("Metrics collection thread panicked");
        watcher_loop_handle
            .join()
            .expect("Worker watcher loop thread panicked");
        worker_handle
            .join()
            .expect("Worker watcher thread panicked");
        trap_handle.join().expect("Trap server thread panicked");
        hypervisor_handle
            .join()
            .expect("Hypervisor thread panicked");
    })
    .expect("thread panicked");

    Ok(())
}
