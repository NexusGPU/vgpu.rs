mod config;
mod gpu_observer;
mod hypervisor;
mod logging;
mod metrics;
mod process;
mod scheduler;
mod worker_watcher;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread;
use std::time::Duration;

use anyhow::Result;
use clap::command;
use clap::Parser;
use gpu_observer::GpuObserver;
use hypervisor::Hypervisor;
use nvml_wrapper::Nvml;
use process::GpuResources;
use scheduler::weighted::WeightedScheduler;
use utils::version;
use worker_watcher::WorkerWatcher;

#[derive(Parser)]
#[command(about, long_about, version = &**version::VERSION)]
struct Cli {
    #[arg(long, value_hint = clap::ValueHint::DirPath, help = "Socket path for hypervisor to control vGPU workers, e.g. /tensor-fusion/worker/sock/")]
    sock_path: PathBuf,

    #[arg(long, value_hint = clap::ValueHint::FilePath, help = "Path for printing GPU and worker metrics, e.g. /logs/metrics.log")]
    gpu_metrics_file: Option<PathBuf>,

    #[arg(
        long,
        default_value = "10",
        help = "Number of metrics to aggregate before printing, default to 10 means aggregated every 10 seconds"
    )]
    metrics_batch_size: usize,

    #[arg(
        long,
        env = "TENSOR_FUSION_GPU_INFO_PATH",
        help = "Path for GPU info list, e.g. /etc/tensor-fusion/gpu-info.yaml"
    )]
    gpu_info_path: Option<PathBuf>,

    #[arg(
        long,
        env = "TENSOR_FUSION_IPC_SERVER_PATH",
        help = "Path for the IPC pipe used for communication between the hypervisor and worker processes, e.g. /tensor-fusion/worker/ipc"
    )]
    ipc_path: PathBuf,
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

    tracing::info!("Starting tensor-fusion-hypervisor {}", &**version::VERSION);

    let nvml = Arc::new(match Nvml::init() {
        Ok(nvml) => Ok(nvml),
        Err(_) => Nvml::builder()
            .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
            .init(),
    }?);

    let mut gpu_limits = HashMap::new();
    let mut gpu_uuid_to_name_map = HashMap::new();
    let device_count = nvml.device_count()?;

    for i in 0..device_count {
        let device = nvml.device_by_index(i)?;
        let memory_info = device.memory_info()?;
        let uuid = device.uuid()?.to_lowercase();
        let name = device.name()?;

        tracing::info!("Found GPU {}: {} ({})", i, uuid, name);
        // Store GPU name and UUID mapping for config lookup
        gpu_uuid_to_name_map.insert(uuid.clone(), name);

        gpu_limits.insert(uuid, GpuResources {
            memory_bytes: memory_info.total,
            compute_percentage: 100,
        });
    }

    let gpu_node = std::env::var("GPU_NODE_NAME").unwrap_or("unknown".to_string());
    let gpu_pool = std::env::var("TENSOR_FUSION_POOL_NAME").unwrap_or("unknown".to_string());

    // Load GPU information from config file
    if let Err(e) = config::load_gpu_info(
        gpu_uuid_to_name_map,
        cli.gpu_info_path.unwrap_or("./gpu-info.yaml".into()),
    ) {
        tracing::warn!("Failed to load GPU information: {}", e);
    }

    let scheduler = WeightedScheduler::new();
    // Create hypervisor with 1-second scheduling interval
    let hypervisor = Arc::new(Hypervisor::new(scheduler, Duration::from_secs(1)));

    let gpu_observer = GpuObserver::create(nvml.clone());
    let worker_pid_mapping = Arc::new(RwLock::new(HashMap::new()));

    // Ensure socket directory exists
    std::fs::create_dir_all(&cli.sock_path)?;

    thread::scope(|s| {
        // Start GPU observer thread
        let gpu_observer_handle = s.spawn({
            let gpu_observer = gpu_observer.clone();
            move || {
                tracing::info!("Starting GPU observer thread");
                gpu_observer.run(Duration::from_secs(1));
            }
        });

        // Start metrics collection thread
        let metrics_handle = s.spawn({
            let gpu_observer = gpu_observer.clone();
            let worker_pid_mapping = worker_pid_mapping.clone();
            let metrics_batch_size = cli.metrics_batch_size;
            move || {
                tracing::info!("Starting metrics collection thread");
                metrics::run_metrics(
                    gpu_observer,
                    worker_pid_mapping,
                    metrics_batch_size,
                    gpu_node,
                    gpu_pool,
                );
            }
        });

        // create trap server
        let trap_server = Arc::new(
            trap::ipc::IpcTrapServer::new(hypervisor.clone())
                .expect("failed to create trap server"),
        );
        // Create the worker watcher instance to be shared by both threads
        let sock_path = cli.sock_path.clone();
        let watcher = Arc::new(
            WorkerWatcher::new(
                &sock_path,
                {
                    let hypervisor = hypervisor.clone();
                    let trap_server = trap_server.clone();
                    move |pid, mut worker| {
                        if hypervisor.process_exists(pid) {
                            return;
                        }

                        if let Err(e) = worker.connect() {
                            tracing::error!("failed to connect to worker: {e}, skipped");
                            return;
                        }

                        hypervisor.add_process(worker);
                        tracing::info!("new worker added: {pid}");
                        // Spawn a standalone thread to wait for client connection
                        let trap_server = trap_server.clone();
                        let ipc_path = cli.ipc_path.clone();
                        thread::spawn(move || {
                            tracing::info!("waiting for client with PID: {}", pid);
                            match trap_server.wait_client(ipc_path, pid) {
                                Ok(_) => tracing::info!("client with PID {} connected", pid),
                                Err(e) => tracing::error!(
                                    "failed to wait for client with PID {}: {}",
                                    pid,
                                    e
                                ),
                            }
                        });
                    }
                },
                {
                    let hypervisor = hypervisor.clone();
                    let trap_server = trap_server.clone();
                    move |pid| {
                        hypervisor.remove_process(pid);
                        trap_server.remove_client(pid);
                    }
                },
                worker_pid_mapping.clone(),
            )
            .expect("new worker watcher"),
        );

        // Start worker watcher loop thread (directory polling)
        let watcher_loop_handle = thread::Builder::new()
            .name("worker-watcher-loop".into())
            .spawn_scoped(s, {
                let watcher = watcher.clone();
                let sock_path = sock_path.clone();
                move || {
                    tracing::info!("Starting worker watcher loop thread");
                    watcher.run_watcher_loop(sock_path);
                }
            })
            .expect("failed to spawn worker watcher loop thread");

        // Start worker watcher event handling thread
        let worker_handle = thread::Builder::new()
            .name("worker-watcher-event-handler".into())
            .spawn_scoped(s, {
                let watcher = watcher.clone();
                let gpu_observer = gpu_observer.clone();
                move || {
                    tracing::info!("Starting worker watcher event handler thread");
                    watcher.run(gpu_observer);
                }
            })
            .expect("failed to spawn worker watcher event handler thread");

        // Start trap server thread
        let trap_handle = thread::Builder::new()
            .name("trap-server".into())
            .spawn_scoped(s, {
                let hypervisor = hypervisor.clone();
                move || {
                    tracing::info!("Starting trap server thread");
                    let trap_server = trap::ipc::IpcTrapServer::new(hypervisor)
                        .expect("failed to create trap server");
                    trap_server.run().expect("trap server failed");
                }
            })
            .expect("failed to spawn trap server thread");

        // Start hypervisor in a separate thread
        let hypervisor_handle = s.spawn({
            let hypervisor = hypervisor.clone();
            move || {
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
    });

    Ok(())
}
