mod api;
mod config;
mod gpu_observer;
mod hypervisor;
mod k8s;
mod logging;
mod metrics;
mod process;
mod scheduler;
mod worker_watcher;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use anyhow::Result;
use api::ApiServer;
use api::PodStorage;
use clap::command;
use clap::Parser;
use gpu_observer::GpuObserver;
use hypervisor::Hypervisor;
use k8s::PodWatcher;
use k8s::WorkerUpdate;
use nvml_wrapper::Nvml;
use process::GpuResources;
use scheduler::weighted::WeightedScheduler;
use tokio::sync::oneshot;
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

    #[arg(long, help = "Enable Kubernetes pod monitoring")]
    enable_k8s: bool,

    #[arg(
        long,
        help = "Kubernetes namespace to monitor (empty for all namespaces)"
    )]
    k8s_namespace: Option<String>,

    #[arg(
        long,
        env = "NODE_NAME",
        help = "Node name for filtering pods to this node only"
    )]
    node_name: Option<String>,

    #[arg(
        long,
        default_value = "127.0.0.1:8080",
        help = "HTTP API server listen address"
    )]
    api_listen_addr: String,
}

#[tokio::main]
async fn main() -> Result<()> {
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
            tflops_request: None,
            tflops_limit: None,
            memory_limit: None,
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

    // Setup Kubernetes pod watcher if enabled
    let (k8s_update_sender, k8s_update_receiver) = mpsc::channel::<WorkerUpdate>();
    let (_k8s_shutdown_sender, k8s_shutdown_receiver) = oneshot::channel::<()>();

    // Setup pod storage for API queries
    let pod_storage: PodStorage = Arc::new(RwLock::new(HashMap::new()));

    // Setup API server shutdown channel
    let (_api_shutdown_sender, api_shutdown_receiver) = oneshot::channel::<()>();

    // Start GPU observer task
    let gpu_observer_task = {
        let gpu_observer = gpu_observer.clone();
        tokio::spawn(async move {
            tracing::info!("Starting GPU observer task");
            gpu_observer.run_async(Duration::from_secs(1)).await;
        })
    };

    // Start metrics collection task
    let metrics_task = {
        let gpu_observer = gpu_observer.clone();
        let worker_pid_mapping = worker_pid_mapping.clone();
        let metrics_batch_size = cli.metrics_batch_size;
        tokio::spawn(async move {
            tracing::info!("Starting metrics collection task");
            metrics::run_metrics_async(
                gpu_observer,
                worker_pid_mapping,
                metrics_batch_size,
                gpu_node,
                gpu_pool,
            )
            .await;
        })
    };

    // create trap server
    let trap_server = Arc::new(
        trap::ipc::IpcTrapServer::new(hypervisor.clone()).expect("failed to create trap server"),
    );

    // Create the worker watcher instance to be shared by both tasks
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
                    // Spawn a standalone task to wait for client connection
                    let trap_server = trap_server.clone();
                    let ipc_path = cli.ipc_path.clone();
                    tokio::spawn(async move {
                        tracing::info!("waiting for client with PID: {}", pid);
                        match tokio::task::spawn_blocking(move || {
                            trap_server.wait_client(ipc_path, pid)
                        })
                        .await
                        {
                            Ok(Ok(_)) => tracing::info!("client with PID {} connected", pid),
                            Ok(Err(e)) => {
                                tracing::error!("failed to wait for client with PID {}: {}", pid, e)
                            }
                            Err(e) => tracing::error!(
                                "blocking task panicked while waiting for client with PID {}: {}",
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

    // Start worker watcher loop task (directory polling)
    let watcher_loop_task = {
        let watcher = watcher.clone();
        let sock_path = sock_path.clone();
        tokio::spawn(async move {
            tracing::info!("Starting worker watcher loop task");
            watcher.run_watcher_loop_async(sock_path).await;
        })
    };

    // Start worker watcher event handling task
    let worker_task = {
        let watcher = watcher.clone();
        let gpu_observer = gpu_observer.clone();
        tokio::spawn(async move {
            tracing::info!("Starting worker watcher event handler task");
            watcher.run_async(gpu_observer).await;
        })
    };

    // Start trap server task
    let trap_task = {
        let hypervisor = hypervisor.clone();
        tokio::spawn(async move {
            tracing::info!("Starting trap server task");
            let trap_server =
                trap::ipc::IpcTrapServer::new(hypervisor).expect("failed to create trap server");
            match tokio::task::spawn_blocking(move || trap_server.run()).await {
                Ok(Ok(_)) => tracing::info!("trap server completed successfully"),
                Ok(Err(e)) => tracing::error!("trap server failed: {}", e),
                Err(e) => tracing::error!("trap server blocking task panicked: {}", e),
            }
        })
    };

    // Start Kubernetes pod watcher if enabled
    let k8s_task = if cli.enable_k8s {
        let k8s_namespace = cli.k8s_namespace.clone();
        let node_name = cli.node_name.clone();
        Some(tokio::spawn(async move {
            tracing::info!("Starting Kubernetes pod watcher task");
            match PodWatcher::new(k8s_namespace, node_name, k8s_update_sender).await {
                Ok(watcher) => {
                    if let Err(e) = watcher.run(k8s_shutdown_receiver).await {
                        tracing::error!("Kubernetes pod watcher failed: {e:?}");
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to create Kubernetes pod watcher: {e:?}");
                }
            }
        }))
    } else {
        // Drop the receiver to avoid blocking
        drop(k8s_shutdown_receiver);
        None
    };

    // Start worker update processor for Kubernetes events
    let k8s_processor_task = if cli.enable_k8s {
        let pod_storage = pod_storage.clone();
        Some(tokio::spawn(async move {
            tracing::info!("Starting Kubernetes update processor task");
            for update in k8s_update_receiver {
                match update {
                    WorkerUpdate::PodCreated {
                        pod_name,
                        namespace,
                        annotations,
                        node_name,
                    } => {
                        tracing::info!(
                            "Pod created: {}/{} with annotations: {:?}, node: {:?}",
                            namespace,
                            pod_name,
                            annotations,
                            node_name
                        );
                        api::update_pod_storage(
                            &pod_storage,
                            pod_name,
                            namespace,
                            node_name,
                            annotations,
                        );
                    }
                    WorkerUpdate::PodUpdated {
                        pod_name,
                        namespace,
                        annotations,
                        node_name,
                    } => {
                        tracing::info!(
                            "Pod updated: {}/{} with annotations: {:?}, node: {:?}",
                            namespace,
                            pod_name,
                            annotations,
                            node_name
                        );
                        api::update_pod_storage(
                            &pod_storage,
                            pod_name,
                            namespace,
                            node_name,
                            annotations,
                        );
                    }
                    WorkerUpdate::PodDeleted {
                        pod_name,
                        namespace,
                    } => {
                        tracing::info!("Pod deleted: {}/{}", namespace, pod_name);
                        api::remove_pod_from_storage(&pod_storage, &pod_name, &namespace);
                    }
                }
            }
        }))
    } else {
        None
    };

    // Start HTTP API server task
    let api_task = {
        let pod_storage = pod_storage.clone();
        let api_listen_addr = cli.api_listen_addr.clone();
        tokio::spawn(async move {
            tracing::info!("Starting HTTP API server task");
            let api_server = ApiServer::new(pod_storage, api_listen_addr);
            if let Err(e) = api_server.run(api_shutdown_receiver).await {
                tracing::error!("HTTP API server failed: {e:?}");
            }
        })
    };

    // Start hypervisor task
    let hypervisor_task = {
        let hypervisor = hypervisor.clone();
        tokio::spawn(async move {
            tracing::info!("Starting hypervisor task");
            hypervisor.run_async().await;
        })
    };

    // Use tokio::select! to wait for any task to complete (which likely means a panic or termination)
    match (k8s_task, k8s_processor_task) {
        (Some(k8s_task), Some(k8s_processor_task)) => {
            tokio::select! {
                result = gpu_observer_task => {
                    tracing::error!("GPU observer task completed: {:?}", result);
                }
                result = metrics_task => {
                    tracing::error!("Metrics collection task completed: {:?}", result);
                }
                result = watcher_loop_task => {
                    tracing::error!("Worker watcher loop task completed: {:?}", result);
                }
                result = worker_task => {
                    tracing::error!("Worker watcher task completed: {:?}", result);
                }
                result = trap_task => {
                    tracing::error!("Trap server task completed: {:?}", result);
                }
                result = k8s_task => {
                    tracing::error!("Kubernetes pod watcher task completed: {:?}", result);
                }
                result = k8s_processor_task => {
                    tracing::error!("Kubernetes update processor task completed: {:?}", result);
                }
                result = api_task => {
                    tracing::error!("HTTP API server task completed: {:?}", result);
                }
                result = hypervisor_task => {
                    tracing::error!("Hypervisor task completed: {:?}", result);
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Received Ctrl+C, shutting down...");
                }
            }
        }
        (Some(k8s_task), None) => {
            tokio::select! {
                result = gpu_observer_task => {
                    tracing::error!("GPU observer task completed: {:?}", result);
                }
                result = metrics_task => {
                    tracing::error!("Metrics collection task completed: {:?}", result);
                }
                result = watcher_loop_task => {
                    tracing::error!("Worker watcher loop task completed: {:?}", result);
                }
                result = worker_task => {
                    tracing::error!("Worker watcher task completed: {:?}", result);
                }
                result = trap_task => {
                    tracing::error!("Trap server task completed: {:?}", result);
                }
                result = k8s_task => {
                    tracing::error!("Kubernetes pod watcher task completed: {:?}", result);
                }
                result = api_task => {
                    tracing::error!("HTTP API server task completed: {:?}", result);
                }
                result = hypervisor_task => {
                    tracing::error!("Hypervisor task completed: {:?}", result);
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Received Ctrl+C, shutting down...");
                }
            }
        }
        (None, Some(k8s_processor_task)) => {
            tokio::select! {
                result = gpu_observer_task => {
                    tracing::error!("GPU observer task completed: {:?}", result);
                }
                result = metrics_task => {
                    tracing::error!("Metrics collection task completed: {:?}", result);
                }
                result = watcher_loop_task => {
                    tracing::error!("Worker watcher loop task completed: {:?}", result);
                }
                result = worker_task => {
                    tracing::error!("Worker watcher task completed: {:?}", result);
                }
                result = trap_task => {
                    tracing::error!("Trap server task completed: {:?}", result);
                }
                result = k8s_processor_task => {
                    tracing::error!("Kubernetes update processor task completed: {:?}", result);
                }
                result = api_task => {
                    tracing::error!("HTTP API server task completed: {:?}", result);
                }
                result = hypervisor_task => {
                    tracing::error!("Hypervisor task completed: {:?}", result);
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Received Ctrl+C, shutting down...");
                }
            }
        }
        (None, None) => {
            tokio::select! {
                result = gpu_observer_task => {
                    tracing::error!("GPU observer task completed: {:?}", result);
                }
                result = metrics_task => {
                    tracing::error!("Metrics collection task completed: {:?}", result);
                }
                result = watcher_loop_task => {
                    tracing::error!("Worker watcher loop task completed: {:?}", result);
                }
                result = worker_task => {
                    tracing::error!("Worker watcher task completed: {:?}", result);
                }
                result = trap_task => {
                    tracing::error!("Trap server task completed: {:?}", result);
                }
                result = api_task => {
                    tracing::error!("HTTP API server task completed: {:?}", result);
                }
                result = hypervisor_task => {
                    tracing::error!("Hypervisor task completed: {:?}", result);
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Received Ctrl+C, shutting down...");
                }
            }
        }
    }

    Ok(())
}
