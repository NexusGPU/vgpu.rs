mod api;
mod config;
mod gpu_observer;
mod host_pid_probe;
mod hypervisor;
mod k8s;
mod limiter_comm;
mod logging;
mod metrics;
mod process;
mod scheduler;
mod worker_manager;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use api::ApiServer;
use api::JwtAuthConfig;
use clap::command;
use clap::Parser;
use gpu_observer::GpuObserver;
use host_pid_probe::HostPidProbe;
use hypervisor::Hypervisor;
use k8s::PodWatcher;
use k8s::WorkerUpdate;
use limiter_comm::CommandDispatcher;
use nvml_wrapper::Nvml;
use process::GpuProcess;
use process::GpuResources;
use scheduler::weighted::WeightedScheduler;
use tokio::sync::oneshot;
use utils::version;
use worker_manager::WorkerManager;

#[derive(Parser)]
#[command(about, long_about, version = &**version::VERSION)]
struct Cli {
    #[arg(
        long,
        env = "GPU_METRICS_FILE",
        value_hint = clap::ValueHint::FilePath,
        default_value = "/logs/metrics.log",
        help = "Path for printing GPU and worker metrics, e.g. /logs/metrics.log"
    )]
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
        help = "Enable Kubernetes pod monitoring",
        default_value = "true"
    )]
    enable_k8s: bool,

    #[arg(
        long,
        help = "Kubernetes namespace to monitor (empty for all namespaces)"
    )]
    k8s_namespace: Option<String>,

    #[arg(
        long,
        env = "GPU_NODE_NAME",
        help = "Node name for filtering pods to this node only"
    )]
    node_name: String,

    #[arg(
        long,
        env = "TENSOR_FUSION_POOL_NAME",
        help = "gpu pool is only used in metrics output"
    )]
    gpu_pool: Option<String>,

    #[arg(
        long,
        env = "KUBECONFIG",
        value_hint = clap::ValueHint::FilePath,
        help = "Path to kubeconfig file (defaults to cluster config or ~/.kube/config)"
    )]
    kubeconfig: Option<PathBuf>,

    #[arg(
        long,
        env = "API_LISTEN_ADDR",
        default_value = "0.0.0.0:8080",
        help = "HTTP API server listen address"
    )]
    api_listen_addr: String,

    #[arg(
        long,
        env = "TF_HYPERVISOR_METRICS_FORMAT",
        default_value = "influx",
        help = "Metrics format, either 'influx' or 'json' or 'otel'"
    )]
    metrics_format: String,

    #[arg(
        long,
        env = "TF_HYPERVISOR_METRICS_EXTRA_LABELS",
        help = "Extra labels to add to metrics"
    )]
    metrics_extra_labels: Option<String>,
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

    tracing::info!("Starting hypervisor {}", &**version::VERSION);

    let nvml = Arc::new(match Nvml::init() {
        Ok(nvml) => Ok(nvml),
        Err(_) => Nvml::builder()
            .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
            .init(),
    }?);

    let mut gpu_uuid_to_name_map = HashMap::new();
    let device_count = nvml.device_count()?;

    for i in 0..device_count {
        let device = nvml.device_by_index(i)?;
        let uuid = device.uuid()?.to_lowercase();
        let name = device.name()?;

        tracing::info!("Found GPU {}: {} ({})", i, uuid, name);
        // Store GPU name and UUID mapping for config lookup
        gpu_uuid_to_name_map.insert(uuid, name);
    }

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

    // HTTP trap handling is now integrated into the API server

    // Setup Kubernetes pod watcher if enabled
    let (k8s_update_sender, k8s_update_receiver) = mpsc::channel::<WorkerUpdate>();
    let (_k8s_shutdown_sender, k8s_shutdown_receiver) = oneshot::channel::<()>();

    let host_pid_probe = Arc::new(HostPidProbe::new(Duration::from_secs(1)));
    // Setup worker manager
    let worker_manager = Arc::new(WorkerManager::new(
        host_pid_probe.clone(),
        {
            let hypervisor = hypervisor.clone();
            move |pid, worker| {
                if hypervisor.process_exists(pid) {
                    return;
                }
                tracing::info!("new worker added: {}", worker.name());
                hypervisor.add_process(worker);
            }
        },
        {
            let hypervisor = hypervisor.clone();
            move |pid| {
                hypervisor.remove_process(pid);
            }
        },
    ));

    // Setup API server shutdown channel
    let (_api_shutdown_sender, api_shutdown_receiver) = oneshot::channel::<()>();

    // create command dispatcher
    let command_dispatcher = Arc::new(CommandDispatcher::new());

    // Start GPU observer task
    let gpu_observer_task = {
        let gpu_observer = gpu_observer.clone();
        tokio::spawn(async move {
            tracing::info!("Starting GPU observer task");
            gpu_observer.run(Duration::from_secs(1)).await;
        })
    };

    // Start metrics collection task
    let metrics_task = {
        let gpu_observer = gpu_observer.clone();
        let metrics_batch_size = cli.metrics_batch_size;
        let node_name = cli.node_name.clone();
        let gpu_pool = cli.gpu_pool.clone();
        let worker_manager = worker_manager.clone();
        tokio::spawn(async move {
            tracing::info!("Starting metrics collection task");
            metrics::run_metrics(
                gpu_observer,
                metrics_batch_size,
                node_name,
                gpu_pool,
                worker_manager,
                cli.metrics_format,
                cli.metrics_extra_labels,
            )
            .await;
        })
    };

    // Trap handling is now integrated into the HTTP API server

    // Start Kubernetes pod watcher if enabled
    let k8s_task = if cli.enable_k8s {
        let k8s_namespace = cli.k8s_namespace.clone();
        let node_name = cli.node_name.clone();
        let kubeconfig = cli.kubeconfig.clone();
        tokio::spawn(async move {
            tracing::info!("Starting Kubernetes pod watcher task");
            match PodWatcher::new(kubeconfig, k8s_namespace, node_name, k8s_update_sender).await {
                Ok(watcher) => {
                    if let Err(e) = watcher.run(k8s_shutdown_receiver).await {
                        tracing::error!("Kubernetes pod watcher failed: {e:?}");
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to create Kubernetes pod watcher: {e:?}");
                }
            }
        })
    } else {
        // Drop the receiver to avoid blocking
        drop(k8s_shutdown_receiver);
        tokio::spawn(async {
            // Empty task that never completes
            std::future::pending::<()>().await;
        })
    };

    // Start worker update processor for Kubernetes events
    let k8s_processor_task = if cli.enable_k8s {
        let worker_manager = worker_manager.clone();
        tokio::spawn(async move {
            tracing::info!("Starting Kubernetes update processor task");
            for update in k8s_update_receiver {
                match update {
                    WorkerUpdate::PodCreated { pod_info } => {
                        tracing::info!(
                            "Pod created: {}/{} with annotations: {:?}, node: {:?}",
                            pod_info.0.namespace,
                            pod_info.0.pod_name,
                            pod_info,
                            pod_info.0.node_name
                        );
                        if let Err(e) = worker_manager.handle_pod_created(pod_info).await {
                            tracing::error!("Failed to handle pod creation: {e}");
                        }
                    }
                    WorkerUpdate::PodUpdated {
                        pod_name,
                        namespace,
                        pod_info,
                        node_name,
                    } => {
                        tracing::info!(
                            "Pod updated: {}/{} with annotations: {:?}, node: {:?}",
                            namespace,
                            pod_name,
                            pod_info,
                            node_name
                        );
                        if let Err(e) = worker_manager
                            .handle_pod_updated(pod_name, namespace, pod_info, node_name)
                            .await
                        {
                            tracing::error!("Failed to handle pod update: {e}");
                        }
                    }
                    WorkerUpdate::PodDeleted {
                        pod_name,
                        namespace,
                    } => {
                        tracing::info!("Pod deleted: {}/{}", namespace, pod_name);
                        if let Err(e) = worker_manager.handle_pod_deleted(pod_name, namespace).await
                        {
                            tracing::error!("Failed to handle pod deletion: {e}");
                        }
                    }
                }
            }
        })
    } else {
        tokio::spawn(async {
            // Empty task that never completes
            std::future::pending::<()>().await;
        })
    };

    // Start API server task
    let api_task = {
        let hypervisor_for_api = hypervisor.clone();
        let command_dispatcher = command_dispatcher.clone();
        tokio::spawn(async move {
            tracing::info!("Starting HTTP API server task");
            // The API server needs to own the trap handler
            let jwt_config = JwtAuthConfig {
                public_key: std::env::var("JWT_PUBLIC_KEY").unwrap_or_else(|_| {
                    tracing::warn!("JWT_PUBLIC_KEY not set, using default placeholder");
                    "placeholder-public-key".to_string()
                }),
            };
            let api_server = ApiServer::new(
                worker_manager.clone(),
                cli.api_listen_addr.clone(),
                jwt_config,
                hypervisor_for_api,
                command_dispatcher.clone(),
                gpu_observer.clone(),
            );
            if let Err(e) = api_server.run(api_shutdown_receiver).await {
                tracing::error!("API server failed: {e}");
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
    tokio::select! {
        result = gpu_observer_task => {
            tracing::error!("GPU observer task completed: {:?}", result);
        }
        result = metrics_task => {
            tracing::error!("Metrics collection task completed: {:?}", result);
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

    // Gracefully shutdown background host PID probe task
    host_pid_probe.shutdown().await;

    Ok(())
}
