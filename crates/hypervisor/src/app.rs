use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::api::server::ApiServer;
use crate::config::DaemonArgs;
use crate::gpu_allocation_watcher::GpuDeviceStateWatcher;
use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::HostPidProbe;
use crate::hypervisor::Hypervisor;
use crate::k8s::PodWatcher;
use crate::k8s::WorkerUpdate;
use crate::limiter_comm::CommandDispatcher;
use crate::metrics;
use crate::pod_management::{LimiterCoordinator, PodManager};
use crate::process::worker::TensorFusionWorker;
use crate::scheduler::weighted::WeightedScheduler;

pub type HypervisorType = Hypervisor<TensorFusionWorker, WeightedScheduler<TensorFusionWorker>>;

// Simplified PodManager type
pub type PodManagerType = PodManager;

/// Application core structure, managing all components
pub struct Application {
    pub hypervisor: Arc<HypervisorType>,
    pub gpu_observer: Arc<GpuObserver>,
    pub pod_manager: Arc<PodManagerType>,
    pub host_pid_probe: Arc<HostPidProbe>,
    pub command_dispatcher: Arc<CommandDispatcher>,
    pub limiter_coordinator: Arc<LimiterCoordinator>,
    pub gpu_device_state_watcher: Arc<GpuDeviceStateWatcher>,
    pub daemon_args: DaemonArgs,
}

impl Application {
    /// Run application, start all tasks and wait for completion
    pub async fn run(&self) -> Result<()> {
        tracing::info!("Starting all application tasks...");

        // Create task manager
        let mut tasks = Tasks::new();

        // Start all background tasks
        if let Err(e) = tasks.spawn_all_tasks(self) {
            tracing::error!("Failed to spawn application tasks: {}", e);
            return Err(e);
        }

        // Wait for tasks to complete or receive shutdown signal
        if let Err(e) = tasks.wait_for_completion().await {
            tracing::error!("Error during task execution: {}", e);
            return Err(e);
        }

        tracing::info!("Application run completed");
        Ok(())
    }

    /// Gracefully shutdown application
    pub async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down application...");

        // Shutdown host PID probe
        self.host_pid_probe.shutdown().await;

        tracing::info!("Application shutdown completed");
        Ok(())
    }
}

/// Task manager, responsible for starting and managing all background tasks
pub struct Tasks {
    pub tasks: Vec<JoinHandle<()>>,
    cancellation_token: CancellationToken,
}

impl Default for Tasks {
    fn default() -> Self {
        Self::new()
    }
}

impl Tasks {
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            cancellation_token: CancellationToken::new(),
        }
    }

    /// Start all background tasks
    pub fn spawn_all_tasks(&mut self, app: &crate::app::Application) -> Result<()> {
        let cli = &app.daemon_args;

        // Start GPU observer task
        let gpu_observer_task = {
            let gpu_observer = app.gpu_observer.clone();
            let token = self.cancellation_token.clone();
            tokio::spawn(async move {
                tracing::info!("Starting GPU observer task");
                gpu_observer.run(Duration::from_secs(1), token).await;
                tracing::info!("GPU observer task completed");
            })
        };
        self.tasks.push(gpu_observer_task);

        // Start metrics collection task
        if cli.enable_metrics {
            let metrics_task = {
                let gpu_observer = app.gpu_observer.clone();
                let metrics_batch_size = cli.metrics_batch_size;
                let node_name = cli.node_name.clone();
                let gpu_pool = cli.gpu_pool.clone();
                let pod_manager = app.pod_manager.clone();
                let metrics_format = cli.metrics_format.clone();
                let metrics_extra_labels = cli.metrics_extra_labels.clone();
                let token = self.cancellation_token.clone();

                tokio::spawn(async move {
                    tracing::info!("Starting metrics collection task");
                    metrics::run_metrics(
                        gpu_observer,
                        metrics_batch_size,
                        &node_name,
                        gpu_pool.as_deref(),
                        pod_manager,
                        &metrics_format,
                        metrics_extra_labels.as_deref(),
                        token,
                    )
                    .await;
                    tracing::info!("Metrics collection task completed");
                })
            };
            self.tasks.push(metrics_task);
        }

        // Set Kubernetes related tasks
        if cli.enable_k8s {
            let (k8s_update_sender, k8s_update_receiver) = mpsc::channel::<WorkerUpdate>(32);

            // Start Kubernetes pod observer task
            let k8s_task = {
                let k8s_namespace = cli.k8s_namespace.clone();
                let node_name = cli.node_name.clone();
                let kubeconfig = cli.kubeconfig.clone();
                let token = self.cancellation_token.clone();

                tokio::spawn(async move {
                    tracing::info!("Starting Kubernetes pod watcher task");
                    match PodWatcher::new(kubeconfig, k8s_namespace, node_name, k8s_update_sender)
                        .await
                    {
                        Ok(watcher) => {
                            if let Err(e) = watcher.run(token).await {
                                tracing::error!("Kubernetes pod watcher failed: {e:?}");
                            } else {
                                tracing::info!("Kubernetes pod watcher completed");
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to create Kubernetes pod watcher: {e:?}");
                        }
                    }
                })
            };
            self.tasks.push(k8s_task);

            // Start Kubernetes update processor task
            let k8s_processor_task =
                self.spawn_k8s_processor_task(k8s_update_receiver, app.pod_manager.clone());
            self.tasks.push(k8s_processor_task);

            if cli.detect_in_used_gpus {
                // Start GPU device state watcher task
                let gpu_device_state_watcher_task = {
                    let gpu_device_state_watcher = app.gpu_device_state_watcher.clone();
                    let token = self.cancellation_token.clone();
                    let kubeconfig = app.daemon_args.kubeconfig.clone();

                    tokio::spawn(async move {
                        tracing::info!("Starting GPU device state watcher task");
                        if let Err(e) = gpu_device_state_watcher.run(token, kubeconfig).await {
                            tracing::error!("GPU device state watcher failed: {e:?}");
                        } else {
                            tracing::info!("GPU device state watcher completed");
                        }
                    })
                };
                self.tasks.push(gpu_device_state_watcher_task);
            }

            // Start API server task
            let api_server_task = {
                let pod_manager = app.pod_manager.clone();
                let listen_addr = cli.api_listen_addr.clone();
                let gpu_observer = app.gpu_observer.clone();
                let command_dispatcher = app.command_dispatcher.clone();
                let hypervisor = app.hypervisor.clone();
                let token = self.cancellation_token.clone();

                tokio::spawn(async move {
                    tracing::info!("Starting API server on {}", listen_addr);

                    // Create default JWT config - you may want to make this configurable
                    let jwt_config = crate::api::types::JwtAuthConfig {
                        public_key: "default-public-key".to_string(),
                    };

                    let api_server = ApiServer::new(
                        pod_manager,
                        listen_addr,
                        jwt_config,
                        hypervisor,
                        command_dispatcher,
                        gpu_observer,
                    );

                    if let Err(e) = api_server.run(token).await {
                        tracing::error!("API server failed: {}", e);
                    } else {
                        tracing::info!("API server completed");
                    }
                })
            };
            self.tasks.push(api_server_task);
        }

        // start hypervisor task
        let hypervisor_task = {
            let hypervisor = app.hypervisor.clone();
            let token = self.cancellation_token.clone();
            tokio::spawn(async move {
                tracing::info!("Starting hypervisor task");
                hypervisor.run(token).await;
                tracing::info!("Hypervisor task completed");
            })
        };
        self.tasks.push(hypervisor_task);

        // start limiter coordinator task
        let limiter_coordinator_task = {
            let limiter_coordinator = app.limiter_coordinator.clone();
            let token = self.cancellation_token.clone();

            tokio::spawn(async move {
                tracing::info!("Starting limiter coordinator task");
                limiter_coordinator.run(token).await;
                tracing::info!("Limiter coordinator task completed");
            })
        };
        self.tasks.push(limiter_coordinator_task);

        // start pod manager resource monitoring task
        let pod_manager_monitor_task = {
            let pod_manager = app.pod_manager.clone();
            let token = self.cancellation_token.clone();

            tokio::spawn(async move {
                tracing::info!("Starting worker manager resource monitoring task");
                // Start monitoring with 30 second interval and cancellation token
                let monitor_handle =
                    pod_manager.start_resource_monitor(Duration::from_secs(30), token);

                // Wait for the monitoring task to complete
                if let Err(e) = monitor_handle.await {
                    tracing::error!("Worker manager resource monitoring task failed: {}", e);
                } else {
                    tracing::info!("Worker manager resource monitoring task completed");
                }
            })
        };
        self.tasks.push(pod_manager_monitor_task);

        Ok(())
    }

    /// wait for tasks to complete or receive shutdown signal
    pub async fn wait_for_completion(&mut self) -> Result<()> {
        use std::time::Duration;
        use tokio::time::{timeout, Instant};

        // take all tasks from Vec to avoid borrow issues
        let tasks = std::mem::take(&mut self.tasks);
        let task_count = tasks.len();

        tracing::info!("Starting {} background tasks...", task_count);

        // Phase 1: Race between signal and natural completion
        let cancellation_token = self.cancellation_token.clone();
        let signal_future = async {
            if let Ok(()) = tokio::signal::ctrl_c().await {
                tracing::info!("Received Ctrl+C signal, initiating graceful shutdown...");
                cancellation_token.cancel();
            }
        };

        let tasks_future = futures::future::join_all(tasks);

        // Use futures::select to avoid ownership issues
        use futures::future::{select, Either};

        match select(Box::pin(signal_future), Box::pin(tasks_future)).await {
            // Signal received first - need graceful shutdown
            Either::Left((_, tasks_future)) => {
                tracing::info!(
                    "Shutdown signal received, waiting for tasks to complete gracefully..."
                );

                let shutdown_timeout = Duration::from_secs(30);
                let start_time = Instant::now();

                match timeout(shutdown_timeout, tasks_future).await {
                    Ok(results) => {
                        let elapsed = start_time.elapsed();
                        tracing::info!(
                            "All {} tasks completed gracefully in {:.2}s",
                            task_count,
                            elapsed.as_secs_f64()
                        );

                        // Check task results
                        for (i, task_result) in results.into_iter().enumerate() {
                            if let Err(e) = task_result {
                                tracing::error!("Task {} failed during shutdown: {}", i, e);
                            }
                        }
                    }
                    Err(_) => {
                        tracing::warn!(
                            "Graceful shutdown timeout reached after {}s. Some tasks may not have completed cleanly.", 
                            shutdown_timeout.as_secs()
                        );
                        // Note: Tasks will be dropped/aborted when this function returns
                    }
                }
            }

            // Tasks completed naturally (normal case)
            Either::Right((results, _)) => {
                tracing::info!("All {} tasks completed normally", task_count);

                // Check task results
                for (i, task_result) in results.into_iter().enumerate() {
                    if let Err(e) = task_result {
                        tracing::error!("Task {} failed: {}", i, e);
                    }
                }
            }
        }

        tracing::info!("Application shutdown completed");
        Ok(())
    }

    /// create Kubernetes update processor task
    fn spawn_k8s_processor_task(
        &self,
        mut k8s_update_receiver: mpsc::Receiver<WorkerUpdate>,
        pod_manager: Arc<crate::app::PodManagerType>,
    ) -> JoinHandle<()> {
        let token = self.cancellation_token.clone();
        tokio::spawn(async move {
            tracing::info!("Starting Kubernetes update processor task");
            loop {
                tokio::select! {
                    update = k8s_update_receiver.recv() => {
                        match update {
                            Some(update) => {
                                match update {
                                    WorkerUpdate::PodCreated { pod_info } => {
                                        tracing::info!(
                                            "Pod created: {}/{} with annotations: {:?}, node: {:?}",
                                            pod_info.0.namespace,
                                            pod_info.0.pod_name,
                                            pod_info,
                                            pod_info.0.node_name
                                        );
                                        if let Err(e) = pod_manager.handle_pod_created(pod_info).await {
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
                                        if let Err(e) = pod_manager
                                            .handle_pod_updated(&pod_name, &namespace, pod_info, node_name)
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
                                        if let Err(e) = pod_manager
                                            .handle_pod_deleted(&pod_name, &namespace)
                                            .await
                                        {
                                            tracing::error!("Failed to handle pod deletion: {e}");
                                        }
                                    }
                                }
                            }
                            None => {
                                tracing::info!("Kubernetes update receiver closed");
                                break;
                            }
                        }
                    }
                    _ = token.cancelled() => {
                        tracing::info!("Kubernetes update processor task cancelled");
                        break;
                    }
                }
            }
        })
    }
}
