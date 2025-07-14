use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::api::server::ApiServer;
use crate::config::DaemonArgs;
use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::HostPidProbe;
use crate::hypervisor::Hypervisor;
use crate::k8s::device_plugin::GpuDevicePlugin;
use crate::k8s::PodWatcher;
use crate::k8s::WorkerUpdate;
use crate::limiter_comm::CommandDispatcher;
use crate::limiter_coordinator::LimiterCoordinator;
use crate::metrics;
use crate::process::worker::TensorFusionWorker;
use crate::scheduler::weighted::WeightedScheduler;
use crate::worker_manager::WorkerManager;
use crate::gpu_allocation_watcher::GpuDeviceStateWatcher;

pub type HypervisorType = Hypervisor<TensorFusionWorker, WeightedScheduler<TensorFusionWorker>>;

// Simplified WorkerManager type
pub type WorkerManagerType = WorkerManager;

/// Application core structure, managing all components
pub struct Application {
    pub hypervisor: Arc<HypervisorType>,
    pub gpu_observer: Arc<GpuObserver>,
    pub worker_manager: Arc<WorkerManagerType>,
    pub host_pid_probe: Arc<HostPidProbe>,
    pub command_dispatcher: Arc<CommandDispatcher>,
    pub device_plugin: Arc<GpuDevicePlugin>,
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
        if let Err(e) = tasks.spawn_all_tasks(self).await {
            tracing::error!("Failed to spawn application tasks: {}", e);
            return Err(e);
        }

        tracing::info!("All application tasks started successfully");

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
    pub async fn spawn_all_tasks(&mut self, app: &crate::app::Application) -> Result<()> {
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
                let worker_manager = app.worker_manager.clone();
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
                        worker_manager,
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
                self.spawn_k8s_processor_task(k8s_update_receiver, app.worker_manager.clone());
            self.tasks.push(k8s_processor_task);

            // Start GPU device state watcher task
            let gpu_device_state_watcher_task = {
                let gpu_device_state_watcher = app.gpu_device_state_watcher.clone();
                let token = self.cancellation_token.clone();
                let kubeconfig = app.cli.kubeconfig.clone();

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
            let worker_manager = app.worker_manager.clone();
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
                    worker_manager,
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

        // start device plugin task
        if cli.enable_device_plugin {
            let device_plugin_task = {
                let device_plugin = app.device_plugin.clone();
                let kubelet_socket_path = cli.kubelet_socket_path.clone();
                let device_plugin_socket_path = cli.device_plugin_socket_path.clone();
                let token = self.cancellation_token.clone();

                tokio::spawn(async move {
                    tracing::info!("Starting device plugin task");

                    if let Err(e) = device_plugin
                        .register_with_kubelet(&kubelet_socket_path)
                        .await
                    {
                        tracing::error!("Failed to register device plugin with kubelet: {}", e);
                        return;
                    }

                    if let Err(e) = device_plugin.start(&device_plugin_socket_path, token).await {
                        tracing::error!("Failed to start device plugin: {}", e);
                        return;
                    }

                    tracing::info!("Device plugin task completed");
                })
            };
            self.tasks.push(device_plugin_task);
        }

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

        Ok(())
    }

    /// wait for tasks to complete or receive shutdown signal
    pub async fn wait_for_completion(&mut self) -> Result<()> {
        // take all tasks from Vec to avoid borrow issues
        let mut tasks = std::mem::take(&mut self.tasks);

        tokio::select! {
            result = async {
                // wait for any task to complete
                while let Some(task) = tasks.pop() {
                    if let Ok(result) = task.await {
                        return Some(result);
                    }
                }
                None
            } => {
                if result.is_some() {
                    tracing::error!("A task completed unexpectedly");
                }
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Received Ctrl+C, shutting down...");
            }
        }

        // Cancel all tasks using the unified cancellation token
        tracing::info!("Cancelling all tasks...");
        self.cancellation_token.cancel();

        // Wait for all tasks to complete gracefully
        futures::future::join_all(tasks).await;

        Ok(())
    }

    /// create Kubernetes update processor task
    fn spawn_k8s_processor_task(
        &self,
        mut k8s_update_receiver: mpsc::Receiver<WorkerUpdate>,
        worker_manager: Arc<crate::app::WorkerManagerType>,
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
                                        if let Err(e) = worker_manager
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
