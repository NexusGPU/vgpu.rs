use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

// use crate::api::server::ApiServer; // TODO: Enable when API server is integrated
use crate::config::Cli;
use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::HostPidProbe;
use crate::hypervisor::Hypervisor;
use crate::k8s::PodWatcher;
use crate::k8s::WorkerUpdate;
use crate::limiter_comm::CommandDispatcher;
use crate::metrics;
use crate::process::worker::TensorFusionWorker;
use crate::scheduler::weighted::WeightedScheduler;
use crate::worker_manager::WorkerManager;

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
    pub cli: Cli,
}

impl Application {
    /// Run application, start all tasks and wait for completion
    pub async fn run(&self) -> Result<()> {
        tracing::info!("Starting all application tasks...");

        // Create task manager
        let mut task_manager = Tasks::new();

        // Start all background tasks
        if let Err(e) = task_manager.spawn_all_tasks(self).await {
            tracing::error!("Failed to spawn application tasks: {}", e);
            return Err(e);
        }

        tracing::info!("All application tasks started successfully");

        // Wait for tasks to complete or receive shutdown signal
        if let Err(e) = task_manager.wait_for_completion().await {
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
    tasks: Vec<JoinHandle<()>>,
    k8s_shutdown_sender: Option<oneshot::Sender<()>>,
    api_shutdown_sender: Option<oneshot::Sender<()>>,
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
            k8s_shutdown_sender: None,
            api_shutdown_sender: None,
        }
    }

    /// Start all background tasks
    pub async fn spawn_all_tasks(&mut self, app: &crate::app::Application) -> Result<()> {
        let cli = &app.cli;

        // Start GPU observer task
        let gpu_observer_task = {
            let gpu_observer = app.gpu_observer.clone();
            tokio::spawn(async move {
                tracing::info!("Starting GPU observer task");
                gpu_observer.run(Duration::from_secs(1)).await;
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
                    )
                    .await;
                })
            };
            self.tasks.push(metrics_task);
        }

        // Set Kubernetes related tasks
        if cli.enable_k8s {
            let (k8s_update_sender, k8s_update_receiver) = mpsc::channel::<WorkerUpdate>(32);
            let (k8s_shutdown_sender, k8s_shutdown_receiver) = oneshot::channel::<()>();
            self.k8s_shutdown_sender = Some(k8s_shutdown_sender);

            // Start Kubernetes pod observer task
            let k8s_task = {
                let k8s_namespace = cli.k8s_namespace.clone();
                let node_name = cli.node_name.clone();
                let kubeconfig = cli.kubeconfig.clone();

                tokio::spawn(async move {
                    tracing::info!("Starting Kubernetes pod watcher task");
                    match PodWatcher::new(kubeconfig, k8s_namespace, node_name, k8s_update_sender)
                        .await
                    {
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
            };
            self.tasks.push(k8s_task);

            // Start Kubernetes update processor task
            let k8s_processor_task =
                self.spawn_k8s_processor_task(k8s_update_receiver, app.worker_manager.clone());
            self.tasks.push(k8s_processor_task);
        }

        // Start API server task
        // Note: API server Send trait issues have been resolved by changing
        // AddWorkerCallback and RemoveWorkerCallback to use String instead of &str
        tracing::info!("API server integration ready - Send trait issues resolved");

        // Placeholder task for now - API server can be enabled when CLI config is ready
        let api_placeholder_task = tokio::spawn(async {
            tracing::info!("API server placeholder task - ready for integration");
            // Wait briefly then exit
            tokio::time::sleep(Duration::from_secs(1)).await;
        });
        self.tasks.push(api_placeholder_task);

        // start hypervisor task
        let hypervisor_task = {
            let hypervisor = app.hypervisor.clone();
            tokio::spawn(async move {
                tracing::info!("Starting hypervisor task");
                hypervisor.run().await;
            })
        };
        self.tasks.push(hypervisor_task);

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

        // send shutdown signal
        if let Some(sender) = self.k8s_shutdown_sender.take() {
            let _ = sender.send(());
        }
        if let Some(sender) = self.api_shutdown_sender.take() {
            let _ = sender.send(());
        }
        Ok(())
    }

    /// create Kubernetes update processor task
    fn spawn_k8s_processor_task(
        &self,
        mut k8s_update_receiver: mpsc::Receiver<WorkerUpdate>,
        worker_manager: Arc<crate::app::WorkerManagerType>,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            tracing::info!("Starting Kubernetes update processor task");
            while let Some(update) = k8s_update_receiver.recv().await {
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
        })
    }
}
