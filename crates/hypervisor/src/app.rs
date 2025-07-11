use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::future::BoxFuture;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::config::Cli;
use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::HostPidProbe;
use crate::hypervisor::Hypervisor;
use crate::k8s::PodWatcher;
use crate::k8s::WorkerUpdate;
use crate::limiter_coordinator::LimiterCoordinator;
use crate::metrics;
use crate::process::worker::TensorFusionWorker;
use crate::scheduler::weighted::WeightedScheduler;
use crate::worker_manager::WorkerManager;

// 类型别名，简化长类型名
pub type HypervisorType = Hypervisor<TensorFusionWorker, WeightedScheduler<TensorFusionWorker>>;

// WorkerManager的回调函数类型
pub type AddWorkerCallback = Box<
    dyn Fn(
            &str, // container_name
            u32,  // container_pid
            u32,  // host_pid
            Arc<TensorFusionWorker>,
            Arc<LimiterCoordinator>,
            Arc<nvml_wrapper::Nvml>,
        ) -> BoxFuture<'static, ()>
        + Send
        + Sync,
>;
pub type RemoveWorkerCallback =
    Box<dyn Fn(&str, &str, u32, u32) -> BoxFuture<'static, ()> + Send + Sync>; // pod_name, container_name, container_pid, host_pid
pub type WorkerManagerType = WorkerManager<AddWorkerCallback, RemoveWorkerCallback>;

/// 应用程序核心结构，管理所有组件
pub struct Application {
    pub hypervisor: Arc<HypervisorType>,
    pub gpu_observer: Arc<GpuObserver>,
    pub worker_manager: Arc<WorkerManagerType>,
    pub host_pid_probe: Arc<HostPidProbe>,
    pub cli: Cli,
}

impl Application {
    /// 运行应用程序，启动所有任务并等待完成
    pub async fn run(&self) -> Result<()> {
        tracing::info!("Starting all application tasks...");

        // 创建任务管理器
        let mut task_manager = Tasks::new();

        // 启动所有后台任务
        if let Err(e) = task_manager.spawn_all_tasks(self).await {
            tracing::error!("Failed to spawn application tasks: {}", e);
            return Err(e);
        }

        tracing::info!("All application tasks started successfully");

        // 等待任务完成或接收到关闭信号
        if let Err(e) = task_manager.wait_for_completion().await {
            tracing::error!("Error during task execution: {}", e);
            return Err(e);
        }

        tracing::info!("Application run completed");
        Ok(())
    }

    /// 优雅关闭应用程序
    pub async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down application...");

        // 关闭host PID probe
        self.host_pid_probe.shutdown().await;

        tracing::info!("Application shutdown completed");
        Ok(())
    }
}

/// 任务管理器，负责启动和管理所有后台任务
pub struct Tasks {
    tasks: Vec<JoinHandle<()>>,
    k8s_shutdown_sender: Option<oneshot::Sender<()>>,
    api_shutdown_sender: Option<oneshot::Sender<()>>,
}

impl Tasks {
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            k8s_shutdown_sender: None,
            api_shutdown_sender: None,
        }
    }

    /// 启动所有后台任务
    pub async fn spawn_all_tasks(&mut self, app: &crate::app::Application) -> Result<()> {
        let cli = &app.cli;

        // 启动GPU观察者任务
        let gpu_observer_task = {
            let gpu_observer = app.gpu_observer.clone();
            tokio::spawn(async move {
                tracing::info!("Starting GPU observer task");
                gpu_observer.run(Duration::from_secs(1)).await;
            })
        };
        self.tasks.push(gpu_observer_task);

        // 启动指标收集任务
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

        // 设置Kubernetes相关任务
        if cli.enable_k8s {
            let (k8s_update_sender, k8s_update_receiver) = mpsc::channel::<WorkerUpdate>(32);
            let (k8s_shutdown_sender, k8s_shutdown_receiver) = oneshot::channel::<()>();
            self.k8s_shutdown_sender = Some(k8s_shutdown_sender);

            // 启动Kubernetes pod观察者任务
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

            // 启动Kubernetes更新处理器任务
            let k8s_processor_task =
                self.spawn_k8s_processor_task(k8s_update_receiver, app.worker_manager.clone());
            self.tasks.push(k8s_processor_task);
        }

        // 启动API服务器任务 (暂时注释掉以修复编译错误)
        // let (api_shutdown_sender, api_shutdown_receiver) = oneshot::channel::<()>();
        // self.api_shutdown_sender = Some(api_shutdown_sender);

        // TODO: 修复API server的trap_handler参数
        tracing::warn!("API server is temporarily disabled during refactoring");

        // 启动hypervisor任务
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

    /// 等待任务完成或接收到关闭信号
    pub async fn wait_for_completion(&mut self) -> Result<()> {
        // 从Vec中取出所有任务，避免借用问题
        let mut tasks = std::mem::take(&mut self.tasks);

        tokio::select! {
            result = async {
                // 等待任何一个任务完成
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

        // 发送关闭信号
        if let Some(sender) = self.k8s_shutdown_sender.take() {
            let _ = sender.send(());
        }
        if let Some(sender) = self.api_shutdown_sender.take() {
            let _ = sender.send(());
        }
        Ok(())
    }

    /// 创建Kubernetes更新处理器任务
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
