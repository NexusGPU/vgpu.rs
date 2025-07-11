use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::future::BoxFuture;

use crate::app::Application;
use crate::config::Cli;
use crate::gpu_init::initialize_gpu_system;
use crate::gpu_init::GpuSystem;
use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::HostPidProbe;
use crate::hypervisor::Hypervisor;
use crate::limiter_comm::CommandDispatcher;
use crate::limiter_coordinator::LimiterCoordinator;
use crate::process::GpuProcess;
use crate::scheduler::weighted::WeightedScheduler;
use crate::worker_manager::WorkerManager;
use crate::worker_registration::register_worker_to_limiter_coordinator;
use crate::worker_registration::unregister_worker_from_limiter_coordinator;

/// 应用程序构建器
pub struct ApplicationBuilder {
    cli: Cli,
}

impl ApplicationBuilder {
    /// 创建新的应用程序构建器
    pub fn new(cli: Cli) -> Self {
        Self { cli }
    }

    /// 构建完整的应用程序
    pub async fn build(self) -> Result<Application> {
        tracing::info!("Building application components...");

        // 初始化GPU系统
        let gpu_system = initialize_gpu_system(&self.cli).await?;

        // 创建核心组件
        let components = self.create_core_components(&gpu_system).await?;

        // 创建worker管理器（需要特殊处理回调函数）
        let worker_manager = self.create_worker_manager(&components, &gpu_system).await?;

        Ok(Application {
            hypervisor: components.hypervisor,
            gpu_observer: components.gpu_observer,
            worker_manager,
            host_pid_probe: components.host_pid_probe,
            cli: self.cli,
        })
    }

    /// 创建核心组件
    async fn create_core_components(&self, gpu_system: &GpuSystem) -> Result<CoreComponents> {
        // 创建调度器
        let scheduler = WeightedScheduler::new();

        // 创建hypervisor，设置1秒调度间隔
        let hypervisor = Arc::new(Hypervisor::new(scheduler, Duration::from_secs(1)));

        // 创建GPU观察者
        let gpu_observer = GpuObserver::create(gpu_system.nvml.clone());

        // 创建主机PID探测器
        let host_pid_probe = Arc::new(HostPidProbe::new(Duration::from_secs(1)));

        // 创建命令分发器
        let command_dispatcher = Arc::new(CommandDispatcher::new());

        // 创建限制器协调器
        let limiter_coordinator = Arc::new(LimiterCoordinator::new(
            Duration::from_millis(100),
            gpu_system.device_count,
        ));

        Ok(CoreComponents {
            hypervisor,
            gpu_observer,
            host_pid_probe,
            command_dispatcher,
            limiter_coordinator,
        })
    }

    /// 创建worker管理器，设置回调函数
    async fn create_worker_manager(
        &self,
        components: &CoreComponents,
        gpu_system: &GpuSystem,
    ) -> Result<Arc<crate::app::WorkerManagerType>> {
        // 创建worker添加回调
        let on_worker_added: crate::app::AddWorkerCallback = {
            let hypervisor = components.hypervisor.clone();
            let limiter_coordinator = components.limiter_coordinator.clone();
            let nvml = gpu_system.nvml.clone();

            Box::new(
                move |container_name,
                      container_pid,
                      host_pid,
                      worker,
                      limiter_coordinator,
                      nvml|
                      -> BoxFuture<'static, ()> {
                    let hypervisor = hypervisor.clone();
                    let worker = worker.clone();
                    let limiter_coordinator = limiter_coordinator.clone();
                    let nvml = nvml.clone();
                    let container_name = container_name.to_string();
                    Box::pin(async move {
                        if hypervisor.process_exists(host_pid) {
                            return;
                        }
                        tracing::info!("new worker added: {}", worker.name());
                        hypervisor.add_process(worker.as_ref().clone());

                        // 注册worker到限制器协调器
                        if let Err(e) = register_worker_to_limiter_coordinator(
                            &limiter_coordinator,
                            &worker,
                            &container_name,
                            container_pid,
                            host_pid,
                            &nvml,
                        )
                        .await
                        {
                            tracing::error!(
                                "Failed to register worker to limiter coordinator: {}",
                                e
                            );
                        } else {
                            tracing::info!("Successfully registered worker to limiter coordinator");
                        }

                        tracing::info!(
                            "Worker {} (container: {}, container_pid: {}, host_pid: {}) registered successfully",
                            worker.name(),
                            container_name,
                            container_pid,
                            host_pid
                        );
                    })
                },
            )
        };

        // 创建worker移除回调
        let on_worker_removed: crate::app::RemoveWorkerCallback = {
            let hypervisor = components.hypervisor.clone();
            let limiter_coordinator = components.limiter_coordinator.clone();

            Box::new(
                move |pod_name,
                      container_name,
                      container_pid,
                      host_pid|
                      -> BoxFuture<'static, ()> {
                    let hypervisor = hypervisor.clone();
                    let limiter_coordinator = limiter_coordinator.clone();
                    let pod_name = pod_name.to_string();
                    let container_name = container_name.to_string();
                    Box::pin(async move {
                        // 从 hypervisor 中移除进程
                        hypervisor.remove_process(host_pid);

                        // 从限制器协调器注销
                        if let Err(e) = unregister_worker_from_limiter_coordinator(
                            limiter_coordinator.as_ref(),
                            &pod_name,
                            &container_name,
                            container_pid,
                        )
                        .await
                        {
                            tracing::error!(
                                "Failed to unregister worker from limiter coordinator: {}",
                                e
                            );
                        }

                        tracing::info!(
                        "Worker (pod: {}, container: {}, container_pid: {}, host_pid: {}) removed successfully",
                        pod_name,
                        container_name,
                        container_pid,
                        host_pid
                    );
                    })
                },
            )
        };

        // 创建worker管理器
        let worker_manager = Arc::new(WorkerManager::new(
            components.host_pid_probe.clone(),
            on_worker_added,
            on_worker_removed,
            components.command_dispatcher.clone(),
        ));

        Ok(worker_manager)
    }
}

/// 核心组件集合
struct CoreComponents {
    hypervisor: Arc<crate::app::HypervisorType>,
    gpu_observer: Arc<GpuObserver>,
    host_pid_probe: Arc<HostPidProbe>,
    command_dispatcher: Arc<CommandDispatcher>,
    limiter_coordinator: Arc<LimiterCoordinator>,
}
