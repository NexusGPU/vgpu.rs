use std::time::Duration;

use anyhow::Result;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::api::server::ApiServer;
use crate::app::core::Application;
use crate::infrastructure::metrics;

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
    pub fn spawn_all_tasks(&mut self, app: &Application) -> Result<()> {
        let cli = app.daemon_args();

        // Start GPU observer task
        let gpu_observer_task = {
            let gpu_observer = app.services().gpu_observer.clone();
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
            let metrics_task = self.spawn_metrics_task(app);
            self.tasks.push(metrics_task);
        }

        // Set Kubernetes related tasks
        if cli.enable_k8s {
            // Start Kubernetes pod observer task
            let k8s_task = self.spawn_k8s_watcher_task(app);
            self.tasks.push(k8s_task);

            if cli.detect_in_used_gpus {
                // Start GPU device state watcher task
                let gpu_device_state_watcher_task = self.spawn_gpu_device_state_watcher_task(app);
                self.tasks.push(gpu_device_state_watcher_task);
            }
        }

        // Start API server task
        let api_server_task = self.spawn_api_server_task(app);
        self.tasks.push(api_server_task);

        // start hypervisor task
        let hypervisor_task = self.spawn_hypervisor_task(app);
        self.tasks.push(hypervisor_task);

        // start limiter coordinator task
        let limiter_coordinator_task = self.spawn_limiter_coordinator_task(app);
        self.tasks.push(limiter_coordinator_task);

        // start pod manager resource monitoring task
        let pod_manager_monitor_task = self.spawn_pod_manager_monitor_task(app);
        self.tasks.push(pod_manager_monitor_task);

        Ok(())
    }

    /// wait for tasks to complete or receive shutdown signal
    pub async fn wait_for_completion(&mut self) -> Result<()> {
        // Set up signal handling for graceful shutdown
        let signal_handler = {
            #[cfg(unix)]
            {
                use tokio::signal::unix::{signal, SignalKind};
                let mut sigterm = signal(SignalKind::terminate())?;
                let mut sigint = signal(SignalKind::interrupt())?;

                tokio::spawn(async move {
                    tokio::select! {
                        _ = sigterm.recv() => {
                            tracing::info!("Received SIGTERM, initiating graceful shutdown");
                        }
                        _ = sigint.recv() => {
                            tracing::info!("Received SIGINT, initiating graceful shutdown");
                        }
                    }
                })
            }
            #[cfg(not(unix))]
            {
                tokio::spawn(async {
                    tokio::signal::ctrl_c()
                        .await
                        .expect("Failed to install Ctrl+C handler");
                    tracing::info!("Received Ctrl+C, initiating graceful shutdown");
                })
            }
        };

        tokio::select! {
            // Wait for shutdown signal
            _ = signal_handler => {
                tracing::info!("Shutdown signal received, cancelling all tasks");
                self.cancellation_token.cancel();

                // Wait for all tasks with timeout
                self.wait_for_tasks_with_timeout(Duration::from_secs(30)).await;
            }
            // Wait for any task to complete unexpectedly
            result = futures::future::select_all(&mut self.tasks) => {
                let (result, _index, _remaining) = result;
                if let Err(e) = result {
                    tracing::error!("Task completed with error: {e}");
                    return Err(e.into());
                }
                tracing::warn!("Task completed unexpectedly");
            }
        }

        Ok(())
    }

    async fn wait_for_tasks_with_timeout(&mut self, timeout: Duration) {
        tokio::time::timeout(timeout, async {
            for task in &mut self.tasks {
                if let Err(e) = task.await {
                    tracing::error!("Task failed during shutdown: {e}");
                }
            }
        })
        .await
        .unwrap_or_else(|_| {
            tracing::warn!("Task shutdown timed out after {:?}", timeout);
        });
    }

    fn spawn_metrics_task(&self, app: &Application) -> JoinHandle<()> {
        let cli = app.daemon_args();
        let gpu_observer = app.services().gpu_observer.clone();
        let metrics_batch_size = cli.metrics_batch_size;
        let node_name = cli.node_name.clone();
        let gpu_pool = cli.gpu_pool.clone();
        let pod_manager = app.services().pod_manager.clone();
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
    }

    fn spawn_k8s_watcher_task(&self, app: &Application) -> JoinHandle<()> {
        let token = self.cancellation_token.clone();
        let pod_info_cache = app.services().pod_info_cache.clone();
        tokio::spawn(async move {
            tracing::info!("Starting Kubernetes pod watcher task");
            if let Err(e) = pod_info_cache.run(token).await {
                tracing::error!("Kubernetes pod watcher failed: {e:?}");
            } else {
                tracing::info!("Kubernetes pod watcher completed");
            }
        })
    }

    fn spawn_gpu_device_state_watcher_task(&self, app: &Application) -> JoinHandle<()> {
        let cli = app.daemon_args();
        let gpu_device_state_watcher = app.services().gpu_device_state_watcher.clone();
        let token = self.cancellation_token.clone();
        let kubeconfig = cli.kubeconfig.clone();

        tokio::spawn(async move {
            tracing::info!("Starting GPU device state watcher task");
            if let Err(e) = gpu_device_state_watcher.run(token, kubeconfig).await {
                tracing::error!("GPU device state watcher failed: {e:?}");
            } else {
                tracing::info!("GPU device state watcher completed");
            }
        })
    }

    fn spawn_api_server_task(&self, app: &Application) -> JoinHandle<()> {
        let cli = app.daemon_args();
        let pod_manager = app.services().pod_manager.clone();
        let listen_addr = cli.api_listen_addr.clone();
        let command_dispatcher = app.services().command_dispatcher.clone();
        let hypervisor = app.services().hypervisor.clone();
        let token = self.cancellation_token.clone();

        tokio::spawn(async move {
            tracing::info!("Starting API server on {}", listen_addr);

            // Create default JWT config - you may want to make this configurable
            let jwt_config = crate::api::JwtAuthConfig {
                public_key: "default-public-key".to_string(),
            };

            let api_server = ApiServer::new(
                pod_manager,
                listen_addr,
                jwt_config,
                hypervisor,
                command_dispatcher,
            );

            if let Err(e) = api_server.run(token).await {
                tracing::error!("API server failed: {}", e);
            } else {
                tracing::info!("API server completed");
            }
        })
    }

    fn spawn_hypervisor_task(&self, app: &Application) -> JoinHandle<()> {
        let hypervisor = app.services().hypervisor.clone();
        let token = self.cancellation_token.clone();
        tokio::spawn(async move {
            tracing::info!("Starting hypervisor task");
            hypervisor.run(token).await;
            tracing::info!("Hypervisor task completed");
        })
    }

    fn spawn_limiter_coordinator_task(&self, app: &Application) -> JoinHandle<()> {
        let limiter_coordinator = app.services().limiter_coordinator.clone();
        let token = self.cancellation_token.clone();

        tokio::spawn(async move {
            tracing::info!("Starting limiter coordinator task");
            limiter_coordinator.run(token).await;
            tracing::info!("Limiter coordinator task completed");
        })
    }

    fn spawn_pod_manager_monitor_task(&self, app: &Application) -> JoinHandle<()> {
        let pod_manager = app.services().pod_manager.clone();
        let token = self.cancellation_token.clone();

        tokio::spawn(async move {
            tracing::info!("Restoring pods from shared memory");
            if let Err(e) = pod_manager.restore_pod_from_shared_memory("tf_shm_*").await {
                panic!("Failed to restore pods from shared memory: {e}");
            }

            tracing::info!("Starting worker manager resource monitoring task");
            // Start monitoring with 30 second interval and cancellation token
            pod_manager
                .start_resource_monitor(Duration::from_secs(30), token)
                .await;
        })
    }
}
