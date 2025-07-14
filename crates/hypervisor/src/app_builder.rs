use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;

use crate::app;
use crate::app::Application;
use crate::config::DaemonArgs;
use crate::gpu_init::initialize_gpu_system;
use crate::gpu_init::GpuSystem;
use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::HostPidProbe;
use crate::hypervisor::Hypervisor;
use crate::k8s::device_plugin::GpuDevicePlugin;
use crate::limiter_comm::CommandDispatcher;
use crate::limiter_coordinator::LimiterCoordinator;
use crate::gpu_allocation_watcher::GpuDeviceStateWatcher;
use crate::scheduler::weighted::WeightedScheduler;
use crate::worker_manager::WorkerManager;

/// Application builder
pub struct ApplicationBuilder {
    daemon_args: DaemonArgs,
}

impl ApplicationBuilder {
    /// Create new application builder
    pub fn new(daemon_args: DaemonArgs) -> Self {
        Self { daemon_args }
    }

    /// Build complete application
    pub async fn build(self) -> Result<Application> {
        tracing::info!("Building application components...");

        // Initialize GPU system
        let gpu_system = initialize_gpu_system(&self.daemon_args).await?;

        // Create core components
        let components = self.create_core_components(&gpu_system).await?;

        // Create worker manager (special handling for callback functions)
        let worker_manager = self.create_worker_manager(&components, &gpu_system).await?;

        Ok(Application {
            hypervisor: components.hypervisor,
            gpu_observer: components.gpu_observer,
            worker_manager,
            host_pid_probe: components.host_pid_probe,
            command_dispatcher: components.command_dispatcher,
            daemon_args: self.daemon_args,
            device_plugin: components.device_plugin,
            limiter_coordinator: components.limiter_coordinator,
            gpu_device_state_watcher: components.gpu_device_state_watcher,
        })

    }

    /// Create core components
    async fn create_core_components(&self, gpu_system: &GpuSystem) -> Result<CoreComponents> {
        // Create scheduler
        let scheduler = WeightedScheduler::new();

        // Create hypervisor, set 1 second scheduling interval
        let hypervisor = Arc::new(Hypervisor::new(scheduler, Duration::from_secs(1)));

        // Create GPU observer
        let gpu_observer = GpuObserver::create(gpu_system.nvml.clone());

        // Create host PID probe
        let host_pid_probe = Arc::new(HostPidProbe::new(Duration::from_secs(1)));

        // Create command dispatcher
        let command_dispatcher = Arc::new(CommandDispatcher::new());

        // Create limiter coordinator
        let limiter_coordinator = Arc::new(LimiterCoordinator::new(
            Duration::from_millis(100),
            gpu_system.device_count,
        ));

        let device_plugin = GpuDevicePlugin::new(
            self.daemon_args.device_plugin_socket_path.clone(),
            "tensor-fusion.ai/shm".to_string(),
            "/dev/shm".to_string(),
            false,
            false,
        );
        // create a GPU device state watcher
        let gpu_device_state_watcher = Arc::new(GpuDeviceStateWatcher::new(
            self.cli.kubelet_device_state_path.clone(),
        ));

        Ok(CoreComponents {
            hypervisor,
            gpu_observer,
            host_pid_probe,
            command_dispatcher,
            limiter_coordinator,
            device_plugin,
            gpu_device_state_watcher,
        })
    }

    /// Create worker manager, set callback functions
    async fn create_worker_manager(
        &self,
        components: &CoreComponents,
        gpu_system: &GpuSystem,
    ) -> Result<Arc<crate::app::WorkerManagerType>> {
        // Create worker manager with direct component dependencies
        let worker_manager = Arc::new(WorkerManager::new(
            components.host_pid_probe.clone(),
            components.command_dispatcher.clone(),
            components.hypervisor.clone(),
            components.limiter_coordinator.clone(),
            gpu_system.nvml.clone(),
        ));

        Ok(worker_manager)
    }
}

/// Core components collection
struct CoreComponents {
    hypervisor: Arc<app::HypervisorType>,
    gpu_observer: Arc<GpuObserver>,
    host_pid_probe: Arc<HostPidProbe>,
    command_dispatcher: Arc<CommandDispatcher>,
    limiter_coordinator: Arc<LimiterCoordinator>,
    device_plugin: Arc<GpuDevicePlugin>,
    gpu_device_state_watcher: Arc<GpuDeviceStateWatcher>,
}
