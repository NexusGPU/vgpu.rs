use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;

use crate::app::Application;
use crate::config::Cli;
use crate::gpu_init::initialize_gpu_system;
use crate::gpu_init::GpuSystem;
use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::HostPidProbe;
use crate::hypervisor::Hypervisor;
use crate::limiter_comm::CommandDispatcher;
use crate::limiter_coordinator::LimiterCoordinator;
use crate::scheduler::weighted::WeightedScheduler;
use crate::worker_manager::WorkerManager;

/// Application builder
pub struct ApplicationBuilder {
    cli: Cli,
}

impl ApplicationBuilder {
    /// Create new application builder
    pub fn new(cli: Cli) -> Self {
        Self { cli }
    }

    /// Build complete application
    pub async fn build(self) -> Result<Application> {
        tracing::info!("Building application components...");

        // Initialize GPU system
        let gpu_system = initialize_gpu_system(&self.cli).await?;

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
            cli: self.cli,
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

        Ok(CoreComponents {
            hypervisor,
            gpu_observer,
            host_pid_probe,
            command_dispatcher,
            limiter_coordinator,
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
    hypervisor: Arc<crate::app::HypervisorType>,
    gpu_observer: Arc<GpuObserver>,
    host_pid_probe: Arc<HostPidProbe>,
    command_dispatcher: Arc<CommandDispatcher>,
    limiter_coordinator: Arc<LimiterCoordinator>,
}
