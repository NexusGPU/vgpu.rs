use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;

use crate::app::{Application, ApplicationServices};
use crate::config::DaemonArgs;
use crate::core::types::{HypervisorType, PodManagerType};
use crate::gpu_device_state_watcher::GpuDeviceStateWatcher;
use crate::gpu_init::initialize_gpu_system;
use crate::gpu_init::GpuSystem;
use crate::gpu_observer::GpuObserver;
use crate::host_pid_probe::HostPidProbe;
use crate::hypervisor::Hypervisor;
use crate::limiter_comm::CommandDispatcher;
use crate::pod_management::{LimiterCoordinator, PodManager};
use crate::scheduler::weighted::WeightedScheduler;

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

        // Create pod manager (special handling for callback functions)
        let pod_manager = self.create_pod_manager(&components, &gpu_system).await?;

        // Build application services with all components
        let services = ApplicationServices {
            hypervisor: components.hypervisor,
            gpu_observer: components.gpu_observer,
            pod_manager,
            host_pid_probe: components.host_pid_probe,
            command_dispatcher: components.command_dispatcher,
            limiter_coordinator: components.limiter_coordinator,
            gpu_device_state_watcher: components.gpu_device_state_watcher,
        };

        Ok(Application::new(services, self.daemon_args))
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
            self.daemon_args.shared_memory_glob_pattern.clone(),
        ));

        // create a GPU device state watcher
        let gpu_device_state_watcher = Arc::new(GpuDeviceStateWatcher::new(
            self.daemon_args.kubelet_device_state_path.clone(),
        ));

        Ok(CoreComponents {
            hypervisor,
            gpu_observer,
            host_pid_probe,
            command_dispatcher,
            limiter_coordinator,
            gpu_device_state_watcher,
        })
    }

    /// Create worker manager, set callback functions
    async fn create_pod_manager(
        &self,
        components: &CoreComponents,
        gpu_system: &GpuSystem,
    ) -> Result<Arc<PodManagerType>> {
        // Create worker manager with direct component dependencies
        let pod_manager = Arc::new(PodManager::new(
            components.host_pid_probe.clone(),
            components.command_dispatcher.clone(),
            components.hypervisor.clone(),
            components.limiter_coordinator.clone(),
            gpu_system.nvml.clone(),
        ));

        Ok(pod_manager)
    }
}

/// Core components collection
struct CoreComponents {
    hypervisor: Arc<HypervisorType>,
    gpu_observer: Arc<GpuObserver>,
    host_pid_probe: Arc<HostPidProbe>,
    command_dispatcher: Arc<CommandDispatcher>,
    limiter_coordinator: Arc<LimiterCoordinator>,
    gpu_device_state_watcher: Arc<GpuDeviceStateWatcher>,
}
