use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use utils::shared_memory::manager::MemoryManager;

use crate::config::DaemonArgs;
use crate::core::hypervisor::HypervisorType;
use crate::core::pod::sampler::{NvmlDeviceSampler, SystemClock};
use crate::core::pod::{
    coordinator::{CoordinatorConfig, LimiterCoordinator},
    manager::PodManager,
    pod_state_store::PodStateStore,
};
use crate::platform::host_pid_probe::HostPidProbe;
use crate::platform::k8s::PodInfoCache;
use crate::platform::limiter_comm::CommandDispatcher;
use crate::platform::nvml::gpu_device_state_watcher::GpuDeviceStateWatcher;
use crate::platform::nvml::gpu_init::initialize_gpu_system;
use crate::platform::nvml::gpu_init::GpuSystem;
use crate::platform::nvml::gpu_observer::GpuObserver;
use crate::scheduler::weighted::WeightedScheduler;
use crate::util::{Application, ApplicationServices};

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
            pod_info_cache: components.pod_info_cache,
        };

        Ok(Application::new(services, self.daemon_args))
    }

    /// Create core components
    async fn create_core_components(&self, gpu_system: &GpuSystem) -> Result<CoreComponents> {
        // Create scheduler
        let scheduler = WeightedScheduler::new();

        // Create hypervisor, set 1 second scheduling interval
        let hypervisor = Arc::new(HypervisorType::new(scheduler, Duration::from_secs(1)));

        // Create pod state store with shared memory base path
        let shm_base_path = self.daemon_args.shared_memory_base_path.clone();
        let pod_state_store = Arc::new(PodStateStore::new(shm_base_path.clone()));

        // Create GPU observer
        let gpu_observer = GpuObserver::create(gpu_system.nvml.clone(), pod_state_store.clone());

        // Create host PID probe
        let host_pid_probe = Arc::new(HostPidProbe::new(Duration::from_secs(1)));

        // Create command dispatcher
        let command_dispatcher = Arc::new(CommandDispatcher::new());

        // Create limiter coordinator
        let shared_memory_manager = Arc::new(MemoryManager::new());
        let snapshot = Arc::new(NvmlDeviceSampler::init()?);
        let time = Arc::new(SystemClock::new());

        let erl_config = crate::config::ErlConfig::from(&self.daemon_args);

        let config = CoordinatorConfig {
            watch_interval: Duration::from_millis(100),
            device_count: gpu_system.device_count,
            shared_memory_glob_pattern: format!(
                "{}/*/*",
                self.daemon_args.shared_memory_base_path.to_string_lossy(),
            ),
            base_path: shm_base_path,
            erl_config,
        };

        let limiter_coordinator = Arc::new(LimiterCoordinator::new(
            config,
            shared_memory_manager,
            pod_state_store.clone(),
            snapshot,
            time,
        ));

        // create a GPU device state watcher
        let gpu_device_state_watcher = Arc::new(GpuDeviceStateWatcher::new(
            &self.daemon_args.kubelet_device_state_path,
            &self.daemon_args.kubelet_socket_path,
        ));

        // create a pod info cache
        let pod_info_cache = Arc::new(
            PodInfoCache::init(
                self.daemon_args.kubeconfig.clone(),
                self.daemon_args.k8s_namespace.clone(),
                self.daemon_args.node_name.clone(),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to initialize pod info cache: {e:?}"))?,
        );

        Ok(CoreComponents {
            hypervisor,
            gpu_observer,
            host_pid_probe,
            command_dispatcher,
            limiter_coordinator,
            gpu_device_state_watcher,
            pod_info_cache,
            pod_state_store,
        })
    }

    /// Create worker manager, set callback functions
    async fn create_pod_manager(
        &self,
        components: &CoreComponents,
        gpu_system: &GpuSystem,
    ) -> Result<Arc<PodManager<MemoryManager, PodStateStore, NvmlDeviceSampler, SystemClock>>> {
        // Create worker manager with direct component dependencies
        let pod_manager = Arc::new(PodManager::new(
            components.host_pid_probe.clone(),
            components.command_dispatcher.clone(),
            components.hypervisor.clone(),
            components.limiter_coordinator.clone(),
            gpu_system.nvml.clone(),
            components.pod_info_cache.clone(),
            components.pod_state_store.clone(),
            components.gpu_observer.clone(),
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
    limiter_coordinator:
        Arc<LimiterCoordinator<MemoryManager, PodStateStore, NvmlDeviceSampler, SystemClock>>,
    gpu_device_state_watcher: Arc<GpuDeviceStateWatcher>,
    pod_info_cache: Arc<PodInfoCache>,
    pod_state_store: Arc<PodStateStore>,
}
