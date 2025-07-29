use std::sync::Arc;

use crate::core::types::{HypervisorType, PodManagerType};
use crate::domain::pod_management::LimiterCoordinator;
use crate::infrastructure::gpu_observer::GpuObserver;
use crate::infrastructure::gpu_watcher::GpuDeviceStateWatcher;
use crate::infrastructure::host_pid_probe::HostPidProbe;
use crate::infrastructure::limiter_comm::CommandDispatcher;

/// Application dependencies - simple struct with Arc-wrapped services
pub struct ApplicationServices {
    pub hypervisor: Arc<HypervisorType>,
    pub gpu_observer: Arc<GpuObserver>,
    pub pod_manager: Arc<PodManagerType>,
    pub host_pid_probe: Arc<HostPidProbe>,
    pub command_dispatcher: Arc<CommandDispatcher>,
    pub limiter_coordinator: Arc<LimiterCoordinator>,
    pub gpu_device_state_watcher: Arc<GpuDeviceStateWatcher>,
}

impl ApplicationServices {
    /// Create new application services with all dependencies
    pub fn new(
        hypervisor: Arc<HypervisorType>,
        gpu_observer: Arc<GpuObserver>,
        pod_manager: Arc<PodManagerType>,
        host_pid_probe: Arc<HostPidProbe>,
        command_dispatcher: Arc<CommandDispatcher>,
        limiter_coordinator: Arc<LimiterCoordinator>,
        gpu_device_state_watcher: Arc<GpuDeviceStateWatcher>,
    ) -> Self {
        Self {
            hypervisor,
            gpu_observer,
            pod_manager,
            host_pid_probe,
            command_dispatcher,
            limiter_coordinator,
            gpu_device_state_watcher,
        }
    }
}
