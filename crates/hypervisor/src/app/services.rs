use std::sync::Arc;

use crate::core::types::{HypervisorType, PodManagerType};
use crate::domain::pod_management::LimiterCoordinator;
use crate::infrastructure::gpu_device_state_watcher::GpuDeviceStateWatcher;
use crate::infrastructure::gpu_observer::GpuObserver;
use crate::infrastructure::host_pid_probe::HostPidProbe;
use crate::infrastructure::limiter_comm::CommandDispatcher;
use crate::k8s::PodWatcher;

/// Application dependencies - simple struct with Arc-wrapped services
pub struct ApplicationServices {
    pub hypervisor: Arc<HypervisorType>,
    pub gpu_observer: Arc<GpuObserver>,
    pub pod_manager: Arc<PodManagerType>,
    pub host_pid_probe: Arc<HostPidProbe>,
    pub command_dispatcher: Arc<CommandDispatcher>,
    pub limiter_coordinator: Arc<LimiterCoordinator>,
    pub gpu_device_state_watcher: Arc<GpuDeviceStateWatcher>,
    pub pod_watcher: Arc<PodWatcher>,
}
