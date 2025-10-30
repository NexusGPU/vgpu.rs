use std::sync::Arc;

use utils::shared_memory::manager::MemoryManager;

use crate::core::hypervisor::HypervisorType;
use crate::core::pod::sampler::{NvmlDeviceSampler, SystemClock};
use crate::core::pod::{
    coordinator::LimiterCoordinator, manager::PodManager, pod_state_store::PodStateStore,
};
use crate::platform::host_pid_probe::HostPidProbe;
use crate::platform::k8s::PodInfoCache;
use crate::platform::limiter_comm::CommandDispatcher;
use crate::platform::nvml::gpu_device_state_watcher::GpuDeviceStateWatcher;
use crate::platform::nvml::gpu_observer::GpuObserver;

/// Application dependencies - simple struct with Arc-wrapped services
pub struct ApplicationServices {
    pub hypervisor: Arc<HypervisorType>,
    pub gpu_observer: Arc<GpuObserver>,
    pub pod_manager: Arc<PodManager<MemoryManager, PodStateStore, NvmlDeviceSampler, SystemClock>>,
    pub host_pid_probe: Arc<HostPidProbe>,
    pub command_dispatcher: Arc<CommandDispatcher>,
    pub limiter_coordinator:
        Arc<LimiterCoordinator<MemoryManager, PodStateStore, NvmlDeviceSampler, SystemClock>>,
    pub gpu_device_state_watcher: Arc<GpuDeviceStateWatcher>,
    pub pod_info_cache: Arc<PodInfoCache>,
}
