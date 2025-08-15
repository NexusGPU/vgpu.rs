use std::sync::Arc;

use utils::shared_memory::manager::ThreadSafeSharedMemoryManager;

use crate::domain::hypervisor::HypervisorType;
use crate::infrastructure::gpu_device_state_watcher::GpuDeviceStateWatcher;
use crate::infrastructure::gpu_observer::GpuObserver;
use crate::infrastructure::host_pid_probe::HostPidProbe;
use crate::infrastructure::limiter_comm::CommandDispatcher;
use crate::k8s::PodInfoCache;
use crate::pod_management::sampler::{NvmlDeviceSampler, SystemClock};
use crate::pod_management::{
    coordinator::LimiterCoordinator, manager::PodManager, pod_state_store::PodStateStore,
};

/// Application dependencies - simple struct with Arc-wrapped services
pub struct ApplicationServices {
    pub hypervisor: Arc<HypervisorType>,
    pub gpu_observer: Arc<GpuObserver>,
    pub pod_manager: Arc<
        PodManager<ThreadSafeSharedMemoryManager, PodStateStore, NvmlDeviceSampler, SystemClock>,
    >,
    pub host_pid_probe: Arc<HostPidProbe>,
    pub command_dispatcher: Arc<CommandDispatcher>,
    pub limiter_coordinator: Arc<
        LimiterCoordinator<
            ThreadSafeSharedMemoryManager,
            PodStateStore,
            NvmlDeviceSampler,
            SystemClock,
        >,
    >,
    pub gpu_device_state_watcher: Arc<GpuDeviceStateWatcher>,
    pub pod_info_cache: Arc<PodInfoCache>,
}
