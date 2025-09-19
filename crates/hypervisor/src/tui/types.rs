use utils::shared_memory::PodIdentifier;

#[derive(Debug, Clone)]
pub struct ShmEntry {
    pub pod_identifier: PodIdentifier,
    pub device_count: usize,
    pub last_heartbeat: u64,
    pub is_healthy: bool,
    pub version: u32,
    pub active_pids: Vec<usize>,
    pub devices: Vec<ShmDeviceInfo>,
}

#[derive(Debug, Clone)]
pub struct ShmDeviceInfo {
    pub device_index: usize,
    pub uuid: String,
    pub available_cores: i32,
    pub total_cores: u32,
    pub mem_limit: u64,
    pub pod_memory_used: u64,
    pub up_limit: u32,
    pub is_active: bool,
}

#[derive(Debug, Clone)]
pub enum AppState {
    Normal,
    DetailDialog(ShmEntry),
}

#[derive(Debug, Clone)]
pub enum RefreshEvent {
    Tick,
}
