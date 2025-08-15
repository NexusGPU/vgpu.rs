#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub uuid: String,
    pub available_cuda_cores: i32,
    pub total_cuda_cores: u32,
    pub mem_limit: u64,
    pub pod_memory_used: u64,
    pub up_limit: u32,
}

#[derive(Debug, Clone)]
pub struct WorkerDetailedInfo {
    pub identifier: String,
    pub devices: Vec<DeviceInfo>,
    pub is_healthy: bool,
    pub last_heartbeat: u64,
    pub active_pids: Vec<usize>,
    pub version: u32,
    pub device_count: usize,
}

#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub identifier: String,
    pub devices: Vec<DeviceInfo>,
    pub is_healthy: bool,
}

#[derive(Debug, Clone)]
pub enum AppState {
    Normal,
    DetailDialog(WorkerDetailedInfo),
}

#[derive(Debug, Clone)]
pub enum RefreshEvent {
    Tick,
    FileSystemChange,
}
