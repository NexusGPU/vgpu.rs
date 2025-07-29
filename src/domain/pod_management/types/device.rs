//! Device allocation and usage types

use std::collections::HashMap;
use utils::shared_memory::DeviceConfig;

/// Device allocation information for a pod
#[derive(Debug, Clone)]
pub struct DeviceAllocation {
    /// Device configurations allocated to this pod
    pub configs: Vec<DeviceConfig>,
    /// Resource quotas per device
    pub quotas: HashMap<u32, DeviceQuota>, // device_idx -> quota
}

impl DeviceAllocation {
    pub fn new(configs: Vec<DeviceConfig>) -> Self {
        let mut quotas = HashMap::new();
        for config in &configs {
            quotas.insert(config.device_idx, DeviceQuota::from_config(config));
        }
        
        Self { configs, quotas }
    }

    /// Get device configuration by device index
    pub fn get_config(&self, device_idx: u32) -> Option<&DeviceConfig> {
        self.configs.iter().find(|c| c.device_idx == device_idx)
    }

    /// Get device quota by device index
    pub fn get_quota(&self, device_idx: u32) -> Option<&DeviceQuota> {
        self.quotas.get(&device_idx)
    }

    /// Get all device indices
    pub fn device_indices(&self) -> Vec<u32> {
        self.configs.iter().map(|c| c.device_idx).collect()
    }
}

/// Device usage tracking for a specific device
#[derive(Debug, Clone)]
pub struct DeviceUsage {
    pub device_idx: u32,
    pub device_uuid: String,
    /// Current CUDA core usage percentage (0-100)
    pub cuda_usage: u32,
    /// Current memory usage in bytes
    pub memory_usage: u64,
    /// Number of active processes on this device
    pub active_processes: u32,
}

impl DeviceUsage {
    pub fn new(device_idx: u32, device_uuid: String) -> Self {
        Self {
            device_idx,
            device_uuid,
            cuda_usage: 0,
            memory_usage: 0,
            active_processes: 0,
        }
    }

    /// Check if device usage exceeds quota
    pub fn exceeds_quota(&self, quota: &DeviceQuota) -> bool {
        self.cuda_usage > quota.cuda_limit || self.memory_usage > quota.memory_limit
    }
}

/// Resource quota for a device
#[derive(Debug, Clone)]
pub struct DeviceQuota {
    pub device_idx: u32,
    /// CUDA core usage limit (0-100)
    pub cuda_limit: u32,
    /// Memory usage limit in bytes
    pub memory_limit: u64,
    /// Total available CUDA cores
    pub total_cuda_cores: u32,
}

impl DeviceQuota {
    pub fn from_config(config: &DeviceConfig) -> Self {
        Self {
            device_idx: config.device_idx,
            cuda_limit: config.up_limit,
            memory_limit: config.mem_limit,
            total_cuda_cores: config.total_cuda_cores,
        }
    }

    /// Calculate allowed CUDA cores based on limit percentage
    pub fn allowed_cuda_cores(&self) -> u32 {
        (self.total_cuda_cores as f64 * (self.cuda_limit as f64 / 100.0)) as u32
    }
}