//! Utilization and device snapshot tracking module

use std::collections::HashMap;

/// Per-process utilization data
#[derive(Debug, Clone, Copy)]
pub struct ProcessUtilization {
    pub sm_util: u32,
    pub codec_util: u32,
}

/// Pod-level utilization summary
#[derive(Debug, Default, Clone, Copy)]
pub struct PodUtilization {
    pub total_utilization: u32, // Sum of SM + codec utilization for all processes in pod
}

/// Complete snapshot of device state including utilization and memory
#[derive(Debug, Clone)]
pub struct DeviceSnapshot {
    // pid -> utilization
    pub process_utilizations: HashMap<u32, ProcessUtilization>,
    // pid -> memory used
    pub process_memories: HashMap<u32, u64>,
    // timestamp of the snapshot
    pub timestamp: u64,
}

impl DeviceSnapshot {
    /// Calculate pod-level utilization from device snapshot
    pub fn get_pod_utilization(&self, pids: &[u32]) -> PodUtilization {
        let mut total_utilization = 0u32;

        for pid in pids {
            if let Some(process_util) = self.process_utilizations.get(pid) {
                total_utilization += process_util.sm_util + process_util.codec_util;
            }
        }

        PodUtilization { total_utilization }
    }

    /// Calculate pod-level memory usage from device snapshot
    pub fn get_pod_memory(&self, pids: &[u32]) -> u64 {
        pids.iter()
            .filter_map(|pid| self.process_memories.get(pid))
            .sum()
    }
}

/// Normalization function for codec utilization.
pub const fn codec_normalize(x: u32) -> u32 {
    x * 85 / 100
}
