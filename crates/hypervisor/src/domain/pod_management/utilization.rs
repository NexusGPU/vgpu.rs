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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_snapshot_pod_utilization_calculation() {
        let mut process_utilizations = HashMap::new();
        process_utilizations.insert(
            1234,
            ProcessUtilization {
                sm_util: 30,
                codec_util: 10,
            },
        );
        process_utilizations.insert(
            5678,
            ProcessUtilization {
                sm_util: 20,
                codec_util: 5,
            },
        );
        process_utilizations.insert(
            9999,
            ProcessUtilization {
                sm_util: 15,
                codec_util: 0,
            },
        );

        let snapshot = DeviceSnapshot {
            process_utilizations,
            process_memories: HashMap::new(),
            timestamp: 1234567890,
        };

        // Test with multiple PIDs
        let pids = vec![1234, 5678];
        let pod_util = snapshot.get_pod_utilization(&pids);
        assert_eq!(pod_util.total_utilization, 65); // 30+10+20+5 = 65

        // Test with all PIDs
        let all_pids = vec![1234, 5678, 9999];
        let pod_util_all = snapshot.get_pod_utilization(&all_pids);
        assert_eq!(pod_util_all.total_utilization, 80); // 65 + 15 = 80

        // Test with single PID
        let single_pid = vec![9999];
        let single_util = snapshot.get_pod_utilization(&single_pid);
        assert_eq!(single_util.total_utilization, 15); // 15+0 = 15
    }

    #[test]
    fn test_device_snapshot_pod_memory_calculation() {
        let mut process_memories = HashMap::new();
        process_memories.insert(1234, 1024); // 1GB
        process_memories.insert(5678, 512); // 512MB
        process_memories.insert(9999, 256); // 256MB

        let snapshot = DeviceSnapshot {
            process_utilizations: HashMap::new(),
            process_memories,
            timestamp: 1234567890,
        };

        // Test with multiple PIDs
        let pids = vec![1234, 5678];
        let pod_memory = snapshot.get_pod_memory(&pids);
        assert_eq!(pod_memory, 1536); // 1024 + 512

        // Test with all PIDs
        let all_pids = vec![1234, 5678, 9999];
        let total_memory = snapshot.get_pod_memory(&all_pids);
        assert_eq!(total_memory, 1792); // 1024 + 512 + 256

        // Test with single PID
        let single_pid = vec![9999];
        let single_memory = snapshot.get_pod_memory(&single_pid);
        assert_eq!(single_memory, 256);
    }

    #[test]
    fn test_empty_pid_list_utilization() {
        let mut process_utilizations = HashMap::new();
        process_utilizations.insert(
            1234,
            ProcessUtilization {
                sm_util: 50,
                codec_util: 25,
            },
        );

        let snapshot = DeviceSnapshot {
            process_utilizations,
            process_memories: HashMap::new(),
            timestamp: 1234567890,
        };

        // Empty PID list should return 0 utilization
        let empty_pids = vec![];
        let pod_util = snapshot.get_pod_utilization(&empty_pids);
        assert_eq!(pod_util.total_utilization, 0);

        let pod_memory = snapshot.get_pod_memory(&empty_pids);
        assert_eq!(pod_memory, 0);
    }

    #[test]
    fn test_missing_process_data_handling() {
        let mut process_utilizations = HashMap::new();
        process_utilizations.insert(
            1234,
            ProcessUtilization {
                sm_util: 30,
                codec_util: 10,
            },
        );

        let mut process_memories = HashMap::new();
        process_memories.insert(1234, 1024);

        let snapshot = DeviceSnapshot {
            process_utilizations,
            process_memories,
            timestamp: 1234567890,
        };

        // Test with PIDs that don't exist in the snapshot
        let missing_pids = vec![9999, 8888];
        let pod_util = snapshot.get_pod_utilization(&missing_pids);
        assert_eq!(pod_util.total_utilization, 0);

        let pod_memory = snapshot.get_pod_memory(&missing_pids);
        assert_eq!(pod_memory, 0);

        // Test mixed scenario: some PIDs exist, some don't
        let mixed_pids = vec![1234, 9999];
        let mixed_util = snapshot.get_pod_utilization(&mixed_pids);
        assert_eq!(mixed_util.total_utilization, 40); // Only PID 1234 contributes

        let mixed_memory = snapshot.get_pod_memory(&mixed_pids);
        assert_eq!(mixed_memory, 1024); // Only PID 1234 contributes
    }

    #[test]
    fn test_utilization_overflow_handling() {
        // Test handling of utilization values that could cause overflow
        let mut process_utilizations = HashMap::new();

        // Add processes with maximum utilization values
        for i in 0..1000 {
            process_utilizations.insert(
                i,
                ProcessUtilization {
                    sm_util: u32::MAX / 2000, // Large but safe values
                    codec_util: u32::MAX / 2000,
                },
            );
        }

        let snapshot = DeviceSnapshot {
            process_utilizations,
            process_memories: HashMap::new(),
            timestamp: 1234567890,
        };

        let pids: Vec<u32> = (0..1000).collect();
        let pod_util = snapshot.get_pod_utilization(&pids);

        // Should not overflow and should be reasonable
        assert!(pod_util.total_utilization > 0);
        assert!(pod_util.total_utilization < u32::MAX);
    }

    #[test]
    fn test_memory_calculation_with_large_values() {
        // Test memory calculation with large values that could cause overflow
        let mut process_memories = HashMap::new();

        // Test with large memory values (but within u64 range)
        process_memories.insert(1, u64::MAX / 1000);
        process_memories.insert(2, u64::MAX / 1000);
        process_memories.insert(3, u64::MAX / 1000);

        let snapshot = DeviceSnapshot {
            process_utilizations: HashMap::new(),
            process_memories,
            timestamp: 1234567890,
        };

        let pids = vec![1, 2, 3];
        let total_memory = snapshot.get_pod_memory(&pids);

        // Should calculate correctly without overflow
        assert_eq!(total_memory, (u64::MAX / 1000) * 3);
    }
}
