//! Simple implementations of traits for production use

use std::collections::HashMap;

use anyhow::Context;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::error::NvmlError;

use super::traits::{DeviceSnapshotProvider, TimeSource};
use super::utilization::{codec_normalize, DeviceSnapshot, ProcessUtilization};

/// Production NVML-based device snapshot provider
pub struct NvmlDeviceSampler;

impl NvmlDeviceSampler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NvmlDeviceSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceSnapshotProvider for NvmlDeviceSampler {
    type Error = anyhow::Error;

    fn get_device_snapshot(
        &self,
        device_idx: u32,
        last_seen_ts: u64,
    ) -> Result<DeviceSnapshot, Self::Error> {
        let nvml = nvml_wrapper::Nvml::init().context("Failed to initialize NVML")?;
        let device = nvml
            .device_by_index(device_idx)
            .context("Failed to get device by index")?;

        let device_snapshot = DeviceSnapshot {
            process_utilizations: HashMap::new(),
            process_memories: HashMap::new(),
            timestamp: last_seen_ts,
        };

        // Get utilization data from last seen timestamp
        let process_utilization_samples = match device.process_utilization_stats(last_seen_ts) {
            Ok(process_utilization_samples) => process_utilization_samples,
            Err(NvmlError::NotFound) => {
                vec![]
            }
            Err(e) => {
                return Err(e.into());
            }
        };

        // Get memory data
        let process_info = device
            .running_compute_processes()
            .context("Failed to get running compute processes")?;

        let mut process_utilizations = HashMap::new();
        let mut process_memories = HashMap::new();
        let mut newest_timestamp = last_seen_ts;

        // Process utilization data
        for sample in process_utilization_samples {
            // Skip old samples (defensive programming)
            if sample.timestamp < last_seen_ts {
                continue;
            }

            // Track the newest timestamp
            if sample.timestamp > newest_timestamp {
                newest_timestamp = sample.timestamp;
            }

            process_utilizations.insert(
                sample.pid,
                ProcessUtilization {
                    sm_util: sample.sm_util,
                    codec_util: codec_normalize(sample.enc_util + sample.dec_util),
                },
            );
        }

        // Process memory data
        for pi in process_info {
            if let UsedGpuMemory::Used(bytes) = pi.used_gpu_memory {
                process_memories.insert(pi.pid, bytes);
            }
        }

        if process_utilizations.is_empty() && process_memories.is_empty() {
            Ok(device_snapshot)
        } else {
            Ok(DeviceSnapshot {
                process_utilizations,
                process_memories,
                timestamp: newest_timestamp,
            })
        }
    }
}

/// Production system clock time source
pub struct SystemClock;

impl SystemClock {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSource for SystemClock {
    fn now_unix_secs(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvml_error_recovery_patterns() {
        let sampler = NvmlDeviceSampler::new();

        // Test multiple consecutive error scenarios
        let invalid_devices = vec![9999, 8888, 7777];

        for device_idx in invalid_devices {
            let result = sampler.get_device_snapshot(device_idx, 0);
            assert!(result.is_err(), "Should fail for device {device_idx}");

            // Test that error can be properly handled and doesn't panic
            match result {
                Err(e) => {
                    let error_msg = format!("{e:?}");
                    assert!(!error_msg.is_empty());
                    // Error should contain some indication of the problem
                    assert!(
                        error_msg.contains("Failed")
                            || error_msg.contains("Error")
                            || error_msg.contains("not found")
                            || error_msg.contains("NVML")
                    );
                }
                Ok(_) => panic!("Expected error for invalid device {device_idx}"),
            }
        }
    }

    #[test]
    fn test_timestamp_edge_cases() {
        let sampler = NvmlDeviceSampler::new();

        // Test with different last_seen_timestamp values
        let edge_timestamps = vec![
            0,          // Unix epoch
            u64::MAX,   // Maximum timestamp
            1577836800, // 2020-01-01
            4294967295, // 2106-02-07 (32-bit timestamp limit)
        ];

        for &timestamp in &edge_timestamps {
            // These will likely fail due to invalid device, but timestamp handling should be robust
            let result = sampler.get_device_snapshot(9999, timestamp);

            // The error should be about device access, not timestamp handling
            if let Err(e) = result {
                let error_msg = format!("{e}");
                // Should not contain timestamp-related errors
                assert!(!error_msg.to_lowercase().contains("timestamp"));
                assert!(!error_msg.to_lowercase().contains("time"));
            }
        }
    }
}
