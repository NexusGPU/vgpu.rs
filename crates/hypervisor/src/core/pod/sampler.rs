//! Simple implementations of traits for production use

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;

use super::traits::{DeviceSnapshotProvider, TimeSource};
use super::utilization::{codec_normalize, DeviceSnapshot, ProcessUtilization};

/// Production NVML-based device snapshot provider
pub struct NvmlDeviceSampler {
    nvml: Arc<Nvml>,
}

impl NvmlDeviceSampler {
    pub fn init() -> Result<Self, anyhow::Error> {
        Ok(Self {
            nvml: Arc::new(Nvml::init().context("Failed to initialize NVML")?),
        })
    }
}

impl DeviceSnapshotProvider for NvmlDeviceSampler {
    type Error = anyhow::Error;

    #[tracing::instrument(skip(self), fields(device_idx = device_idx, last_seen_ts = last_seen_ts))]
    fn get_device_snapshot(
        &self,
        device_idx: u32,
        last_seen_ts: u64,
    ) -> Result<DeviceSnapshot, Self::Error> {
        let device = self
            .nvml
            .device_by_index(device_idx)
            .context("Failed to get device by index")?;

        let device_snapshot = DeviceSnapshot {
            process_utilizations: HashMap::new(),
            process_memories: HashMap::new(),
            timestamp: last_seen_ts,
        };

        // Get utilization data from current time - 1 second
        // Note: NVML expects timestamps in microseconds (μs), not seconds
        // This matches the C implementation: (cur.tv_sec - 1) * 1000000 + cur.tv_usec
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let current_time_us = now.as_micros() as u64;
        let query_time_us = current_time_us.saturating_sub(1_000_000); // 1 second = 1,000,000 μs
        let process_utilization_samples = match device.process_utilization_stats(query_time_us) {
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
    pub const fn new() -> Self {
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
