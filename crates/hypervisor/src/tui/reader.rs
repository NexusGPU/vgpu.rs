use anyhow::{Context, Result};
use glob::glob;
use std::path::Path;
use std::time::Duration;
use utils::shared_memory::handle::{SharedMemoryHandle, SHM_PATH_SUFFIX};
use utils::shared_memory::PodIdentifier;

use crate::tui::types::{ShmDeviceInfo, ShmEntry};

pub struct ShmReader;

impl ShmReader {
    pub fn read_all_shm_entries(pattern: &str) -> Result<Vec<ShmEntry>> {
        let mut entries = Vec::new();

        for entry in glob(pattern).context("Failed to parse glob pattern")? {
            let path = entry.context("Failed to read glob entry")?;

            match Self::read_shm_entry(&path) {
                Ok(shm_entry) => entries.push(shm_entry),
                Err(e) => {
                    tracing::warn!("Failed to read shared memory at {}: {}", path.display(), e);
                    continue;
                }
            }
        }

        entries.sort_by(|a, b| a.pod_identifier.name.cmp(&b.pod_identifier.name));
        Ok(entries)
    }

    fn read_shm_entry(path: &Path) -> Result<ShmEntry> {
        let handle = SharedMemoryHandle::open(path)?;
        let state = handle.get_state();

        let device_count = state.device_count();
        let last_heartbeat = state.get_last_heartbeat();
        let is_healthy = state.is_healthy(Duration::from_secs(2));
        let version = state.get_version();
        let active_pids = state.get_all_pids();

        let mut devices = Vec::new();

        if version >= 2 {
            for (i, device) in state.iter_active_devices() {
                devices.push(ShmDeviceInfo {
                    device_index: i,
                    uuid: device.get_uuid_owned(),
                    available_cores: 0,
                    total_cores: device.device_info.get_total_cores(),
                    mem_limit: device.device_info.get_mem_limit(),
                    pod_memory_used: device.device_info.get_pod_memory_used(),
                    up_limit: device.device_info.get_up_limit(),
                    is_active: device.is_active(),
                    erl_avg_cost: Some(device.device_info.get_erl_avg_cost()),
                    erl_token_capacity: Some(device.device_info.get_erl_token_capacity()),
                    erl_token_refill_rate: Some(device.device_info.get_erl_token_refill_rate()),
                    erl_current_tokens: Some(device.device_info.get_erl_current_tokens()),
                    erl_last_token_update: Some(device.device_info.get_erl_last_token_update()),
                });
            }
        } else {
            // V1 does not support iter_active_devices(). Iterate by index heuristically.
            for i in 0..device_count {
                if let Some((
                    uuid,
                    available_cores,
                    total_cores,
                    mem_limit,
                    pod_memory_used,
                    up_limit,
                    is_active,
                )) = state.get_device_info(i)
                {
                    if is_active {
                        devices.push(ShmDeviceInfo {
                            device_index: i,
                            uuid,
                            available_cores,
                            total_cores,
                            mem_limit,
                            pod_memory_used,
                            up_limit,
                            is_active,
                            erl_avg_cost: None,
                            erl_token_capacity: None,
                            erl_token_refill_rate: None,
                            erl_current_tokens: None,
                            erl_last_token_update: None,
                        });
                    }
                }
            }
        }

        let shm_file_str = path.join(SHM_PATH_SUFFIX).to_string_lossy().to_string();
        let pod_identifier = PodIdentifier::from_shm_file_path(&shm_file_str).ok_or(
            anyhow::anyhow!("Failed to parse PodIdentifier from path: {shm_file_str}"),
        )?;

        Ok(ShmEntry {
            pod_identifier,
            device_count,
            last_heartbeat,
            is_healthy,
            version,
            active_pids,
            devices,
        })
    }
}
