use std::collections::HashMap;
use std::collections::HashSet;
use std::env;

use nvml_wrapper::device;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;

use crate::limiter::DeviceConfig;

pub fn parse_limits_and_create_device_configs(nvml: &Nvml) -> Vec<DeviceConfig> {
    let up_limit_json = env::var("TENSOR_FUSION_CUDA_UP_LIMIT").unwrap_or("{}".to_string());
    let mem_limit_json = env::var("TENSOR_FUSION_CUDA_MEM_LIMIT").unwrap_or("{}".to_string());
    let (device_configs, visible_devices) =
        parse_limits_json_and_create_device_configs(&up_limit_json, &mem_limit_json, nvml);
    // Set CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES environment variables
    if !visible_devices.is_empty() {
        env::set_var("CUDA_VISIBLE_DEVICES", &visible_devices);
        env::set_var("NVIDIA_VISIBLE_DEVICES", &visible_devices);
        tracing::info!(
            "Set CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES to {}",
            visible_devices
        );
    }
    device_configs
}

/// Parse GPU limits from environment variables and create device configurations
///
/// Returns:
/// - A vector of DeviceConfig for each device with limits
pub trait NvmlInterface {
    fn device_count(&self) -> Result<u32, NvmlError>;
    fn device_by_index(&self, index: u32) -> Result<Box<dyn DeviceInterface + '_>, NvmlError>;
}

impl NvmlInterface for Nvml {
    fn device_count(&self) -> Result<u32, NvmlError> {
        self.device_count()
    }

    fn device_by_index(&self, index: u32) -> Result<Box<dyn DeviceInterface + '_>, NvmlError> {
        match self.device_by_index(index) {
            Ok(device) => Ok(Box::new(DeviceWrapper(device))),
            Err(e) => Err(e),
        }
    }
}

pub trait DeviceInterface {
    fn uuid(&self) -> Result<String, NvmlError>;
}
struct DeviceWrapper<'nvml>(device::Device<'nvml>);

impl<'nvml> DeviceInterface for DeviceWrapper<'nvml> {
    fn uuid(&self) -> Result<String, NvmlError> {
        self.0.uuid()
    }
}

fn parse_limits_json_and_create_device_configs(
    up_limit_json: &str,
    mem_limit_json: &str,
    nvml: &dyn NvmlInterface,
) -> (Vec<DeviceConfig>, String) {
    // Parse JSON objects and convert keys to lowercase
    let up_limit_map = match serde_json::from_str::<HashMap<String, u32>>(up_limit_json) {
        Ok(map) => {
            // Convert all keys to lowercase
            let result = map
                .into_iter()
                .map(|(k, v)| (k.to_lowercase(), v))
                .collect();
            result
        }
        Err(e) => {
            tracing::error!("Failed to parse TENSOR_FUSION_CUDA_UP_LIMIT as JSON: {}", e);
            HashMap::new()
        }
    };

    let mem_limit_map = match serde_json::from_str::<HashMap<String, u64>>(mem_limit_json) {
        Ok(map) => {
            // Convert all keys to lowercase
            map.into_iter()
                .map(|(k, v)| (k.to_lowercase(), v))
                .collect()
        }
        Err(e) => {
            tracing::error!(
                "Failed to parse TENSOR_FUSION_CUDA_MEM_LIMIT as JSON: {}",
                e
            );
            HashMap::new()
        }
    };

    let mut device_configs = Vec::new();
    let mut visible_device_indices = Vec::new();

    // only process devices with user-defined limits
    if !up_limit_map.is_empty() || !mem_limit_map.is_empty() {
        // Map all device UUIDs to their indices
        let mut uuid_to_device_idx = HashMap::new();
        let mut idx_to_uuid = HashMap::new();

        let device_count = match nvml.device_count() {
            Ok(count) => count,
            Err(e) => {
                tracing::error!("Failed to get device count: {}", e);
                0
            }
        };

        // First, establish a mapping from UUID to device index
        for device_idx in 0..device_count {
            let device = match nvml.device_by_index(device_idx) {
                Ok(device) => device,
                Err(e) => {
                    tracing::error!("Failed to get device at index {}: {}", device_idx, e);
                    continue;
                }
            };

            let uuid = match device.uuid() {
                Ok(uuid) => uuid.to_lowercase(),
                Err(e) => {
                    tracing::error!("Failed to get UUID for device {}: {}", device_idx, e);
                    continue;
                }
            };

            uuid_to_device_idx.insert(uuid.clone(), device_idx);
            idx_to_uuid.insert(device_idx, uuid);
        }

        // collect all limit keys (could be UUIDs or indices)
        let mut limit_keys = HashSet::new();
        for key in up_limit_map.keys() {
            limit_keys.insert(key);
        }
        for key in mem_limit_map.keys() {
            limit_keys.insert(key);
        }

        // process each limit key
        for key in limit_keys {
            // try to parse the key as a device index
            if let Ok(idx) = key.parse::<u32>() {
                if idx < device_count {
                    // use the string form of key to get limit values from the maps
                    let up_limit = up_limit_map.get(key).copied().unwrap_or(0);
                    let mem_limit = mem_limit_map.get(key).copied().unwrap_or(0);

                    device_configs.push(DeviceConfig {
                        device_idx: idx,
                        up_limit,
                        mem_limit,
                    });

                    visible_device_indices.push(idx);
                }
            }
            // Check if it's a UUID (starts with gpu-)
            else if key.starts_with("gpu-") {
                if let Some(&device_idx) = uuid_to_device_idx.get(key) {
                    let up_limit = up_limit_map.get(key).copied().unwrap_or(0);
                    let mem_limit = mem_limit_map.get(key).copied().unwrap_or(0);

                    tracing::info!(
                        "Device {}: UUID {}, up_limit: {}, mem_limit: {}",
                        device_idx,
                        key,
                        up_limit,
                        mem_limit
                    );

                    device_configs.push(DeviceConfig {
                        device_idx,
                        up_limit,
                        mem_limit,
                    });

                    visible_device_indices.push(device_idx);
                } else {
                    tracing::warn!("UUID {} has limits set but no matching device found", key);
                }
            } else {
                tracing::warn!("Key {} is neither a valid device index nor a GPU UUID", key);
            }
        }
    }

    // If no devices were found, use a default configuration
    if device_configs.is_empty() {
        device_configs.push(DeviceConfig {
            device_idx: 0,
            up_limit: 0,
            mem_limit: 0,
        });
        visible_device_indices.push(0);
    }

    // Create visible device string
    let visible_devices = visible_device_indices
        .iter()
        .map(|idx| idx.to_string())
        .collect::<Vec<String>>()
        .join(",");

    (device_configs, visible_devices)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;

    use super::*;

    struct MockNvml {
        device_count: RefCell<u32>,
        device_uuids: RefCell<HashMap<u32, String>>,
    }

    impl MockNvml {
        fn new(device_count: u32, device_uuids: HashMap<u32, String>) -> Self {
            Self {
                device_count: RefCell::new(device_count),
                device_uuids: RefCell::new(device_uuids),
            }
        }
    }

    impl NvmlInterface for MockNvml {
        fn device_count(&self) -> Result<u32, NvmlError> {
            Ok(*self.device_count.borrow())
        }

        fn device_by_index(&self, index: u32) -> Result<Box<dyn DeviceInterface + '_>, NvmlError> {
            let uuids = self.device_uuids.borrow();
            if let Some(uuid) = uuids.get(&index) {
                Ok(Box::new(MockDevice { uuid: uuid.clone() }))
            } else {
                // If no UUID is set, use a default UUID
                Ok(Box::new(MockDevice {
                    uuid: format!("default-uuid-{}", index),
                }))
            }
        }
    }

    struct MockDevice {
        uuid: String,
    }

    impl DeviceInterface for MockDevice {
        fn uuid(&self) -> Result<String, NvmlError> {
            Ok(self.uuid.clone())
        }
    }

    #[test]
    fn test_parse_empty_env_vars() {
        // Create a mock NVML with 1 device
        let mock_nvml = MockNvml::new(1, HashMap::new());
        let up_limit_json = "{}";
        let mem_limit_json = "{}";

        let (configs, visible_devices) =
            parse_limits_json_and_create_device_configs(up_limit_json, mem_limit_json, &mock_nvml);
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].device_idx, 0);
        assert_eq!(configs[0].up_limit, 0);
        assert_eq!(configs[0].mem_limit, 0);
        assert_eq!(visible_devices, "0");
    }

    #[test]
    fn test_parse_device_index_limits() {
        // Set device index-based limits
        let up_limit_json = "{\"0\": 50, \"1\": 75}";
        let mem_limit_json = "{\"0\": 1024, \"1\": 2048}";

        // Create a mock NVML with 2 devices
        let mut device_uuids = HashMap::new();
        device_uuids.insert(0, "gpu-0-uuid".to_string());
        device_uuids.insert(1, "gpu-1-uuid".to_string());
        let mock_nvml = MockNvml::new(2, device_uuids);

        let (mut configs, visible_devices) =
            parse_limits_json_and_create_device_configs(up_limit_json, mem_limit_json, &mock_nvml);

        // Sort by device index to ensure stable test results
        configs.sort_by_key(|c| c.device_idx);

        assert!(!configs.is_empty(), "No device configs created");
        assert_eq!(configs[0].device_idx, 0);
        assert_eq!(configs[0].up_limit, 50);
        assert_eq!(configs[0].mem_limit, 1024);
        if configs.len() > 1 {
            assert_eq!(configs[1].device_idx, 1);
            assert_eq!(configs[1].up_limit, 75);
            assert_eq!(configs[1].mem_limit, 2048);
        }
        // visible_devices can be "0,1" or "1,0"
        assert!(visible_devices == "0,1" || visible_devices == "1,0");
    }

    #[test]
    fn test_parse_uuid_limits() {
        let up_limit_json = "{\"gpu-uuid-1\": 60, \"gpu-uuid-2\": 80}";
        let mem_limit_json = "{\"gpu-uuid-1\": 4096, \"gpu-uuid-2\": 8192}";

        let mut device_uuids = HashMap::new();
        device_uuids.insert(0, "gpu-uuid-1".to_string());
        device_uuids.insert(1, "gpu-uuid-2".to_string());
        let mock_nvml = MockNvml::new(2, device_uuids);

        let (configs, visible_devices) =
            parse_limits_json_and_create_device_configs(up_limit_json, mem_limit_json, &mock_nvml);
        assert_eq!(configs.len(), 2);
        let mut found_device0 = false;
        let mut found_device1 = false;
        for config in &configs {
            match config.device_idx {
                0 => {
                    found_device0 = true;
                    assert_eq!(config.up_limit, 60);
                    assert_eq!(config.mem_limit, 4096);
                }
                1 => {
                    found_device1 = true;
                    assert_eq!(config.up_limit, 80);
                    assert_eq!(config.mem_limit, 8192);
                }
                _ => panic!("Unexpected device index"),
            }
        }
        assert!(found_device0, "Device 0 config not found");
        assert!(found_device1, "Device 1 config not found");
        assert!(visible_devices == "0,1" || visible_devices == "1,0");
    }

    #[test]
    fn test_invalid_json() {
        let up_limit_json = "invalid json";
        let mem_limit_json = "also invalid";
        let mock_nvml = MockNvml::new(1, HashMap::new());
        let (configs, visible_devices) =
            parse_limits_json_and_create_device_configs(up_limit_json, mem_limit_json, &mock_nvml);
        assert_eq!(configs.len(), 1); // should use default config
        assert_eq!(configs[0].device_idx, 0);
        assert_eq!(configs[0].up_limit, 0); // due to invalid JSON, limits should be 0
        assert_eq!(configs[0].mem_limit, 0);
        assert_eq!(visible_devices, "0");
    }
}
