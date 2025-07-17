use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use nvml_wrapper::Nvml;

use crate::config::DaemonArgs;

/// GPU system information structure
pub struct GpuSystem {
    pub nvml: Arc<Nvml>,
    pub device_count: u32,
}

/// Initialize GPU system
pub async fn initialize_gpu_system(daemon_args: &DaemonArgs) -> Result<GpuSystem> {
    tracing::info!("Initializing GPU system...");

    // Initialize NVML
    let nvml = Arc::new(init_nvml()?);

    // Discover GPU devices
    let (device_count, gpu_uuid_to_name_map) = discover_gpu_devices(&nvml)?;

    // Load GPU information configuration
    load_gpu_config(&gpu_uuid_to_name_map, daemon_args).await?;

    Ok(GpuSystem { nvml, device_count })
}

fn init_nvml() -> Result<Nvml> {
    match Nvml::init() {
        Ok(nvml) => {
            tracing::info!("NVML initialized successfully");
            Ok(nvml)
        }
        Err(_) => {
            tracing::warn!("Standard NVML init failed, trying with explicit library path");
            let nvml = Nvml::builder()
                .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
                .init()?;
            tracing::info!("NVML initialized with explicit library path");
            Ok(nvml)
        }
    }
}

fn discover_gpu_devices(nvml: &Nvml) -> Result<(u32, HashMap<String, String>)> {
    let device_count = nvml.device_count()?;
    let mut gpu_uuid_to_name_map = HashMap::new();

    tracing::info!("Discovered {} GPU device(s)", device_count);

    for i in 0..device_count {
        let device = nvml.device_by_index(i)?;
        let uuid = device.uuid()?.to_lowercase();
        let name = device.name()?;

        tracing::info!("Found GPU {}: {} ({})", i, uuid, name);
        gpu_uuid_to_name_map.insert(uuid, name);
    }

    Ok((device_count, gpu_uuid_to_name_map))
}

async fn load_gpu_config(
    gpu_uuid_to_name_map: &HashMap<String, String>,
    daemon_args: &DaemonArgs,
) -> Result<()> {
    let gpu_info_path = daemon_args
        .gpu_info_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("./gpu-info.yaml"));

    if let Err(e) = crate::config::load_gpu_info(gpu_uuid_to_name_map, gpu_info_path).await {
        tracing::warn!("Failed to load GPU information: {}", e);
    } else {
        tracing::info!("GPU configuration loaded successfully");
    }

    Ok(())
}
