use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use nvml_wrapper::Nvml;

use crate::config::Cli;

/// GPU系统信息结构
pub struct GpuSystem {
    pub nvml: Arc<Nvml>,
    pub device_count: u32,
    pub gpu_uuid_to_name_map: HashMap<String, String>,
}

/// 初始化GPU系统
pub async fn initialize_gpu_system(cli: &Cli) -> Result<GpuSystem> {
    tracing::info!("Initializing GPU system...");

    // 初始化NVML
    let nvml = Arc::new(init_nvml()?);

    // 发现GPU设备
    let (device_count, gpu_uuid_to_name_map) = discover_gpu_devices(&nvml)?;

    // 加载GPU信息配置
    load_gpu_config(&gpu_uuid_to_name_map, cli).await?;

    Ok(GpuSystem {
        nvml,
        device_count,
        gpu_uuid_to_name_map,
    })
}

/// 初始化NVML库
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

/// 发现GPU设备并构建UUID到名称的映射
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

/// 加载GPU配置信息
async fn load_gpu_config(gpu_uuid_to_name_map: &HashMap<String, String>, cli: &Cli) -> Result<()> {
    let gpu_info_path = cli
        .gpu_info_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("./gpu-info.yaml"));

    if let Err(e) = crate::config::load_gpu_info(gpu_uuid_to_name_map.clone(), gpu_info_path).await
    {
        tracing::warn!("Failed to load GPU information: {}", e);
        // 不将配置加载失败视为致命错误，只记录警告
    } else {
        tracing::info!("GPU configuration loaded successfully");
    }

    Ok(())
}
