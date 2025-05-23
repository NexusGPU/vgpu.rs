use anyhow::{anyhow, Context, Result};
use clap::Parser;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

const DEFAULT_TFLOPS: f64 = 10.0;

/// GPU information from the configuration file
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GpuInfo {
    pub model: String,
    #[serde(rename = "fullModelName")]
    pub full_model_name: String,
    pub vendor: String,
    #[serde(rename = "costPerHour")]
    pub cost_per_hour: f64,
    #[serde(rename = "fp16TFlops")]
    pub fp16_tflops: f64,
}

/// Command line arguments for GPU configuration
#[derive(Parser, Debug)]
struct GpuConfigArgs {
    /// Path to the GPU info YAML file
    #[clap(long, env = "TENSOR_FUSION_GPU_INFO_PATH")]
    gpu_info_path: Option<PathBuf>,
}

/// Global GPU capacity map that maps GPU UUIDs to their fp16TFlops capacity
static GPU_CAPACITY_MAP: Lazy<RwLock<HashMap<String, f64>>> = Lazy::new(|| RwLock::new(HashMap::new()));

/// Safe way to get read access to the GPU capacity map
fn with_capacity_map<F, T>(f: F) -> T
where
    F: FnOnce(RwLockReadGuard<HashMap<String, f64>>) -> T,
{
    let guard = GPU_CAPACITY_MAP.read().unwrap_or_else(|_| {
        tracing::error!("Failed to acquire read lock for GPU capacity map");
        panic!("Failed to acquire read lock for GPU capacity map");
    });
    f(guard)
}

/// Safe way to get write access to the GPU capacity map
fn with_capacity_map_mut<F, T>(f: F) -> Result<T>
where
    F: FnOnce(RwLockWriteGuard<HashMap<String, f64>>) -> T,
{
    let guard = GPU_CAPACITY_MAP.write()
        .map_err(|_| anyhow!("Failed to acquire write lock for GPU capacity map"))?;
    Ok(f(guard))
}

/// Load GPU information from a YAML file and store it in a map
pub fn load_gpu_info(gpu_name_to_uuid_map: HashMap<String, String>) -> Result<()> {
    let file_path = get_config_path()?;
    tracing::info!("Loading GPU information from {}", file_path.display());
    
    // Load GPU info list from YAML
    let file_content = std::fs::read_to_string(&file_path)
        .with_context(|| format!("Failed to read GPU info file {}", file_path.display()))?;
    
    let gpu_info_list: Vec<GpuInfo> = serde_yaml::from_str(&file_content)
        .with_context(|| format!("Failed to parse GPU info file {}", file_path.display()))?;
    
    tracing::info!("Loaded {} GPU configurations", gpu_info_list.len());
    
    // Create lookup maps
    let model_mappings: HashMap<String, (String, f64)> = gpu_info_list.iter()
        .map(|info| (info.model.clone(), (info.full_model_name.clone(), info.fp16_tflops)))
        .collect();
    
    // Update the global GPU capacity map
    with_capacity_map_mut(|mut capacity_map| {
        for (gpu_name, uuid) in gpu_name_to_uuid_map {
            map_gpu_to_capacity(&mut capacity_map, &gpu_name, &uuid, &model_mappings);
        }
    })?;
    
    Ok(())
}

/// Map a GPU to its capacity and insert into the capacity map
fn map_gpu_to_capacity(
    capacity_map: &mut HashMap<String, f64>,
    gpu_name: &str,
    uuid: &str,
    model_mappings: &HashMap<String, (String, f64)>,
) {
    // Try direct model match first
    if let Some((_, capacity)) = model_mappings.get(gpu_name) {
        capacity_map.insert(uuid.to_string(), *capacity);
        tracing::info!("Mapped GPU {} to model {} with capacity {} TFlops", uuid, gpu_name, capacity);
        return;
    }
    
    // Try fuzzy matching with full model names
    for (model, (full_name, capacity)) in model_mappings {
        if full_name.contains(gpu_name) || gpu_name.contains(model) {
            capacity_map.insert(uuid.to_string(), *capacity);
            tracing::info!("Mapped GPU {} to model {} with capacity {} TFlops", uuid, model, capacity);
            return;
        }
    }
    
    // No match found, use default
    tracing::warn!("Could not find matching GPU model for {} in config", gpu_name);
    capacity_map.insert(uuid.to_string(), DEFAULT_TFLOPS);
}

/// Get the device capacity for a GPU UUID
pub fn get_device_capacity(gpu_uuid: &str) -> f64 {
    with_capacity_map(|capacity_map| {
        *capacity_map.get(gpu_uuid).unwrap_or(&DEFAULT_TFLOPS)
    })
}

/// Get the configuration file path based on priority order
fn get_config_path() -> Result<PathBuf> {
    // Parse command line arguments using clap
    let args = GpuConfigArgs::parse();
    
    // Return the path if provided via arguments or environment
    if let Some(path) = args.gpu_info_path {
        return Ok(path);
    }
    
    // Default path
    Ok(PathBuf::from("./gpu-info.yaml"))
}
