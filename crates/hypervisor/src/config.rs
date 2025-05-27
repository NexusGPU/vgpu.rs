use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

const DEFAULT_TFLOPS: f64 = 10.0;

/// GPU information from the configuration file
#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct GpuInfo {
    pub model: String,
    #[serde(rename = "fullModelName")]
    pub full_model_name: String,
    pub vendor: String,
    #[serde(rename = "costPerHour")]
    pub cost_per_hour: f64,
    #[serde(rename = "fp16TFlops")]
    pub fp16_tflops: f64,
}

/// Global GPU capacity map that maps GPU UUIDs to their fp16TFlops capacity
pub(crate) static GPU_CAPACITY_MAP: Lazy<RwLock<HashMap<String, f64>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Load GPU information from a YAML file and store it in a map
pub(crate) fn load_gpu_info(
    gpu_uuid_to_name_map: HashMap<String, String>,
    file_path: PathBuf,
) -> Result<()> {
    // Load GPU info list from YAML
    let file_content = std::fs::read_to_string(&file_path).unwrap_or_else(|err| {
        tracing::warn!(
            "Failed to read GPU info file {}: {}",
            file_path.display(),
            err
        );
        "[]".to_string()
    });

    let gpu_info_list: Vec<GpuInfo> = serde_yaml::from_str(&file_content)
        .with_context(|| format!("Failed to parse GPU info file {}", file_path.display()))?;

    tracing::info!("Loaded {} GPU configurations", gpu_info_list.len());

    // Create lookup maps
    let model_mappings: HashMap<String, (String, f64)> = gpu_info_list
        .into_iter()
        .map(|gpu| {
            let GpuInfo {
                model,
                full_model_name,
                fp16_tflops,
                ..
            } = gpu;
            (model, (full_model_name, fp16_tflops))
        })
        .collect();

    // Update the global GPU capacity map
    let mut capacity_map_guard = GPU_CAPACITY_MAP.write().expect("poisoned");
    for (uuid, gpu_name) in gpu_uuid_to_name_map {
        map_gpu_to_capacity(&mut capacity_map_guard, &gpu_name, &uuid, &model_mappings);
    }
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
        tracing::info!(
            "Mapped GPU {} to model {} with capacity {} TFlops",
            uuid,
            gpu_name,
            capacity
        );
        return;
    }
    // No match found, use default
    tracing::warn!(
        "Could not find matching GPU model for {} in config",
        gpu_name
    );
    capacity_map.insert(uuid.to_string(), DEFAULT_TFLOPS);
}
