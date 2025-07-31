use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

/// global GPU capacity map, map GPU UUID to their fp16TFlops capacity
pub static GPU_CAPACITY_MAP: Lazy<RwLock<HashMap<String, f64>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// GPU information structure corresponding to YAML config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU model name (e.g., "T4", "A100_SXM")
    pub model: String,
    /// Full model name (e.g., "Tesla T4", "NVIDIA A100-SXM4-80GB")
    #[serde(rename = "fullModelName")]
    pub full_model_name: String,
    /// GPU vendor (e.g., "NVIDIA")
    pub vendor: String,
    /// Cost per hour in USD
    #[serde(rename = "costPerHour")]
    pub cost_per_hour: f64,
    /// FP16 TFLOPS capacity
    #[serde(rename = "fp16TFlops")]
    pub fp16_tflops: f64,
}

/// load GPU info from config file
pub async fn load_gpu_info(
    gpu_uuid_to_name_map: &HashMap<String, String>,
    gpu_info_path: PathBuf,
) -> anyhow::Result<()> {
    tracing::info!("Loading GPU configuration from {:?}", gpu_info_path);

    // Read YAML file
    let yaml_content = tokio::fs::read_to_string(&gpu_info_path).await?;

    // Parse YAML into GpuInfo vector
    let gpu_info_list: Vec<GpuInfo> = serde_yaml::from_str(&yaml_content)?;

    tracing::info!("Loaded {} GPU configurations", gpu_info_list.len());

    // Create a mapping from GPU model names to GpuInfo for efficient lookup
    let mut model_to_info: HashMap<String, &GpuInfo> = HashMap::new();

    for info in &gpu_info_list {
        // Store mappings for both model and full model name
        model_to_info.insert(info.model.clone(), info);
        model_to_info.insert(info.full_model_name.clone(), info);
    }

    // Create a mapping from GPU UUID to TFlops for spawn_blocking
    let mut gpu_tflops_map = HashMap::new();
    for (gpu_uuid, gpu_name) in gpu_uuid_to_name_map {
        let tflops = if let Some(gpu_info) = model_to_info.get(gpu_name) {
            tracing::info!(
                "Mapped GPU {} ({}) to {} TFlops",
                gpu_uuid,
                gpu_name,
                gpu_info.fp16_tflops
            );
            gpu_info.fp16_tflops
        } else {
            tracing::warn!(
                "GPU {} ({}) not found in configuration, using default 0.0 TFlops",
                gpu_uuid,
                gpu_name
            );
            0.0
        };
        gpu_tflops_map.insert(gpu_uuid.clone(), tflops);
    }

    // Populate GPU_CAPACITY_MAP using spawn_blocking
    let matched_count = gpu_tflops_map
        .iter()
        .filter(|(_, &tflops)| tflops > 0.0)
        .count();
    let total_gpus = gpu_tflops_map.len();

    tokio::task::spawn_blocking(move || {
        let mut capacity_map = GPU_CAPACITY_MAP
            .write()
            .expect("Failed to acquire write lock");

        for (gpu_uuid, tflops) in gpu_tflops_map {
            capacity_map.insert(gpu_uuid, tflops);
        }
    })
    .await
    .expect("Failed to execute blocking task");

    tracing::info!(
        "Successfully loaded GPU capacity map: {}/{} GPUs matched",
        matched_count,
        total_gpus
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    #[tokio::test]
    async fn test_load_gpu_info() {
        // create temporary YAML file
        let yaml_content = r#"
# Turing Architecture Series
- model: T4
  fullModelName: "Tesla T4"
  vendor: NVIDIA
  costPerHour: 0.53
  fp16TFlops: 65

# Ampere Architecture Series
- model: A100_SXM
  fullModelName: "NVIDIA A100-SXM4-80GB"
  vendor: NVIDIA
  costPerHour: 1.89
  fp16TFlops: 312
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(yaml_content.as_bytes()).unwrap();
        let temp_path = temp_file.path().to_path_buf();

        // create GPU UUID to name mapping
        let mut gpu_uuid_to_name_map = HashMap::new();
        gpu_uuid_to_name_map.insert(
            "GPU-12345678-1234-1234-1234-123456789012".to_string(),
            "Tesla T4".to_string(),
        );
        gpu_uuid_to_name_map.insert(
            "GPU-87654321-4321-4321-4321-210987654321".to_string(),
            "NVIDIA A100-SXM4-80GB".to_string(),
        );
        gpu_uuid_to_name_map.insert(
            "GPU-11111111-1111-1111-1111-111111111111".to_string(),
            "Unknown GPU".to_string(),
        );

        // test loading GPU config
        let result = load_gpu_info(&gpu_uuid_to_name_map, temp_path).await;
        assert!(result.is_ok());

        // verify GPU_CAPACITY_MAP content
        let capacity_map = GPU_CAPACITY_MAP
            .read()
            .expect("Failed to acquire read lock");

        // verify T4 GPU's TFlops value
        let t4_uuid = "GPU-12345678-1234-1234-1234-123456789012";
        assert_eq!(capacity_map.get(t4_uuid), Some(&65.0));

        // verify A100 GPU's TFlops value
        let a100_uuid = "GPU-87654321-4321-4321-4321-210987654321";
        assert_eq!(capacity_map.get(a100_uuid), Some(&312.0));

        // verify unknown GPU's TFlops value (should be set to 0.0)
        let unknown_uuid = "GPU-11111111-1111-1111-1111-111111111111";
        assert_eq!(capacity_map.get(unknown_uuid), Some(&0.0));
    }
}
