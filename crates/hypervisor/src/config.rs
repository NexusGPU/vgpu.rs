use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

use clap::Parser;
use clap::Subcommand;
use once_cell::sync::Lazy;
use serde::Deserialize;
use serde::Serialize;
use utils::version;

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

#[derive(Parser)]
#[command(about, long_about, version = &**version::VERSION)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run hypervisor daemon
    Daemon(DaemonArgs),
    /// Mount shared memory
    #[command(name = "mount-shm")]
    MountShm(MountShmArgs),
}

#[derive(Parser)]
pub struct DaemonArgs {
    #[arg(
        long,
        env = "GPU_METRICS_FILE",
        value_hint = clap::ValueHint::FilePath,
        default_value = "/logs/metrics.log",
        help = "Path for printing GPU and worker metrics, e.g. /logs/metrics.log"
    )]
    pub gpu_metrics_file: Option<PathBuf>,

    #[arg(
        long,
        default_value = "10",
        help = "Number of metrics to aggregate before printing, default to 10 means aggregated every 10 seconds"
    )]
    pub metrics_batch_size: usize,

    #[arg(
        long,
        env = "TENSOR_FUSION_GPU_INFO_PATH",
        help = "Path for GPU info list, e.g. /etc/tensor-fusion/gpu-info.yaml"
    )]
    pub gpu_info_path: Option<PathBuf>,

    #[arg(
        long,
        help = "Enable Kubernetes pod monitoring",
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
    pub enable_k8s: bool,

    #[arg(
        long,
        help = "Kubernetes namespace to monitor (empty for all namespaces)"
    )]
    pub k8s_namespace: Option<String>,

    #[arg(
        long,
        env = "GPU_NODE_NAME",
        help = "Node name for filtering pods to this node only"
    )]
    pub node_name: String,

    #[arg(
        long,
        env = "TENSOR_FUSION_POOL_NAME",
        help = "gpu pool is only used in metrics output"
    )]
    pub gpu_pool: Option<String>,

    #[arg(
        long,
        env = "KUBECONFIG",
        value_hint = clap::ValueHint::FilePath,
        help = "Path to kubeconfig file (defaults to cluster config or ~/.kube/config)"
    )]
    pub kubeconfig: Option<PathBuf>,

    #[arg(
        long,
        env = "API_LISTEN_ADDR",
        default_value = "0.0.0.0:8080",
        help = "HTTP API server listen address"
    )]
    pub api_listen_addr: String,

    #[arg(
        long,
        help = "Enable metrics collection",
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
    pub enable_metrics: bool,

    #[arg(
        long,
        env = "TF_HYPERVISOR_METRICS_FORMAT",
        default_value = "influx",
        help = "Metrics format, either 'influx' or 'json' or 'otel'"
    )]
    pub metrics_format: String,

    #[arg(
        long,
        env = "TF_HYPERVISOR_METRICS_EXTRA_LABELS",
        help = "Extra labels to add to metrics"
    )]
    pub metrics_extra_labels: Option<String>,

    #[arg(
        long,
        help = "Enable device plugin",
        default_value_t = true,
        env = "ENABLE_DEVICE_PLUGIN",
        action = clap::ArgAction::Set
    )]
    pub enable_device_plugin: bool,

    #[arg(
        long,
        help = "kubelet socket path",
        env = "KUBELET_SOCKET_PATH",
        default_value = "/var/lib/kubelet/device-plugins/kubelet.sock"
    )]
    pub kubelet_socket_path: String,

    #[arg(
        long,
        help = "device plugin socket path",
        env = "DEVICE_PLUGIN_SOCKET_PATH",
        default_value = "/var/lib/kubelet/device-plugins/tensor-fusion.sock"
    )]
    pub device_plugin_socket_path: String,
}

#[derive(Parser)]
pub struct MountShmArgs {
    #[arg(
        long,
        help = "Shared memory mount point path",
        default_value = "/tensor-fusion/shm"
    )]
    pub mount_point: PathBuf,

    #[arg(long, help = "Shared memory size in MB", default_value = "64")]
    pub size_mb: u64,
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

    // Populate GPU_CAPACITY_MAP
    let mut capacity_map = GPU_CAPACITY_MAP
        .write()
        .expect("Failed to acquire write lock");
    let mut matched_count = 0;
    let total_gpus = gpu_uuid_to_name_map.len();

    for (gpu_uuid, gpu_name) in gpu_uuid_to_name_map {
        if let Some(gpu_info) = model_to_info.get(gpu_name) {
            capacity_map.insert(gpu_uuid.clone(), gpu_info.fp16_tflops);
            tracing::info!(
                "Mapped GPU {} ({}) to {} TFlops",
                gpu_uuid,
                gpu_name,
                gpu_info.fp16_tflops
            );
            matched_count += 1;
        } else {
            tracing::warn!(
                "GPU {} ({}) not found in configuration, using default 0.0 TFlops",
                gpu_uuid,
                gpu_name
            );
            capacity_map.insert(gpu_uuid.clone(), 0.0);
        }
    }

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
