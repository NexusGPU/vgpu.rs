use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

use clap::Parser;
use once_cell::sync::Lazy;
use utils::version;

/// 全局GPU容量映射，将GPU UUID映射到它们的fp16TFlops容量
pub static GPU_CAPACITY_MAP: Lazy<RwLock<HashMap<String, f64>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

#[derive(Parser)]
#[command(about, long_about, version = &**version::VERSION)]
pub struct Cli {
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
}

/// 加载GPU信息配置
pub async fn load_gpu_info(
    gpu_uuid_to_name_map: std::collections::HashMap<String, String>,
    gpu_info_path: PathBuf,
) -> anyhow::Result<()> {
    // GPU信息加载逻辑的占位符
    // 这里应该实现实际的配置文件加载逻辑
    tracing::info!("Loading GPU configuration from {:?}", gpu_info_path);
    Ok(())
}
