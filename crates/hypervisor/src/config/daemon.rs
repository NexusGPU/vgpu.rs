use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Clone)]
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
        help = "Kubelet device state path for fetching GPU allocation state of other device plugins",
        default_value = "/var/lib/kubelet/device-plugins/kubelet_internal_checkpoint",
        value_hint = clap::ValueHint::FilePath,
    )]
    pub kubelet_device_state_path: PathBuf,

    #[arg(
        long,
        help = "Kubelet socket path for fetching GPU allocation state of other device plugins",
        default_value = "/var/lib/kubelet/pod-resources/kubelet.sock",
        value_hint = clap::ValueHint::FilePath,
    )]
    pub kubelet_socket_path: PathBuf,

    #[arg(
        long,
        help = "Detect in-use GPUs for other device plugins",
        default_value_t = false,
        env = "DETECT_IN_USED_GPUS",
        action = clap::ArgAction::Set
    )]
    pub detect_in_used_gpus: bool,

    #[arg(
        long,
        help = "Base path for shared memory files",
        default_value = "/run/tensor-fusion/shm",
        env = "SHM_BASE_PATH",
        value_hint = clap::ValueHint::DirPath,
    )]
    pub shared_memory_base_path: PathBuf,

    #[arg(
        long,
        help = "Controller update interval in milliseconds",
        default_value = "100",
        env = "ERL_UPDATE_INTERVAL_MS"
    )]
    pub erl_update_interval_ms: u64,

    #[arg(
        long,
        help = "System-wide maximum refill rate safety limit (tokens/sec)",
        default_value = "1000000.0",
        env = "ERL_RATE_LIMIT"
    )]
    pub erl_rate_limit: f64,
}
