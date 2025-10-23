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

    // ==================== ERL (Elastic Rate Limiter) Parameters ====================
    #[arg(
        long,
        help = "ERL base token refill rate per second (scaled by target utilization)",
        default_value = "500.0",
        env = "ERL_BASE_REFILL_RATE"
    )]
    pub erl_base_refill_rate: f64,

    #[arg(
        long,
        help = "ERL token bucket burst duration in seconds",
        default_value = "1.5",
        env = "ERL_BURST_DURATION"
    )]
    pub erl_burst_duration: f64,

    #[arg(
        long,
        help = "ERL minimum token bucket capacity",
        default_value = "50.0",
        env = "ERL_MIN_CAPACITY"
    )]
    pub erl_min_capacity: f64,

    #[arg(
        long,
        help = "ERL initial average cost for new controllers",
        default_value = "0.05",
        env = "ERL_INITIAL_AVG_COST"
    )]
    pub erl_initial_avg_cost: f64,

    #[arg(
        long,
        help = "ERL minimum average cost (lower bound)",
        default_value = "0.001",
        env = "ERL_MIN_AVG_COST"
    )]
    pub erl_min_avg_cost: f64,

    #[arg(
        long,
        help = "ERL maximum average cost (upper bound)",
        default_value = "10.0",
        env = "ERL_MAX_AVG_COST"
    )]
    pub erl_max_avg_cost: f64,

    #[arg(
        long,
        help = "ERL CUBIC congestion control C parameter",
        default_value = "0.4",
        env = "ERL_CUBIC_C"
    )]
    pub erl_cubic_c: f64,

    #[arg(
        long,
        help = "ERL CUBIC beta (multiplicative decrease factor for recovery)",
        default_value = "1.5",
        env = "ERL_CUBIC_BETA"
    )]
    pub erl_cubic_beta: f64,

    #[arg(
        long,
        help = "ERL CUBIC slow start factor",
        default_value = "1.2",
        env = "ERL_CUBIC_SLOW_START_FACTOR"
    )]
    pub erl_cubic_slow_start_factor: f64,

    #[arg(
        long,
        help = "ERL congestion avoidance alpha (smoothing factor, 0-1)",
        default_value = "0.5",
        env = "ERL_CONGESTION_ALPHA"
    )]
    pub erl_congestion_alpha: f64,

    #[arg(
        long,
        help = "ERL congestion avoidance adjustment threshold",
        default_value = "0.01",
        env = "ERL_ADJUSTMENT_THRESHOLD"
    )]
    pub erl_adjustment_threshold: f64,

    #[arg(
        long,
        help = "ERL congestion avoidance adjustment coefficient",
        default_value = "1.0",
        env = "ERL_ADJUSTMENT_COEFFICIENT"
    )]
    pub erl_adjustment_coefficient: f64,
}
