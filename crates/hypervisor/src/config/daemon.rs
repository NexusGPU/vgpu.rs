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
        help = "PID proportional gain (Kp)",
        default_value = "1.5",
        env = "ERL_PID_KP"
    )]
    pub erl_pid_kp: f64,

    #[arg(
        long,
        help = "PID integral gain (Ki)",
        default_value = "0.4",
        env = "ERL_PID_KI"
    )]
    pub erl_pid_ki: f64,

    #[arg(
        long,
        help = "PID derivative gain (Kd)",
        default_value = "0.1",
        env = "ERL_PID_KD"
    )]
    pub erl_pid_kd: f64,

    #[arg(
        long,
        help = "Minimum token refill rate (tokens/sec)",
        default_value = "1.0",
        env = "ERL_MIN_REFILL_RATE"
    )]
    pub erl_min_refill_rate: f64,

    #[arg(
        long,
        help = "Maximum token refill rate (tokens/sec)",
        default_value = "5000.0",
        env = "ERL_MAX_REFILL_RATE"
    )]
    pub erl_max_refill_rate: f64,

    #[arg(
        long,
        help = "Initial token refill rate for new controllers (tokens/sec)",
        default_value = "100.0",
        env = "ERL_INITIAL_REFILL_RATE"
    )]
    pub erl_initial_refill_rate: f64,

    #[arg(
        long,
        help = "Token bucket burst allowance in seconds",
        default_value = "0.25",
        env = "ERL_BURST_SECONDS"
    )]
    pub erl_burst_seconds: f64,

    #[arg(
        long,
        help = "Minimum token bucket capacity",
        default_value = "10.0",
        env = "ERL_CAPACITY_FLOOR"
    )]
    pub erl_capacity_floor: f64,

    #[arg(
        long,
        help = "Derivative term smoothing factor (0.0 disables)",
        default_value = "0.2",
        env = "ERL_DERIVATIVE_FILTER"
    )]
    pub erl_derivative_filter: f64,

    #[arg(
        long,
        help = "Integral term clamp limit",
        default_value = "500.0",
        env = "ERL_INTEGRAL_LIMIT"
    )]
    pub erl_integral_limit: f64,

    #[arg(
        long,
        help = "Minimum delta time between PID updates in seconds",
        default_value = "0.05",
        env = "ERL_MIN_DELTA_TIME"
    )]
    pub erl_min_delta_time: f64,
}
