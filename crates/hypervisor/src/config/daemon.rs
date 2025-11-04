use clap::Parser;
use serde::{Deserialize, Deserializer};
use std::path::PathBuf;

/// Hypervisor scheduling configuration containing all scheduling-related parameters
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HypervisorScheduling {
    #[serde(default)]
    pub elastic_rate_limit_parameters: ElasticRateLimitParameters,
}

/// Elastic Rate Limit parameters for controlling GPU resource allocation
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ElasticRateLimitParameters {
    #[serde(deserialize_with = "deserialize_optional_f64_from_string")]
    pub max_refill_rate: f64,

    #[serde(deserialize_with = "deserialize_optional_f64_from_string")]
    pub min_refill_rate: f64,

    #[serde(deserialize_with = "deserialize_optional_f64_from_string")]
    pub filter_alpha: f64,

    #[serde(deserialize_with = "deserialize_optional_f64_from_string")]
    pub ki: f64,

    #[serde(deserialize_with = "deserialize_optional_f64_from_string")]
    pub kd: f64,

    #[serde(deserialize_with = "deserialize_optional_f64_from_string")]
    pub kp: f64,

    #[serde(deserialize_with = "deserialize_optional_f64_from_string")]
    pub burst_window: f64,

    #[serde(deserialize_with = "deserialize_optional_f64_from_string")]
    pub capacity_min: f64,

    #[serde(deserialize_with = "deserialize_optional_f64_from_string")]
    pub capacity_max: f64,

    #[serde(deserialize_with = "deserialize_optional_f64_from_string")]
    pub integral_decay_factor: f64,
}

impl Default for ElasticRateLimitParameters {
    fn default() -> Self {
        let defaults = erl::DeviceControllerConfig::default();
        Self {
            max_refill_rate: defaults.rate_max,
            min_refill_rate: defaults.rate_min,
            filter_alpha: defaults.filter_alpha,
            ki: defaults.ki,
            kd: defaults.kd,
            kp: defaults.kp,
            burst_window: defaults.burst_window,
            capacity_min: defaults.capacity_min,
            capacity_max: defaults.capacity_max,
            integral_decay_factor: defaults.integral_decay_factor,
        }
    }
}

/// Custom deserializer for f64 that accepts both string and number formats
/// Handles Go's JSON string representation of numeric values
fn deserialize_optional_f64_from_string<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrFloat {
        String(String),
        Float(f64),
    }

    match StringOrFloat::deserialize(deserializer)? {
        StringOrFloat::String(s) => s.parse::<f64>().map_err(|e| {
            serde::de::Error::custom(format!("Failed to parse float from string '{s}': {e}"))
        }),
        StringOrFloat::Float(f) => Ok(f),
    }
}

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
        help = "Hypervisor scheduling configuration as JSON string (contains elasticRateLimitParameters, etc.)",
        env = "TF_HYPERVISOR_SCHEDULING_CONFIG",
        value_parser = parse_scheduling_config
    )]
    pub scheduling_config: Option<HypervisorScheduling>,
}

/// Parse JSON string into HypervisorScheduling configuration
fn parse_scheduling_config(s: &str) -> Result<HypervisorScheduling, String> {
    serde_json::from_str(s).map_err(|e| format!("Failed to parse scheduling config JSON: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_elastic_rate_limit_params_from_go_json() {
        let json = r#"{
            "elasticRateLimitParameters": {
                "maxRefillRate": "100000.0",
                "minRefillRate": "10.0",
                "filterAlpha": "0.3",
                "ki": "0.1",
                "kd": "0.05",
                "kp": "0.5",
                "burstWindow": "2.0",
                "capacityMin": "100.0",
                "capacityMax": "200000.0",
                "integralDecayFactor": "0.9"
            }
        }"#;

        let config: HypervisorScheduling = serde_json::from_str(json)
            .expect("should deserialize HypervisorScheduling from Go JSON with string numbers");

        let params = &config.elastic_rate_limit_parameters;
        assert_eq!(
            params.max_refill_rate, 100_000.0,
            "max_refill_rate should match"
        );
        assert_eq!(params.min_refill_rate, 10.0, "min_refill_rate should match");
        assert_eq!(params.filter_alpha, 0.3, "filter_alpha should match");
        assert_eq!(params.ki, 0.1, "ki should match");
        assert_eq!(params.kd, 0.05, "kd should match");
        assert_eq!(params.kp, 0.5, "kp should match");
        assert_eq!(params.burst_window, 2.0, "burst_window should match");
        assert_eq!(params.capacity_min, 100.0, "capacity_min should match");
        assert_eq!(params.capacity_max, 200_000.0, "capacity_max should match");
        assert_eq!(
            params.integral_decay_factor, 0.9,
            "integral_decay_factor should match provided value"
        );
    }

    #[test]
    fn deserialize_elastic_rate_limit_params_from_numeric_json() {
        let json = r#"{
            "elasticRateLimitParameters": {
                "maxRefillRate": 100000.0,
                "minRefillRate": 10.0,
                "filterAlpha": 0.3,
                "ki": 0.1,
                "kd": 0.05,
                "kp": 0.5,
                "burstWindow": 2.0,
                "capacityMin": 100.0,
                "capacityMax": 200000.0
            }
        }"#;

        let config: HypervisorScheduling = serde_json::from_str(json)
            .expect("should deserialize HypervisorScheduling from numeric JSON");

        let params = &config.elastic_rate_limit_parameters;
        assert_eq!(
            params.max_refill_rate, 100_000.0,
            "max_refill_rate should match"
        );
        assert_eq!(params.min_refill_rate, 10.0, "min_refill_rate should match");
        assert_eq!(params.filter_alpha, 0.3, "filter_alpha should match");
    }

    #[test]
    fn deserialize_with_defaults_when_fields_missing() {
        let json = r#"{
            "elasticRateLimitParameters": {
                "maxRefillRate": "50000.0"
            }
        }"#;

        let config: HypervisorScheduling = serde_json::from_str(json)
            .expect("should deserialize with default values for missing fields");

        let params = &config.elastic_rate_limit_parameters;
        assert_eq!(
            params.max_refill_rate, 50_000.0,
            "max_refill_rate should use provided value"
        );
        assert_eq!(
            params.min_refill_rate, 10.0,
            "min_refill_rate should use default"
        );
        assert_eq!(params.filter_alpha, 0.3, "filter_alpha should use default");
        assert_eq!(params.ki, 0.1, "ki should use default");
        assert_eq!(
            params.integral_decay_factor, 0.95,
            "integral_decay_factor should use default"
        );
    }

    #[test]
    fn deserialize_empty_config_uses_all_defaults() {
        let json = r#"{}"#;

        let config: HypervisorScheduling =
            serde_json::from_str(json).expect("should deserialize empty config with all defaults");

        let params = &config.elastic_rate_limit_parameters;
        assert_eq!(
            params.max_refill_rate, 100_000.0,
            "should use default max_refill_rate"
        );
        assert_eq!(
            params.min_refill_rate, 10.0,
            "should use default min_refill_rate"
        );
        assert_eq!(params.filter_alpha, 0.3, "should use default filter_alpha");
        assert_eq!(params.ki, 0.1, "should use default ki");
        assert_eq!(params.kd, 0.05, "should use default kd");
        assert_eq!(params.kp, 0.5, "should use default kp");
        assert_eq!(params.burst_window, 2.0, "should use default burst_window");
        assert_eq!(
            params.capacity_min, 100.0,
            "should use default capacity_min"
        );
        assert_eq!(
            params.capacity_max, 200_000.0,
            "should use default capacity_max"
        );
        assert_eq!(
            params.integral_decay_factor, 0.95,
            "should use default integral_decay_factor"
        );
    }

    #[test]
    fn parse_scheduling_config_from_string() {
        let json_str = r#"{"elasticRateLimitParameters":{"maxRefillRate":"100000.0"}}"#;

        let config =
            parse_scheduling_config(json_str).expect("should parse scheduling config from string");

        assert_eq!(
            config.elastic_rate_limit_parameters.max_refill_rate,
            100_000.0
        );
    }
}
