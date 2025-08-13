use std::collections::HashMap;

pub mod influx;
pub mod json;

/// Parameters for encoding GPU metrics
#[derive(Debug, Clone)]
pub struct GpuMetricsParams<'a> {
    pub gpu_uuid: &'a str,
    pub node_name: &'a str,
    pub gpu_pool: &'a str,
    pub rx: f64,
    pub tx: f64,
    pub temperature: f64,
    pub graphics_clock_mhz: f64,
    pub sm_clock_mhz: f64,
    pub memory_clock_mhz: f64,
    pub video_clock_mhz: f64,
    pub memory_bytes: u64,
    pub memory_percentage: f64,
    pub compute_percentage: f64,
    pub compute_tflops: f64,
    pub power_usage: i64,
    pub nvlink_rx_bandwidth: i64,
    pub nvlink_tx_bandwidth: i64,
    pub timestamp: i64,
}

/// Parameters for encoding worker metrics
#[derive(Debug, Clone)]
pub struct WorkerMetricsParams<'a> {
    pub gpu_uuid: &'a str,
    pub node_name: &'a str,
    pub gpu_pool: &'a str,
    pub pod_identifier: &'a str,
    pub namespace: &'a str,
    pub workload: &'a str,
    pub memory_bytes: u64,
    pub compute_percentage: f64,
    pub compute_tflops: f64,
    pub memory_percentage: f64,
    pub timestamp: i64,
    pub extra_labels: &'a HashMap<String, String>,
}

/// Field value for metrics
#[derive(Debug, Clone)]
pub enum FieldValue {
    String(String),
    Integer(i64),
    UnsignedInteger(u64),
    Float(f64),
    Boolean(bool),
}

impl From<String> for FieldValue {
    fn from(value: String) -> Self {
        FieldValue::String(value)
    }
}

impl From<&str> for FieldValue {
    fn from(value: &str) -> Self {
        FieldValue::String(value.to_string())
    }
}

impl From<i64> for FieldValue {
    fn from(value: i64) -> Self {
        FieldValue::Integer(value)
    }
}

impl From<u64> for FieldValue {
    fn from(value: u64) -> Self {
        FieldValue::UnsignedInteger(value)
    }
}

impl From<f64> for FieldValue {
    fn from(value: f64) -> Self {
        FieldValue::Float(value)
    }
}

impl From<bool> for FieldValue {
    fn from(value: bool) -> Self {
        FieldValue::Boolean(value)
    }
}

/// Trait for encoding metrics in different formats
pub trait MetricsEncoder: Send + Sync {
    /// Encode metrics with measurement name, tags, fields, and timestamp
    fn encode_metrics(
        &self,
        measurement: &str,
        tags: &HashMap<String, String>,
        fields: &HashMap<String, FieldValue>,
        timestamp: i64,
    ) -> String;

    /// Encode GPU metrics using parameters struct
    fn encode_gpu_metrics_with_params(&self, params: &GpuMetricsParams) -> String {
        let mut tags = HashMap::new();
        tags.insert("node".to_string(), params.node_name.to_string());
        tags.insert("pool".to_string(), params.gpu_pool.to_string());
        tags.insert("uuid".to_string(), params.gpu_uuid.to_string());

        let mut fields = HashMap::new();
        fields.insert("rx".to_string(), params.rx.into());
        fields.insert("tx".to_string(), params.tx.into());
        fields.insert("temperature".to_string(), params.temperature.into());
        fields.insert(
            "graphics_clock_mhz".to_string(),
            params.graphics_clock_mhz.into(),
        );
        fields.insert("sm_clock_mhz".to_string(), params.sm_clock_mhz.into());
        fields.insert(
            "memory_clock_mhz".to_string(),
            params.memory_clock_mhz.into(),
        );
        fields.insert("video_clock_mhz".to_string(), params.video_clock_mhz.into());
        fields.insert("memory_bytes".to_string(), params.memory_bytes.into());
        fields.insert(
            "compute_percentage".to_string(),
            params.compute_percentage.into(),
        );
        fields.insert("memory_percentage".to_string(), params.memory_percentage.into());
        fields.insert("compute_tflops".to_string(), params.compute_tflops.into());
        fields.insert("power_usage".to_string(), params.power_usage.into());
        fields.insert("nvlink_rx".to_string(), params.nvlink_rx_bandwidth.into());
        fields.insert("nvlink_tx".to_string(), params.nvlink_tx_bandwidth.into());

        self.encode_metrics("tf_gpu_usage", &tags, &fields, params.timestamp)
    }

    /// Encode worker metrics using parameters struct
    fn encode_worker_metrics_with_params(&self, params: &WorkerMetricsParams) -> String {
        let mut tags = HashMap::new();
        tags.insert("node".to_string(), params.node_name.to_string());
        tags.insert("pool".to_string(), params.gpu_pool.to_string());
        tags.insert("uuid".to_string(), params.gpu_uuid.to_string());
        tags.insert("worker".to_string(), params.pod_identifier.to_string());
        tags.insert("namespace".to_string(), params.namespace.to_string());
        tags.insert("workload".to_string(), params.workload.to_string());

        // Add extra labels as tags - avoid unnecessary cloning
        for (key, value) in params.extra_labels {
            tags.insert(key.clone(), value.clone());
        }

        let mut fields = HashMap::new();
        fields.insert("memory_bytes".to_string(), params.memory_bytes.into());
        fields.insert(
            "compute_percentage".to_string(),
            params.compute_percentage.into(),
        );
        fields.insert("compute_tflops".to_string(), params.compute_tflops.into());
        fields.insert(
            "memory_percentage".to_string(),
            params.memory_percentage.into(),
        );

        self.encode_metrics("tf_worker_usage", &tags, &fields, params.timestamp)
    }
}

/// Concrete encoder without dynamic dispatch
pub enum Encoder {
    Json(json::JsonEncoder),
    Influx(influx::InfluxEncoder),
}

impl MetricsEncoder for Encoder {
    fn encode_metrics(
        &self,
        measurement: &str,
        tags: &HashMap<String, String>,
        fields: &HashMap<String, FieldValue>,
        timestamp: i64,
    ) -> String {
        match self {
            Encoder::Json(inner) => inner.encode_metrics(measurement, tags, fields, timestamp),
            Encoder::Influx(inner) => inner.encode_metrics(measurement, tags, fields, timestamp),
        }
    }
}

/// Factory function to create encoders based on format string (static dispatch)
pub fn create_encoder(format: &str) -> Encoder {
    match format.to_lowercase().as_str() {
        "json" => Encoder::Json(json::JsonEncoder::new()),
        "influx" => Encoder::Influx(influx::InfluxEncoder::new()),
        _ => Encoder::Influx(influx::InfluxEncoder::new()),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_field_value_from_string() {
        let value: FieldValue = "test".into();
        match value {
            FieldValue::String(s) => assert_eq!(s, "test"),
            _ => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_field_value_from_string_owned() {
        let value: FieldValue = String::from("test").into();
        match value {
            FieldValue::String(s) => assert_eq!(s, "test"),
            _ => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_field_value_from_i64() {
        let value: FieldValue = 42i64.into();
        match value {
            FieldValue::Integer(i) => assert_eq!(i, 42),
            _ => panic!("Expected Integer variant"),
        }
    }

    #[test]
    fn test_field_value_from_u64() {
        let value: FieldValue = 42u64.into();
        match value {
            FieldValue::UnsignedInteger(u) => assert_eq!(u, 42),
            _ => panic!("Expected UnsignedInteger variant"),
        }
    }

    #[test]
    fn test_field_value_from_f64() {
        let value: FieldValue = 42.5f64.into();
        match value {
            FieldValue::Float(f) => assert_eq!(f, 42.5),
            _ => panic!("Expected Float variant"),
        }
    }

    #[test]
    fn test_field_value_from_bool() {
        let value: FieldValue = true.into();
        match value {
            FieldValue::Boolean(b) => assert!(b),
            _ => panic!("Expected Boolean variant"),
        }
    }

    #[test]
    fn test_create_encoder_json() {
        let encoder = create_encoder("json");
        // Test that we can encode something basic
        let mut tags = HashMap::new();
        tags.insert("test".to_string(), "value".to_string());
        let mut fields = HashMap::new();
        fields.insert("metric".to_string(), 42.0.into());
        let result = encoder.encode_metrics("test_measurement", &tags, &fields, 1234567890);
        assert!(result.contains("test_measurement"));
        assert!(result.contains("json") || result.contains("{"));
    }

    #[test]
    fn test_create_encoder_influx() {
        let encoder = create_encoder("influx");
        let mut tags = HashMap::new();
        tags.insert("test".to_string(), "value".to_string());
        let mut fields = HashMap::new();
        fields.insert("metric".to_string(), 42.0.into());
        let result = encoder.encode_metrics("test_measurement", &tags, &fields, 1234567890);
        assert!(result.contains("test_measurement"));
    }

    #[test]
    fn test_create_encoder_influxdb() {
        let encoder = create_encoder("influxdb");
        let mut tags = HashMap::new();
        tags.insert("test".to_string(), "value".to_string());
        let mut fields = HashMap::new();
        fields.insert("metric".to_string(), 42.0.into());
        let result = encoder.encode_metrics("test_measurement", &tags, &fields, 1234567890);
        assert!(result.contains("test_measurement"));
    }

    #[test]
    fn test_create_encoder_default() {
        let encoder = create_encoder("unknown_format");
        let mut tags = HashMap::new();
        tags.insert("test".to_string(), "value".to_string());
        let mut fields = HashMap::new();
        fields.insert("metric".to_string(), 42.0.into());
        let result = encoder.encode_metrics("test_measurement", &tags, &fields, 1234567890);
        assert!(result.contains("test_measurement"));
    }

    #[test]
    fn test_encode_gpu_metrics() {
        let encoder = create_encoder("json");
        let result = encoder.encode_gpu_metrics_with_params(&GpuMetricsParams {
            gpu_uuid: "gpu-uuid-123",
            node_name: "node1",
            gpu_pool: "pool1",
            rx: 100.5,
            tx: 200.5,
            temperature: 75.0,
            graphics_clock_mhz: 1200.0,
            sm_clock_mhz: 1100.0,
            memory_clock_mhz: 2000.0,
            video_clock_mhz: 800.0,
            memory_bytes: 1024000000,
            compute_percentage: 85.5,
            compute_tflops: 12.5,
            power_usage: 150,
            nvlink_rx_bandwidth: 1000000,
            nvlink_tx_bandwidth: 2000000,
            timestamp: 1234567890,
            memory_percentage: 0.5,
        });

        // Should contain all the expected fields and tags
        assert!(result.contains("tf_gpu_usage"));
        assert!(result.contains("gpu-uuid-123"));
        assert!(result.contains("node1"));
        assert!(result.contains("pool1"));
    }

    #[test]
    fn test_encode_worker_metrics() {
        let encoder = create_encoder("json");
        let mut extra_labels = HashMap::new();
        extra_labels.insert("custom_label".to_string(), "custom_value".to_string());

        let result = encoder.encode_worker_metrics_with_params(&WorkerMetricsParams {
            gpu_uuid: "gpu-uuid-456",
            node_name: "node2",
            gpu_pool: "pool2",
            pod_identifier: "worker-123",
            namespace: "default",
            workload: "tensorflow",
            memory_bytes: 2048000000,
            compute_percentage: 90.0,
            compute_tflops: 15.0,
            memory_percentage: 0.5,
            timestamp: 1234567890,
            extra_labels: &extra_labels,
        });

        // Should contain all the expected fields and tags
        assert!(result.contains("tf_worker_usage"));
        assert!(result.contains("gpu-uuid-456"));
        assert!(result.contains("node2"));
        assert!(result.contains("pool2"));
        assert!(result.contains("worker-123"));
        assert!(result.contains("default"));
        assert!(result.contains("tensorflow"));
        assert!(result.contains("custom_label"));
        assert!(result.contains("custom_value"));
    }

    #[test]
    fn test_encode_gpu_metrics_influx() {
        let encoder = create_encoder("influx");
        let result = encoder.encode_gpu_metrics_with_params(&GpuMetricsParams {
            gpu_uuid: "gpu-uuid-789",
            node_name: "node3",
            gpu_pool: "pool3",
            rx: 150.0,
            tx: 250.0,
            temperature: 80.0,
            graphics_clock_mhz: 1300.0,
            sm_clock_mhz: 1150.0,
            memory_clock_mhz: 2100.0,
            video_clock_mhz: 850.0,
            memory_bytes: 4096000000,
            compute_percentage: 88.0,
            compute_tflops: 25.0,
            power_usage: 200,
            nvlink_rx_bandwidth: -1,
            nvlink_tx_bandwidth: -1,
            memory_percentage: 92.0,
            timestamp: 1234567890,
        });

        // InfluxDB line protocol format
        assert!(result.contains("tf_gpu_usage"));
        assert!(result.contains("node=node3"));
        assert!(result.contains("pool=pool3"));
        assert!(result.contains("uuid=gpu-uuid-789"));
        assert!(result.contains("1234567890"));
    }

    #[test]
    fn test_encode_worker_metrics_influx() {
        let encoder = create_encoder("influx");
        let mut extra_labels = HashMap::new();
        extra_labels.insert("env".to_string(), "production".to_string());

        let result = encoder.encode_worker_metrics_with_params(&WorkerMetricsParams {
            gpu_uuid: "gpu-uuid-abc",
            node_name: "node4",
            gpu_pool: "pool4",
            pod_identifier: "worker-456",
            namespace: "kube-system",
            workload: "pytorch",
            memory_bytes: 8192000000,
            compute_percentage: 88.0,
            compute_tflops: 25.0,
            memory_percentage: 92.0,
            timestamp: 1234567890,
            extra_labels: &extra_labels,
        });

        // InfluxDB line protocol format
        assert!(result.contains("tf_worker_usage"));
        assert!(result.contains("node=node4"));
        assert!(result.contains("pool=pool4"));
        assert!(result.contains("uuid=gpu-uuid-abc"));
        assert!(result.contains("worker=worker-456"));
        assert!(result.contains("namespace=kube-system"));
        assert!(result.contains("workload=pytorch"));
        assert!(result.contains("env=production"));
        assert!(result.contains("1234567890"));
    }

    #[test]
    fn test_encode_metrics_empty_fields() {
        let encoder = create_encoder("json");
        let tags = HashMap::new();
        let fields = HashMap::new();
        let result = encoder.encode_metrics("empty_test", &tags, &fields, 1234567890);
        assert!(result.contains("empty_test"));
    }

    #[test]
    fn test_encode_metrics_empty_tags() {
        let encoder = create_encoder("influx");
        let tags = HashMap::new();
        let mut fields = HashMap::new();
        fields.insert("value".to_string(), 42.0.into());
        let result = encoder.encode_metrics("no_tags_test", &tags, &fields, 1234567890);
        assert!(result.contains("no_tags_test"));
        assert!(result.contains("value=42"));
    }

    #[test]
    fn test_encode_metrics_multiple_field_types() {
        let encoder = create_encoder("json");
        let mut tags = HashMap::new();
        tags.insert("service".to_string(), "test".to_string());

        let mut fields = HashMap::new();
        fields.insert("string_field".to_string(), "hello".into());
        fields.insert("int_field".to_string(), 42i64.into());
        fields.insert("uint_field".to_string(), 100u64.into());
        fields.insert("float_field".to_string(), std::f64::consts::PI.into());
        fields.insert("bool_field".to_string(), true.into());

        let result = encoder.encode_metrics("mixed_types", &tags, &fields, 1234567890);
        assert!(result.contains("mixed_types"));
        assert!(result.contains("hello"));
        assert!(result.contains("42"));
        assert!(result.contains("100"));
        assert!(result.contains(std::f64::consts::PI.to_string().as_str()));
        assert!(result.contains("true"));
    }
}
