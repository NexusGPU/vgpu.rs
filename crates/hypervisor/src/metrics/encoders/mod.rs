use std::collections::HashMap;

pub mod influx;
pub mod json;

/// Represents a field value that can be encoded in metrics
#[derive(Debug, Clone, serde::Serialize)]
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

/// Trait for encoding metrics data into different formats
pub trait MetricsEncoder: Send + Sync {
    /// Encode metrics with measurement name, tags, fields, and timestamp
    fn encode_metrics(
        &self,
        measurement: &str,
        tags: &HashMap<String, String>,
        fields: &HashMap<String, FieldValue>,
        timestamp: i64,
    ) -> String;

    /// Encode GPU metrics (convenience method)
    fn encode_gpu_metrics(
        &self,
        gpu_uuid: &str,
        node_name: &str,
        gpu_pool: &str,
        rx: f64,
        tx: f64,
        temperature: f64,
        memory_bytes: u64,
        compute_percentage: f64,
        compute_tflops: f64,
        timestamp: i64,
    ) -> String {
        let mut tags = HashMap::new();
        tags.insert("node".to_string(), node_name.to_string());
        tags.insert("pool".to_string(), gpu_pool.to_string());
        tags.insert("uuid".to_string(), gpu_uuid.to_string());

        let mut fields = HashMap::new();
        fields.insert("rx".to_string(), rx.into());
        fields.insert("tx".to_string(), tx.into());
        fields.insert("temperature".to_string(), temperature.into());
        fields.insert("memory_bytes".to_string(), memory_bytes.into());
        fields.insert("compute_percentage".to_string(), compute_percentage.into());
        fields.insert("compute_tflops".to_string(), compute_tflops.into());

        self.encode_metrics("tf_gpu_usage", &tags, &fields, timestamp)
    }

    /// Encode worker metrics (convenience method)
    fn encode_worker_metrics(
        &self,
        gpu_uuid: &str,
        node_name: &str,
        gpu_pool: &str,
        worker_identifier: &str,
        namespace: &str,
        workload: &str,
        memory_bytes: u64,
        compute_percentage: f64,
        compute_tflops: f64,
        memory_percentage: f64,
        timestamp: i64,
        extra_labels: &HashMap<String, String>,
    ) -> String {
        let mut tags = HashMap::new();
        tags.insert("node".to_string(), node_name.to_string());
        tags.insert("pool".to_string(), gpu_pool.to_string());
        tags.insert("uuid".to_string(), gpu_uuid.to_string());
        tags.insert("worker".to_string(), worker_identifier.to_string());
        tags.insert("namespace".to_string(), namespace.to_string());
        tags.insert("workload".to_string(), workload.to_string());

        // Add extra labels as tags
        for (key, value) in extra_labels {
            tags.insert(key.clone(), value.clone());
        }

        let mut fields = HashMap::new();
        fields.insert("memory_bytes".to_string(), memory_bytes.into());
        fields.insert("compute_percentage".to_string(), compute_percentage.into());
        fields.insert("compute_tflops".to_string(), compute_tflops.into());
        fields.insert("memory_percentage".to_string(), memory_percentage.into());

        self.encode_metrics("tf_worker_usage", &tags, &fields, timestamp)
    }
}

/// Factory function to create encoders based on format string
pub fn create_encoder(format: &str) -> Box<dyn MetricsEncoder + Send + Sync> {
    match format.to_lowercase().as_str() {
        "json" => Box::new(json::JsonEncoder::new()),
        "influx" | _ => Box::new(influx::InfluxEncoder::new()),
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
        let result = encoder.encode_gpu_metrics(
            "gpu-uuid-123",
            "node1",
            "pool1",
            100.5,
            200.5,
            75.0,
            1024000000,
            85.5,
            12.5,
            1234567890,
        );

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

        let result = encoder.encode_worker_metrics(
            "gpu-uuid-456",
            "node2",
            "pool2",
            "worker-123",
            "default",
            "tensorflow",
            2048000000,
            90.0,
            15.0,
            75.0,
            1234567890,
            &extra_labels,
        );

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
        let result = encoder.encode_gpu_metrics(
            "gpu-uuid-789",
            "node3",
            "pool3",
            150.0,
            250.0,
            80.0,
            4096000000,
            95.0,
            20.0,
            1234567890,
        );

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

        let result = encoder.encode_worker_metrics(
            "gpu-uuid-abc",
            "node4",
            "pool4",
            "worker-456",
            "kube-system",
            "pytorch",
            8192000000,
            88.0,
            25.0,
            92.0,
            1234567890,
            &extra_labels,
        );

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
        fields.insert("float_field".to_string(), 3.14f64.into());
        fields.insert("bool_field".to_string(), true.into());

        let result = encoder.encode_metrics("mixed_types", &tags, &fields, 1234567890);
        assert!(result.contains("mixed_types"));
        assert!(result.contains("hello"));
        assert!(result.contains("42"));
        assert!(result.contains("100"));
        assert!(result.contains("3.14"));
        assert!(result.contains("true"));
    }
}
