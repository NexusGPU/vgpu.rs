use std::collections::HashMap;

use influxdb_line_protocol::LineProtocolBuilder;

use super::FieldValue;
use super::MetricsEncoder;
use crate::metrics::BytesWrapper;

/// InfluxDB line protocol encoder
pub struct InfluxEncoder;

impl InfluxEncoder {
    pub fn new() -> Self {
        Self
    }
}

impl MetricsEncoder for InfluxEncoder {
    fn encode_metrics(
        &self,
        measurement: &str,
        tags: &HashMap<String, String>,
        fields: &HashMap<String, FieldValue>,
        timestamp: i64,
    ) -> String {
        // Start with measurement
        let mut builder = LineProtocolBuilder::new().measurement(measurement);

        // Add all tags
        for (key, value) in tags {
            builder = builder.tag(key, value);
        }

        // Add fields - we need to handle the first field specially to get the right type state
        let mut field_entries: Vec<_> = fields.iter().collect();
        field_entries.sort_by_key(|(k, _)| *k); // Sort for consistent ordering

        if let Some((first_key, first_value)) = field_entries.first() {
            // Add the first field to transition to AfterField state
            let mut after_first_field = match first_value {
                FieldValue::String(s) => builder.field(first_key, s.as_str()),
                FieldValue::Integer(i) => builder.field(first_key, *i),
                FieldValue::UnsignedInteger(u) => builder.field(first_key, *u),
                FieldValue::Float(f) => builder.field(first_key, *f),
                FieldValue::Boolean(b) => builder.field(first_key, *b),
            };

            // Add remaining fields
            for (key, value) in field_entries.iter().skip(1) {
                after_first_field = match value {
                    FieldValue::String(s) => after_first_field.field(key, s.as_str()),
                    FieldValue::Integer(i) => after_first_field.field(key, *i),
                    FieldValue::UnsignedInteger(u) => after_first_field.field(key, *u),
                    FieldValue::Float(f) => after_first_field.field(key, *f),
                    FieldValue::Boolean(b) => after_first_field.field(key, *b),
                };
            }

            let lp_built = after_first_field.timestamp(timestamp).close_line().build();
            BytesWrapper::from(lp_built).to_string()
        } else {
            // No fields case - this shouldn't happen in practice but handle gracefully
            let lp_built = builder
                .field("_empty", true)
                .timestamp(timestamp)
                .close_line()
                .build();
            BytesWrapper::from(lp_built).to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_encode_metrics_basic() {
        let encoder = InfluxEncoder::new();
        let mut tags = HashMap::new();
        tags.insert("host".to_string(), "server1".to_string());
        tags.insert("region".to_string(), "us-west".to_string());

        let mut fields = HashMap::new();
        fields.insert("cpu_usage".to_string(), 85.5.into());
        fields.insert("memory_usage".to_string(), 1024u64.into());

        let result = encoder.encode_metrics("system_metrics", &tags, &fields, 1609459200000000000);

        // Verify InfluxDB line protocol format
        assert!(result.starts_with("system_metrics"));
        assert!(result.contains("host=server1"));
        assert!(result.contains("region=us-west"));
        assert!(result.contains("cpu_usage=85.5"));
        assert!(result.contains("memory_usage=1024u"));
        assert!(result.contains("1609459200000000000"));
    }

    #[test]
    fn test_encode_metrics_single_tag_single_field() {
        let encoder = InfluxEncoder::new();
        let mut tags = HashMap::new();
        tags.insert("service".to_string(), "api".to_string());

        let mut fields = HashMap::new();
        fields.insert("response_time".to_string(), 123.45.into());

        let result = encoder.encode_metrics("http_requests", &tags, &fields, 1234567890000000000);

        assert!(result.starts_with("http_requests"));
        assert!(result.contains("service=api"));
        assert!(result.contains("response_time=123.45"));
        assert!(result.contains("1234567890000000000"));
    }

    #[test]
    fn test_encode_metrics_no_tags() {
        let encoder = InfluxEncoder::new();
        let tags = HashMap::new();
        let mut fields = HashMap::new();
        fields.insert("value".to_string(), 42.0.into());

        let result = encoder.encode_metrics("simple_metric", &tags, &fields, 1234567890000000000);

        assert!(result.starts_with("simple_metric "));
        assert!(result.contains("value=42"));
        assert!(result.contains("1234567890000000000"));
        // Should not contain any tag separators
        assert!(!result.contains(",") || result.matches(',').count() == 0);
    }

    #[test]
    fn test_encode_metrics_empty_fields() {
        let encoder = InfluxEncoder::new();
        let mut tags = HashMap::new();
        tags.insert("service".to_string(), "test".to_string());
        let fields = HashMap::new();

        let result = encoder.encode_metrics("empty_fields", &tags, &fields, 1234567890000000000);

        // Should handle empty fields gracefully with _empty field
        assert!(result.starts_with("empty_fields"));
        assert!(result.contains("service=test"));
        assert!(result.contains("_empty=true"));
        assert!(result.contains("1234567890000000000"));
    }

    #[test]
    fn test_encode_metrics_all_field_types() {
        let encoder = InfluxEncoder::new();
        let mut tags = HashMap::new();
        tags.insert("app".to_string(), "test".to_string());

        let mut fields = HashMap::new();
        fields.insert("string_val".to_string(), "hello world".into());
        fields.insert("int_val".to_string(), (-42i64).into());
        fields.insert("uint_val".to_string(), 42u64.into());
        fields.insert("float_val".to_string(), std::f64::consts::PI.into());
        fields.insert("bool_val".to_string(), true.into());

        let result = encoder.encode_metrics("mixed_types", &tags, &fields, 1234567890000000000);

        assert!(result.starts_with("mixed_types"));
        assert!(result.contains("app=test"));
        assert!(result.contains("string_val=\"hello world\""));
        assert!(result.contains("int_val=-42i"));
        assert!(result.contains("uint_val=42u"));
        assert!(result.contains(&format!("float_val={}", std::f64::consts::PI)));
        assert!(result.contains("bool_val=true"));
        assert!(result.contains("1234567890000000000"));
    }

    #[test]
    fn test_encode_gpu_metrics() {
        let encoder = InfluxEncoder::new();
        let result = encoder.encode_gpu_metrics(
            "gpu-123",
            "worker-node-1",
            "default-pool",
            150.0,
            250.0,
            78.5,
            2048000000,
            92.3,
            18.7,
            1609459200000000000,
        );

        assert!(result.starts_with("tf_gpu_usage"));
        assert!(result.contains("node=worker-node-1"));
        assert!(result.contains("pool=default-pool"));
        assert!(result.contains("uuid=gpu-123"));
        assert!(result.contains("rx=150"));
        assert!(result.contains("tx=250"));
        assert!(result.contains("temperature=78.5"));
        assert!(result.contains("memory_bytes=2048000000u"));
        assert!(result.contains("compute_percentage=92.3"));
        assert!(result.contains("compute_tflops=18.7"));
        assert!(result.contains("1609459200000000000"));
    }

    #[test]
    fn test_encode_worker_metrics() {
        let encoder = InfluxEncoder::new();
        let mut extra_labels = HashMap::new();
        extra_labels.insert("environment".to_string(), "production".to_string());
        extra_labels.insert("team".to_string(), "ml-ops".to_string());

        let result = encoder.encode_worker_metrics(
            "gpu-456",
            "worker-node-2",
            "ml-pool",
            "worker-abc",
            "ml-namespace",
            "pytorch-training",
            4096000000,
            88.5,
            22.1,
            95.2,
            1609459200000000000,
            &extra_labels,
        );

        assert!(result.starts_with("tf_worker_usage"));
        assert!(result.contains("node=worker-node-2"));
        assert!(result.contains("pool=ml-pool"));
        assert!(result.contains("uuid=gpu-456"));
        assert!(result.contains("worker=worker-abc"));
        assert!(result.contains("namespace=ml-namespace"));
        assert!(result.contains("workload=pytorch-training"));
        assert!(result.contains("environment=production"));
        assert!(result.contains("team=ml-ops"));
        assert!(result.contains("memory_bytes=4096000000u"));
        assert!(result.contains("compute_percentage=88.5"));
        assert!(result.contains("compute_tflops=22.1"));
        assert!(result.contains("memory_percentage=95.2"));
        assert!(result.contains("1609459200000000000"));
    }

    #[test]
    fn test_encode_worker_metrics_no_extra_labels() {
        let encoder = InfluxEncoder::new();
        let extra_labels = HashMap::new();

        let result = encoder.encode_worker_metrics(
            "gpu-789",
            "worker-node-3",
            "test-pool",
            "worker-def",
            "default",
            "tensorflow",
            1024000000,
            75.0,
            15.5,
            80.0,
            1609459200000000000,
            &extra_labels,
        );

        assert!(result.starts_with("tf_worker_usage"));
        assert!(result.contains("node=worker-node-3"));
        assert!(result.contains("pool=test-pool"));
        assert!(result.contains("uuid=gpu-789"));
        assert!(result.contains("worker=worker-def"));
        assert!(result.contains("namespace=default"));
        assert!(result.contains("workload=tensorflow"));
        // Should not have extra labels
        assert!(!result.contains("environment="));
        assert!(!result.contains("team="));
        assert!(result.contains("1609459200000000000"));
    }

    #[test]
    fn test_field_ordering_consistency() {
        let encoder = InfluxEncoder::new();
        let mut tags = HashMap::new();
        tags.insert("service".to_string(), "test".to_string());

        let mut fields = HashMap::new();
        fields.insert("z_field".to_string(), 1.0.into());
        fields.insert("a_field".to_string(), 2.0.into());
        fields.insert("m_field".to_string(), 3.0.into());

        let result1 = encoder.encode_metrics("test_ordering", &tags, &fields, 1234567890000000000);
        let result2 = encoder.encode_metrics("test_ordering", &tags, &fields, 1234567890000000000);

        // Results should be identical (fields are sorted for consistency)
        assert_eq!(result1, result2);

        // Fields should appear in alphabetical order
        let a_pos = result1.find("a_field").unwrap();
        let m_pos = result1.find("m_field").unwrap();
        let z_pos = result1.find("z_field").unwrap();

        assert!(a_pos < m_pos);
        assert!(m_pos < z_pos);
    }

    #[test]
    fn test_special_characters_in_tags_and_fields() {
        let encoder = InfluxEncoder::new();
        let mut tags = HashMap::new();
        tags.insert("service".to_string(), "test-service".to_string());
        tags.insert("env".to_string(), "prod_env".to_string());

        let mut fields = HashMap::new();
        fields.insert("message".to_string(), "hello world with spaces".into());
        fields.insert("count".to_string(), 42i64.into());

        let result = encoder.encode_metrics("special_chars", &tags, &fields, 1234567890000000000);

        assert!(result.starts_with("special_chars"));
        assert!(result.contains("service=test-service"));
        assert!(result.contains("env=prod_env"));
        assert!(result.contains("message=\"hello world with spaces\""));
        assert!(result.contains("count=42i"));
        assert!(result.contains("1234567890000000000"));
    }

    #[test]
    fn test_boolean_field_encoding() {
        let encoder = InfluxEncoder::new();
        let tags = HashMap::new();
        let mut fields = HashMap::new();
        fields.insert("is_active".to_string(), true.into());
        fields.insert("is_disabled".to_string(), false.into());

        let result = encoder.encode_metrics("boolean_test", &tags, &fields, 1234567890000000000);

        assert!(result.contains("is_active=true"));
        assert!(result.contains("is_disabled=false"));
    }

    #[test]
    fn test_negative_numbers() {
        let encoder = InfluxEncoder::new();
        let tags = HashMap::new();
        let mut fields = HashMap::new();
        fields.insert("negative_int".to_string(), (-123i64).into());
        fields.insert("negative_float".to_string(), (-45.67f64).into());

        let result = encoder.encode_metrics("negative_test", &tags, &fields, 1234567890000000000);

        assert!(result.contains("negative_int=-123i"));
        assert!(result.contains("negative_float=-45.67"));
    }

    #[test]
    fn test_zero_values() {
        let encoder = InfluxEncoder::new();
        let tags = HashMap::new();
        let mut fields = HashMap::new();
        fields.insert("zero_int".to_string(), 0i64.into());
        fields.insert("zero_uint".to_string(), 0u64.into());
        fields.insert("zero_float".to_string(), 0.0f64.into());

        let result = encoder.encode_metrics("zero_test", &tags, &fields, 1234567890000000000);

        assert!(result.contains("zero_int=0i"));
        assert!(result.contains("zero_uint=0u"));
        assert!(result.contains("zero_float=0"));
    }
}
