use std::collections::HashMap;

use serde_json::json;

use super::FieldValue;
use super::MetricsEncoder;

/// JSON encoder for metrics
pub struct JsonEncoder;

impl JsonEncoder {
    pub fn new() -> Self {
        Self
    }
}

impl MetricsEncoder for JsonEncoder {
    fn encode_metrics(
        &self,
        measurement: &str,
        tags: &HashMap<String, String>,
        fields: &HashMap<String, FieldValue>,
        timestamp: i64,
    ) -> String {
        // Convert FieldValue enum to proper JSON values without intermediate collection
        let json_fields: serde_json::Map<String, serde_json::Value> = fields
            .iter()
            .map(|(k, v)| {
                let json_value = match v {
                    FieldValue::String(s) => serde_json::Value::String(s.clone()),
                    FieldValue::Integer(i) => {
                        serde_json::Value::Number(serde_json::Number::from(*i))
                    }
                    FieldValue::UnsignedInteger(u) => {
                        serde_json::Value::Number(serde_json::Number::from(*u))
                    }
                    FieldValue::Float(f) => serde_json::Value::Number(
                        serde_json::Number::from_f64(*f).unwrap_or(serde_json::Number::from(0)),
                    ),
                    FieldValue::Boolean(b) => serde_json::Value::Bool(*b),
                };
                (k.clone(), json_value)
            })
            .collect();

        let metrics = json!({
            "measure": measurement,
            "ts": timestamp,
            "tag": tags,
            "field": json_fields,
        });
        metrics.to_string() + "\n"
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::Value;

    use super::*;

    #[test]
    fn test_encode_metrics_basic() {
        let encoder = JsonEncoder::new();
        let mut tags = HashMap::new();
        tags.insert("host".to_string(), "server1".to_string());
        tags.insert("region".to_string(), "us-west".to_string());

        let mut fields = HashMap::new();
        fields.insert("cpu_usage".to_string(), 85.5.into());
        fields.insert("memory_usage".to_string(), 1024u64.into());

        let result = encoder.encode_metrics("system_metrics", &tags, &fields, 1609459200);

        // Parse the JSON to verify structure
        let parsed: Value = serde_json::from_str(&result).expect("Should be valid JSON");

        assert_eq!(parsed["measure"], "system_metrics");
        assert_eq!(parsed["ts"], 1609459200);
        assert_eq!(parsed["tag"]["host"], "server1");
        assert_eq!(parsed["tag"]["region"], "us-west");
        assert_eq!(parsed["field"]["cpu_usage"], 85.5);
        assert_eq!(parsed["field"]["memory_usage"], 1024);
    }

    #[test]
    fn test_encode_metrics_empty_tags() {
        let encoder = JsonEncoder::new();
        let tags = HashMap::new();
        let mut fields = HashMap::new();
        fields.insert("value".to_string(), 42.0.into());

        let result = encoder.encode_metrics("test_metric", &tags, &fields, 1234567890);
        let parsed: Value = serde_json::from_str(&result).expect("Should be valid JSON");

        assert_eq!(parsed["measure"], "test_metric");
        assert_eq!(parsed["ts"], 1234567890);
        assert!(parsed["tag"].as_object().unwrap().is_empty());
        assert_eq!(parsed["field"]["value"], 42.0);
    }

    #[test]
    fn test_encode_metrics_empty_fields() {
        let encoder = JsonEncoder::new();
        let mut tags = HashMap::new();
        tags.insert("service".to_string(), "api".to_string());
        let fields = HashMap::new();

        let result = encoder.encode_metrics("empty_fields", &tags, &fields, 1234567890);
        let parsed: Value = serde_json::from_str(&result).expect("Should be valid JSON");

        assert_eq!(parsed["measure"], "empty_fields");
        assert_eq!(parsed["ts"], 1234567890);
        assert_eq!(parsed["tag"]["service"], "api");
        assert!(parsed["field"].as_object().unwrap().is_empty());
    }

    #[test]
    fn test_encode_metrics_all_field_types() {
        let encoder = JsonEncoder::new();
        let mut tags = HashMap::new();
        tags.insert("app".to_string(), "test".to_string());

        let mut fields = HashMap::new();
        fields.insert("string_val".to_string(), "hello world".into());
        fields.insert("int_val".to_string(), (-42i64).into());
        fields.insert("uint_val".to_string(), 42u64.into());
        fields.insert("float_val".to_string(), std::f64::consts::PI.into());
        fields.insert("bool_val".to_string(), true.into());

        let result = encoder.encode_metrics("mixed_types", &tags, &fields, 1234567890);
        let parsed: Value = serde_json::from_str(&result).expect("Should be valid JSON");

        assert_eq!(parsed["measure"], "mixed_types");
        assert_eq!(parsed["ts"], 1234567890);
        assert_eq!(parsed["tag"]["app"], "test");
        assert_eq!(parsed["field"]["string_val"], "hello world");
        assert_eq!(parsed["field"]["int_val"], -42);
        assert_eq!(parsed["field"]["uint_val"], 42);
        assert_eq!(parsed["field"]["float_val"], std::f64::consts::PI);
        assert_eq!(parsed["field"]["bool_val"], true);
    }

    #[test]
    fn test_encode_gpu_metrics() {
        let encoder = JsonEncoder::new();
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
            1609459200,
        );

        let parsed: Value = serde_json::from_str(&result).expect("Should be valid JSON");

        assert_eq!(parsed["measure"], "tf_gpu_usage");
        assert_eq!(parsed["ts"], 1609459200);
        assert_eq!(parsed["tag"]["node"], "worker-node-1");
        assert_eq!(parsed["tag"]["pool"], "default-pool");
        assert_eq!(parsed["tag"]["uuid"], "gpu-123");
        assert_eq!(parsed["field"]["rx"], 150.0);
        assert_eq!(parsed["field"]["tx"], 250.0);
        assert_eq!(parsed["field"]["temperature"], 78.5);
        assert_eq!(parsed["field"]["memory_bytes"], 2048000000u64);
        assert_eq!(parsed["field"]["compute_percentage"], 92.3);
        assert_eq!(parsed["field"]["compute_tflops"], 18.7);
    }

    #[test]
    fn test_encode_worker_metrics() {
        let encoder = JsonEncoder::new();
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
            1609459200,
            &extra_labels,
        );

        let parsed: Value = serde_json::from_str(&result).expect("Should be valid JSON");

        assert_eq!(parsed["measure"], "tf_worker_usage");
        assert_eq!(parsed["ts"], 1609459200);
        assert_eq!(parsed["tag"]["node"], "worker-node-2");
        assert_eq!(parsed["tag"]["pool"], "ml-pool");
        assert_eq!(parsed["tag"]["uuid"], "gpu-456");
        assert_eq!(parsed["tag"]["worker"], "worker-abc");
        assert_eq!(parsed["tag"]["namespace"], "ml-namespace");
        assert_eq!(parsed["tag"]["workload"], "pytorch-training");
        assert_eq!(parsed["tag"]["environment"], "production");
        assert_eq!(parsed["tag"]["team"], "ml-ops");
        assert_eq!(parsed["field"]["memory_bytes"], 4096000000u64);
        assert_eq!(parsed["field"]["compute_percentage"], 88.5);
        assert_eq!(parsed["field"]["compute_tflops"], 22.1);
        assert_eq!(parsed["field"]["memory_percentage"], 95.2);
    }

    #[test]
    fn test_encode_worker_metrics_no_extra_labels() {
        let encoder = JsonEncoder::new();
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
            1609459200,
            &extra_labels,
        );

        let parsed: Value = serde_json::from_str(&result).expect("Should be valid JSON");

        assert_eq!(parsed["measure"], "tf_worker_usage");
        assert_eq!(parsed["tag"]["node"], "worker-node-3");
        assert_eq!(parsed["tag"]["pool"], "test-pool");
        assert_eq!(parsed["tag"]["uuid"], "gpu-789");
        assert_eq!(parsed["tag"]["worker"], "worker-def");
        assert_eq!(parsed["tag"]["namespace"], "default");
        assert_eq!(parsed["tag"]["workload"], "tensorflow");
        // Should not have extra labels
        assert!(parsed["tag"].get("environment").is_none());
        assert!(parsed["tag"].get("team").is_none());
    }

    #[test]
    fn test_json_output_is_valid() {
        let encoder = JsonEncoder::new();
        let mut tags = HashMap::new();
        tags.insert(
            "key with spaces".to_string(),
            "value with spaces".to_string(),
        );
        tags.insert(
            "key_with_special_chars".to_string(),
            "value!@#$%^&*()".to_string(),
        );

        let mut fields = HashMap::new();
        fields.insert(
            "field with spaces".to_string(),
            "string with \"quotes\"".into(),
        );
        fields.insert("numeric_field".to_string(), 123.456.into());

        let result = encoder.encode_metrics("special_chars_test", &tags, &fields, 1234567890);

        // Should be able to parse as valid JSON despite special characters
        let parsed: Value = serde_json::from_str(&result).expect("Should be valid JSON");
        assert_eq!(parsed["measure"], "special_chars_test");
        assert_eq!(parsed["tag"]["key with spaces"], "value with spaces");
        assert_eq!(parsed["tag"]["key_with_special_chars"], "value!@#$%^&*()");
        assert_eq!(
            parsed["field"]["field with spaces"],
            "string with \"quotes\""
        );
        assert_eq!(parsed["field"]["numeric_field"], 123.456);
    }
}
