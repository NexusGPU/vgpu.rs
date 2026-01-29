use std::collections::BTreeMap;

use api_types::PodResourceInfo;
use api_types::QosLevel;
use error_stack::Report;
use error_stack::ResultExt;

use crate::platform::k8s::KubernetesError;

/// Domain prefix for tensor-fusion annotations.
const TENSOR_FUSION_DOMAIN: &str = "tensor-fusion.ai";

/// Tensor-fusion specific annotations extracted from Kubernetes pods.
#[derive(Debug, Clone, Default)]
pub struct TensorFusionPodInfo(pub PodResourceInfo);

impl TensorFusionPodInfo {
    /// Parse tensor-fusion annotations from a Kubernetes pod's annotations.
    ///
    /// Extracts and validates tensor-fusion specific annotations from the provided
    /// annotation map. Only processes annotations with the tensor-fusion.ai domain.
    ///
    /// # Errors
    ///
    /// - [`KubernetesError::AnnotationParseError`] if annotation values are invalid
    pub(crate) fn from_pod_annotations_labels(
        annotations: &BTreeMap<String, String>,
        labels: &BTreeMap<String, String>,
    ) -> Result<Self, Report<KubernetesError>> {
        let mut pod_info = PodResourceInfo {
            labels: labels.clone(),
            workload_name: labels
                .get(&format!("{TENSOR_FUSION_DOMAIN}/workload"))
                .cloned(),
            ..Default::default()
        };

        // Parse TFLOPS request
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/tflops-request")) {
            pod_info.tflops_request = Some(parse_tflops_value(value)?);
        }

        // Parse VRAM request
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/vram-request")) {
            pod_info.vram_request = Some(parse_memory_value(value)?);
        }

        // Parse TFLOPS limit
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/tflops-limit")) {
            pod_info.tflops_limit = Some(parse_tflops_value(value)?);
        }

        // Parse VRAM limit
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/vram-limit")) {
            pod_info.vram_limit = Some(parse_memory_value(value)?);
        }

        // Parse GPU UUIDs
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/gpu-ids")) {
            pod_info.gpu_uuids = Some(value.split(',').map(|s| s.to_string()).collect());
        }

        // Parse container-level GPU mappings
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/container-gpus")) {
            let container_gpus: BTreeMap<String, Vec<String>> = serde_json::from_str(value)
                .change_context(KubernetesError::AnnotationParseError {
                    message: format!("Invalid container-gpus JSON format: {value}"),
                })?;

            // Clean and validate: trim keys/values, filter empty GPU lists
            let cleaned_container_gpus: BTreeMap<String, Vec<String>> = container_gpus
                .into_iter()
                .map(|(container_name, gpu_list)| {
                    let cleaned_gpus: Vec<String> = gpu_list
                        .into_iter()
                        .map(|gpu| gpu.trim().to_string())
                        .filter(|gpu| !gpu.is_empty())
                        .collect();
                    (container_name.trim().to_string(), cleaned_gpus)
                })
                .filter(|(_, gpus)| !gpus.is_empty())
                .collect();

            if !cleaned_container_gpus.is_empty() {
                pod_info.container_gpu_uuids = Some(cleaned_container_gpus);
            }
        }

        // Parse QoS level
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/qos")) {
            // Extract worker configuration from annotations
            let qos_level = match value.as_str() {
                "High" | "high" => QosLevel::High,
                "Low" | "low" => QosLevel::Low,
                "Medium" | "medium" => QosLevel::Medium,
                "Critical" | "critical" => QosLevel::Critical,
                _ => QosLevel::Medium,
            };

            pod_info.qos_level = Some(qos_level);
        }

        // Parse container names
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/inject-container")) {
            pod_info.containers = Some(value.split(',').map(|s| s.to_string()).collect());
        }

        // Parse isolation level
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/isolation")) {
            pod_info.compute_shard = value == "shard";
            pod_info.isolation = Some(value.clone());
        } else {
            pod_info.compute_shard = false;
            pod_info.isolation = None;
        }

        Ok(Self(pod_info))
    }

    /// Check if any tensor-fusion annotations are present.
    pub(crate) const fn has_annotations(&self) -> bool {
        self.0.tflops_request.is_some()
            || self.0.vram_request.is_some()
            || self.0.tflops_limit.is_some()
            || self.0.vram_limit.is_some()
            || self.0.gpu_uuids.is_some()
            || self.0.container_gpu_uuids.is_some()
            || self.0.qos_level.is_some()
            || self.0.containers.is_some()
    }
}

/// Parse TFLOPS value from string, supporting units like "71200m", "5k", etc.
///
/// Supports the following units:
/// - No suffix: raw TFLOPS value
/// - "m": milli-TFLOPS (0.001)
/// - "k" or "K": kilo-TFLOPS (1000)
/// - "M": mega-TFLOPS (1,000,000)
/// - "G": giga-TFLOPS (1,000,000,000)
///
/// # Errors
///
/// - [`KubernetesError::AnnotationParseError`] if the TFLOPS value format is invalid
fn parse_tflops_value(value: &str) -> Result<f64, Report<KubernetesError>> {
    let value = value.trim();

    // Handle plain numbers (raw TFLOPS)
    if let Ok(tflops) = value.parse::<f64>() {
        return Ok(tflops);
    }

    // Find the numeric part and unit part
    let (numeric_part, unit) = if let Some(pos) = value.find(|c: char| c.is_alphabetic()) {
        (&value[..pos], &value[pos..])
    } else {
        return Err(Report::new(KubernetesError::AnnotationParseError {
            message: format!("Invalid TFLOPS value format: {value}"),
        }));
    };

    let numeric_value: f64 =
        numeric_part
            .parse::<f64>()
            .change_context(KubernetesError::AnnotationParseError {
                message: format!("Invalid numeric part in TFLOPS value: {numeric_part}"),
            })?;

    let multiplier = match unit {
        "m" => 0.001,
        "k" | "K" => 1000.0,
        "M" => 1_000_000.0,
        "G" => 1_000_000_000.0,
        _ => {
            return Err(Report::new(KubernetesError::AnnotationParseError {
                message: format!("Unsupported TFLOPS unit: {unit}"),
            }));
        }
    };

    Ok(numeric_value * multiplier)
}

/// Parse memory value from string, supporting units like "1Gi", "500Mi", etc.
///
/// Supports the following units:
/// - Bytes: no suffix or "B"
/// - Kilobytes: "Ki", "K"
/// - Megabytes: "Mi", "M"
/// - Gigabytes: "Gi", "G"
/// - Terabytes: "Ti", "T"
///
/// # Errors
///
/// - [`KubernetesError::AnnotationParseError`] if the memory value format is invalid
fn parse_memory_value(value: &str) -> Result<u64, Report<KubernetesError>> {
    let value = value.trim();

    // Handle plain numbers (bytes)
    if let Ok(bytes) = value.parse::<u64>() {
        return Ok(bytes);
    }

    // Find the numeric part and unit part
    let (numeric_part, unit) = if let Some(pos) = value.find(|c: char| c.is_alphabetic()) {
        (&value[..pos], &value[pos..])
    } else {
        return Err(Report::new(KubernetesError::AnnotationParseError {
            message: format!("Invalid memory value format: {value}"),
        }));
    };

    let numeric_value: f64 =
        numeric_part
            .parse::<f64>()
            .change_context(KubernetesError::AnnotationParseError {
                message: format!("Invalid numeric part in memory value: {numeric_part}"),
            })?;

    let multiplier = match unit.to_uppercase().as_str() {
        "B" | "" => 1,
        "K" | "KI" => 1024,
        "M" | "MI" => 1024 * 1024,
        "G" | "GI" => 1024 * 1024 * 1024,
        "T" | "TI" => 1024_u64.pow(4),
        _ => {
            return Err(Report::new(KubernetesError::AnnotationParseError {
                message: format!("Unsupported memory unit: {unit}"),
            }));
        }
    };

    Ok((numeric_value * multiplier as f64) as u64)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn parse_memory_value_bytes() {
        assert_eq!(parse_memory_value("1024").unwrap(), 1024);
        assert_eq!(parse_memory_value("1024B").unwrap(), 1024);
    }

    #[test]
    fn parse_memory_value_with_units() {
        assert_eq!(parse_memory_value("1Ki").unwrap(), 1024);
        assert_eq!(parse_memory_value("1Mi").unwrap(), 1024 * 1024);
        assert_eq!(parse_memory_value("2Gi").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_value("1Ti").unwrap(), 1024_u64.pow(4));
    }

    #[test]
    fn parse_memory_value_case_insensitive() {
        assert_eq!(parse_memory_value("1gi").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_memory_value("1GI").unwrap(), 1024 * 1024 * 1024);
    }

    #[test]
    fn parse_tflops_value_raw() {
        assert_eq!(parse_tflops_value("1000").unwrap(), 1000.0);
        assert_eq!(parse_tflops_value("1000.5").unwrap(), 1000.5);
    }

    #[test]
    fn parse_tflops_value_with_units() {
        assert_eq!(parse_tflops_value("1k").unwrap(), 1000.0);
        assert_eq!(parse_tflops_value("1K").unwrap(), 1000.0);
        assert_eq!(parse_tflops_value("1m").unwrap(), 0.001);
        assert_eq!(parse_tflops_value("1M").unwrap(), 1_000_000.0);
        assert_eq!(parse_tflops_value("1G").unwrap(), 1_000_000_000.0);
    }

    #[test]
    fn parse_tflops_value_case_insensitive() {
        assert_eq!(parse_tflops_value("1k").unwrap(), 1000.0);
        assert_eq!(parse_tflops_value("1K").unwrap(), 1000.0);
        assert_eq!(parse_tflops_value("1m").unwrap(), 0.001);
        assert_eq!(parse_tflops_value("1M").unwrap(), 1_000_000.0);
        assert_eq!(parse_tflops_value("1G").unwrap(), 1_000_000_000.0);
    }

    #[test]
    fn parse_tflops_value_error_case_71200m() {
        // Test the specific case that was failing in the error report
        assert_eq!(parse_tflops_value("71200m").unwrap(), 71.2);
    }

    #[test]
    fn from_pod_annotations_with_tflops_units() {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/tflops-request".to_string(),
            "71200m".to_string(), // This was causing the original error
        );
        annotations.insert(
            "tensor-fusion.ai/tflops-limit".to_string(),
            "5k".to_string(),
        );

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();

        assert!(result.has_annotations());
        assert_eq!(result.0.tflops_request, Some(71.2));
        assert_eq!(result.0.tflops_limit, Some(5000.0));
    }

    #[test]
    fn from_pod_annotations_empty() {
        let annotations = BTreeMap::new();
        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();
        assert!(!result.has_annotations());
    }

    #[test]
    fn from_pod_annotations_with_values() {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/tflops-request".to_string(),
            "10.5".to_string(),
        );
        annotations.insert(
            "tensor-fusion.ai/vram-request".to_string(),
            "2Gi".to_string(),
        );
        annotations.insert(
            "tensor-fusion.ai/tflops-limit".to_string(),
            "20.0".to_string(),
        );
        annotations.insert("tensor-fusion.ai/vram-limit".to_string(), "4Gi".to_string());

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();

        assert!(result.has_annotations());
        assert_eq!(result.0.tflops_request, Some(10.5));
        assert_eq!(result.0.vram_request, Some(2 * 1024 * 1024 * 1024));
        assert_eq!(result.0.tflops_limit, Some(20.0));
        assert_eq!(result.0.vram_limit, Some(4 * 1024 * 1024 * 1024));
    }

    #[test]
    fn from_pod_annotations_ignores_other_annotations() {
        let mut annotations = BTreeMap::new();
        annotations.insert("other.domain/annotation".to_string(), "value".to_string());
        annotations.insert(
            "tensor-fusion.ai/tflops-request".to_string(),
            "5.0".to_string(),
        );

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();

        assert!(result.has_annotations());
        assert_eq!(result.0.tflops_request, Some(5.0));
        assert_eq!(result.0.vram_request, None);
    }

    #[test]
    fn from_pod_annotations_with_gpu_uuids_single() {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/gpu-ids".to_string(),
            "GPU-12345678-1234-1234-1234-123456789abc".to_string(),
        );

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();

        assert!(result.has_annotations());
        assert_eq!(
            result.0.gpu_uuids,
            Some(vec!["GPU-12345678-1234-1234-1234-123456789abc".to_string()])
        );
    }

    #[test]
    fn from_pod_annotations_with_gpu_uuids_multiple() {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/gpu-ids".to_string(),
            "GPU-12345678-1234-1234-1234-123456789abc,GPU-87654321-4321-4321-4321-cba987654321"
                .to_string(),
        );

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();

        assert!(result.has_annotations());
        assert_eq!(
            result.0.gpu_uuids,
            Some(vec![
                "GPU-12345678-1234-1234-1234-123456789abc".to_string(),
                "GPU-87654321-4321-4321-4321-cba987654321".to_string()
            ])
        );
    }

    #[test]
    fn from_pod_annotations_with_gpu_uuids_empty() {
        let mut annotations = BTreeMap::new();
        annotations.insert("tensor-fusion.ai/gpu-ids".to_string(), "".to_string());

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();

        assert!(result.has_annotations());
        assert_eq!(result.0.gpu_uuids, Some(vec!["".to_string()]));
    }

    #[test]
    fn from_pod_annotations_comprehensive_with_gpu_uuids() {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/tflops-request".to_string(),
            "10.5".to_string(),
        );
        annotations.insert(
            "tensor-fusion.ai/vram-request".to_string(),
            "2Gi".to_string(),
        );
        annotations.insert(
            "tensor-fusion.ai/tflops-limit".to_string(),
            "20.0".to_string(),
        );
        annotations.insert("tensor-fusion.ai/vram-limit".to_string(), "4Gi".to_string());
        annotations.insert(
            "tensor-fusion.ai/gpu-ids".to_string(),
            "GPU-11111111-1111-1111-1111-111111111111,GPU-22222222-2222-2222-2222-222222222222"
                .to_string(),
        );

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();

        assert!(result.has_annotations());
        assert_eq!(result.0.tflops_request, Some(10.5));
        assert_eq!(result.0.vram_request, Some(2 * 1024 * 1024 * 1024));
        assert_eq!(result.0.tflops_limit, Some(20.0));
        assert_eq!(result.0.vram_limit, Some(4 * 1024 * 1024 * 1024));
        assert_eq!(
            result.0.gpu_uuids,
            Some(vec![
                "GPU-11111111-1111-1111-1111-111111111111".to_string(),
                "GPU-22222222-2222-2222-2222-222222222222".to_string()
            ])
        );
    }

    #[test]
    fn from_pod_annotations_with_container_gpus_single_container() {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/container-gpus".to_string(),
            r#"{"container1": ["GPU-aaaa", "GPU-bbbb"]}"#.to_string(),
        );

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();

        assert!(result.has_annotations());
        assert!(result.0.container_gpu_uuids.is_some());

        let container_gpus = result.0.container_gpu_uuids.unwrap();
        assert_eq!(container_gpus.len(), 1);
        assert_eq!(
            container_gpus.get("container1"),
            Some(&vec!["GPU-aaaa".to_string(), "GPU-bbbb".to_string()])
        );
    }

    #[test]
    fn from_pod_annotations_with_container_gpus_multiple_containers() {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/container-gpus".to_string(),
            r#"{"container1": ["GPU-1111"], "container2": ["GPU-2222", "GPU-3333"]}"#.to_string(),
        );

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();

        assert!(result.has_annotations());
        assert!(result.0.container_gpu_uuids.is_some());

        let container_gpus = result.0.container_gpu_uuids.unwrap();
        assert_eq!(container_gpus.len(), 2);
        assert_eq!(
            container_gpus.get("container1"),
            Some(&vec!["GPU-1111".to_string()])
        );
        assert_eq!(
            container_gpus.get("container2"),
            Some(&vec!["GPU-2222".to_string(), "GPU-3333".to_string()])
        );
    }

    #[test]
    fn from_pod_annotations_with_container_gpus_trim_whitespace() {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/container-gpus".to_string(),
            r#"{"  container1  ": ["  GPU-aaaa  ", "GPU-bbbb"]}"#.to_string(),
        );

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();

        assert!(result.has_annotations());
        let container_gpus = result.0.container_gpu_uuids.unwrap();
        assert_eq!(
            container_gpus.get("container1"),
            Some(&vec!["GPU-aaaa".to_string(), "GPU-bbbb".to_string()])
        );
    }

    #[test]
    fn from_pod_annotations_with_container_gpus_filter_empty() {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/container-gpus".to_string(),
            r#"{"container1": ["GPU-aaaa", "", "GPU-bbbb"], "container2": []}"#.to_string(),
        );

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();

        assert!(result.has_annotations());
        let container_gpus = result.0.container_gpu_uuids.unwrap();

        assert_eq!(container_gpus.len(), 1);
        assert_eq!(
            container_gpus.get("container1"),
            Some(&vec!["GPU-aaaa".to_string(), "GPU-bbbb".to_string()])
        );
        assert_eq!(container_gpus.get("container2"), None);
    }

    #[test]
    fn from_pod_annotations_with_container_gpus_invalid_json() {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/container-gpus".to_string(),
            "not a valid json".to_string(),
        );

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new());

        assert!(result.is_err());
    }

    #[test]
    fn from_pod_annotations_with_both_gpu_ids_and_container_gpus() {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/gpu-ids".to_string(),
            "GPU-default-1,GPU-default-2".to_string(),
        );
        annotations.insert(
            "tensor-fusion.ai/container-gpus".to_string(),
            r#"{"container1": ["GPU-container-1"]}"#.to_string(),
        );

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();

        assert!(result.has_annotations());
        assert_eq!(
            result.0.gpu_uuids,
            Some(vec![
                "GPU-default-1".to_string(),
                "GPU-default-2".to_string()
            ])
        );
        assert!(result.0.container_gpu_uuids.is_some());
        let container_gpus = result.0.container_gpu_uuids.unwrap();
        assert_eq!(
            container_gpus.get("container1"),
            Some(&vec!["GPU-container-1".to_string()])
        );
    }

    #[test]
    fn from_pod_annotations_with_container_gpus_empty_after_cleaning() {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/container-gpus".to_string(),
            r#"{"container1": ["", "  "]}"#.to_string(),
        );

        let result =
            TensorFusionPodInfo::from_pod_annotations_labels(&annotations, &BTreeMap::new())
                .unwrap();

        assert!(!result.has_annotations());
        assert_eq!(result.0.container_gpu_uuids, None);
    }
}
