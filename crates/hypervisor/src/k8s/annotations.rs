use std::collections::BTreeMap;

use error_stack::Report;
use error_stack::ResultExt;

use crate::k8s::types::KubernetesError;

/// Domain prefix for tensor-fusion annotations.
const TENSOR_FUSION_DOMAIN: &str = "tensor-fusion.ai";

/// Tensor-fusion specific annotations extracted from Kubernetes pods.
#[derive(Debug, Clone, Default)]
pub(crate) struct TensorFusionAnnotations {
    /// Requested TFLOPS for the workload
    pub tflops_request: Option<f64>,
    /// Requested VRAM in bytes for the workload
    pub vram_request: Option<u64>,
    /// Maximum TFLOPS limit for the workload
    pub tflops_limit: Option<f64>,
    /// Maximum VRAM limit in bytes for the workload
    pub vram_limit: Option<u64>,
    /// GPU UUIDs
    pub gpu_uuids: Option<Vec<String>>,
}

impl TensorFusionAnnotations {
    /// Parse tensor-fusion annotations from a Kubernetes pod's annotations.
    ///
    /// Extracts and validates tensor-fusion specific annotations from the provided
    /// annotation map. Only processes annotations with the tensor-fusion.ai domain.
    ///
    /// # Errors
    ///
    /// - [`KubernetesError::AnnotationParseError`] if annotation values are invalid
    pub(crate) fn from_pod_annotations(
        annotations: &BTreeMap<String, String>,
    ) -> Result<Self, Report<KubernetesError>> {
        let mut result = Self::default();

        // Parse TFLOPS request
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/tflops-request")) {
            result.tflops_request = Some(value.parse::<f64>().change_context(
                KubernetesError::AnnotationParseError {
                    message: format!("Invalid tflops-request value: {value}"),
                },
            )?);
        }

        // Parse VRAM request
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/vram-request")) {
            result.vram_request = Some(parse_memory_value(value)?);
        }

        // Parse TFLOPS limit
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/tflops-limit")) {
            result.tflops_limit = Some(value.parse::<f64>().change_context(
                KubernetesError::AnnotationParseError {
                    message: format!("Invalid tflops-limit value: {value}"),
                },
            )?);
        }

        // Parse VRAM limit
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/vram-limit")) {
            result.vram_limit = Some(parse_memory_value(value)?);
        }

        // Parse GPU UUIDs
        if let Some(value) = annotations.get(&format!("{TENSOR_FUSION_DOMAIN}/gpu-ids")) {
            result.gpu_uuids = Some(value.split(',').map(|s| s.to_string()).collect());
        }

        Ok(result)
    }

    /// Check if any tensor-fusion annotations are present.
    pub(crate) const fn has_annotations(&self) -> bool {
        self.tflops_request.is_some()
            || self.vram_request.is_some()
            || self.tflops_limit.is_some()
            || self.vram_limit.is_some()
            || self.gpu_uuids.is_some()
    }
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
    fn from_pod_annotations_empty() {
        let annotations = BTreeMap::new();
        let result = TensorFusionAnnotations::from_pod_annotations(&annotations).unwrap();
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

        let result = TensorFusionAnnotations::from_pod_annotations(&annotations).unwrap();

        assert!(result.has_annotations());
        assert_eq!(result.tflops_request, Some(10.5));
        assert_eq!(result.vram_request, Some(2 * 1024 * 1024 * 1024));
        assert_eq!(result.tflops_limit, Some(20.0));
        assert_eq!(result.vram_limit, Some(4 * 1024 * 1024 * 1024));
    }

    #[test]
    fn from_pod_annotations_ignores_other_annotations() {
        let mut annotations = BTreeMap::new();
        annotations.insert("other.domain/annotation".to_string(), "value".to_string());
        annotations.insert(
            "tensor-fusion.ai/tflops-request".to_string(),
            "5.0".to_string(),
        );

        let result = TensorFusionAnnotations::from_pod_annotations(&annotations).unwrap();

        assert!(result.has_annotations());
        assert_eq!(result.tflops_request, Some(5.0));
        assert_eq!(result.vram_request, None);
    }

    #[test]
    fn from_pod_annotations_with_gpu_uuids_single() {
        let mut annotations = BTreeMap::new();
        annotations.insert(
            "tensor-fusion.ai/gpu-ids".to_string(),
            "GPU-12345678-1234-1234-1234-123456789abc".to_string(),
        );

        let result = TensorFusionAnnotations::from_pod_annotations(&annotations).unwrap();

        assert!(result.has_annotations());
        assert_eq!(
            result.gpu_uuids,
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

        let result = TensorFusionAnnotations::from_pod_annotations(&annotations).unwrap();

        assert!(result.has_annotations());
        assert_eq!(
            result.gpu_uuids,
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

        let result = TensorFusionAnnotations::from_pod_annotations(&annotations).unwrap();

        assert!(result.has_annotations());
        assert_eq!(result.gpu_uuids, Some(vec!["".to_string()]));
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

        let result = TensorFusionAnnotations::from_pod_annotations(&annotations).unwrap();

        assert!(result.has_annotations());
        assert_eq!(result.tflops_request, Some(10.5));
        assert_eq!(result.vram_request, Some(2 * 1024 * 1024 * 1024));
        assert_eq!(result.tflops_limit, Some(20.0));
        assert_eq!(result.vram_limit, Some(4 * 1024 * 1024 * 1024));
        assert_eq!(
            result.gpu_uuids,
            Some(vec![
                "GPU-11111111-1111-1111-1111-111111111111".to_string(),
                "GPU-22222222-2222-2222-2222-222222222222".to_string()
            ])
        );
    }
}
