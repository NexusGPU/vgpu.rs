//! Pod management metrics integration

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::domain::pod_management::{PodManager, registry::PodEntry};
use super::encoders::MetricsEncoder;

/// Resource usage metrics for a pod
#[derive(Debug, Clone)]
pub struct PodMetrics {
    pub pod_identifier: String,
    pub namespace: String,
    pub pod_name: String,
    pub timestamp: SystemTime,
    pub worker_count: usize,
    pub healthy_workers: usize,
    pub gpu_devices: Vec<GpuDeviceUsage>,
}

/// GPU device usage for a specific pod
#[derive(Debug, Clone)]
pub struct GpuDeviceUsage {
    pub device_idx: u32,
    pub device_uuid: String,
    pub cuda_usage: f32,
    pub memory_usage: u64,
    pub active_processes: u32,
}


/// Configuration for pod metrics monitoring
#[derive(Debug, Clone)]
pub struct PodMetricsConfig {
    /// Monitoring interval
    pub check_interval: Duration,
}

impl Default for PodMetricsConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(10),
        }
    }
}

/// Pod metrics collector that integrates with existing metrics infrastructure
pub struct PodMetricsCollector {
    config: PodMetricsConfig,
    encoder: Box<dyn MetricsEncoder + Send + Sync>,
    violations: Arc<RwLock<HashMap<String, Vec<ResourceViolation>>>>,
}

impl PodMetricsCollector {
    /// Create a new pod metrics collector
    pub fn new(encoder: Box<dyn MetricsEncoder + Send + Sync>) -> Self {
        Self {
            config: PodMetricsConfig::default(),
            encoder,
            violations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        encoder: Box<dyn MetricsEncoder + Send + Sync>,
        config: PodMetricsConfig,
    ) -> Self {
        Self {
            config,
            encoder,
            violations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Collect and output metrics for all pods
    pub async fn collect_pod_metrics(&self, pod_manager: &PodManager) {
        let registry = pod_manager.registry().read().await;
        let pod_count = registry.len();
        
        // Output overall pod count metric
        let timestamp = current_time();
        let pod_count_metric = format!(
            r#"{{"metric_type":"pod_count","count":{},"timestamp":{}}}"#,
            pod_count, timestamp
        );
        
        tracing::info!(
            target: "metrics",
            msg = %pod_count_metric,
        );

        for (pod_identifier, pod_entry) in registry.iter() {
            if let Some(metrics) = self.collect_pod_entry_metrics(pod_identifier, pod_entry).await {
                let metrics_str = self.encode_pod_metrics(&metrics);
                tracing::info!(
                    target: "metrics",
                    msg = %metrics_str,
                );

                // Check for violations
                if let Some(violations) = self.check_violations(&metrics).await {
                    for violation in violations {
                        let violation_str = self.encode_violation(&violation);
                        tracing::warn!(
                            target: "metrics",
                            msg = %violation_str,
                        );
                    }
                }
            }
        }
    }

    /// Collect metrics for a single pod entry
    async fn collect_pod_entry_metrics(
        &self,
        pod_identifier: &str,
        pod_entry: &PodEntry,
    ) -> Option<PodMetrics> {
        let info = &pod_entry.info;
        
        // Extract namespace and pod name from identifier
        let (namespace, pod_name) = extract_namespace_and_name_from_identifier(pod_identifier);
        
        // TODO: Integrate with actual resource collection from shared memory or process monitoring
        // For now, return a basic metrics structure
        Some(PodMetrics {
            pod_identifier: pod_identifier.to_string(),
            namespace,
            pod_name,
            timestamp: SystemTime::now(),
            cpu_usage: 0.0, // TODO: Get from actual monitoring
            memory_usage: 0, // TODO: Get from actual monitoring
            memory_limit: info.vram_limit.unwrap_or(0),
            worker_count: pod_entry.containers.len(),
            healthy_workers: pod_entry.containers.len(), // TODO: Check actual health
            gpu_devices: vec![], // TODO: Get from actual GPU monitoring
        })
    }

    /// Check for resource violations
    async fn check_violations(&self, metrics: &PodMetrics) -> Option<Vec<ResourceViolation>> {
        let mut violations = Vec::new();

        // Check CPU usage
        if metrics.cpu_usage > self.config.cpu_threshold {
            violations.push(ResourceViolation {
                pod_identifier: metrics.pod_identifier.clone(),
                namespace: metrics.namespace.clone(),
                violation_type: ViolationType::CpuUsage,
                current_value: metrics.cpu_usage,
                threshold: self.config.cpu_threshold,
                timestamp: metrics.timestamp,
                count: 1, // TODO: Track actual violation count over time
            });
        }

        // Check memory usage
        if metrics.memory_limit > 0 {
            let memory_percentage = (metrics.memory_usage as f64 / metrics.memory_limit as f64) * 100.0;
            if memory_percentage > self.config.memory_threshold {
                violations.push(ResourceViolation {
                    pod_identifier: metrics.pod_identifier.clone(),
                    namespace: metrics.namespace.clone(),
                    violation_type: ViolationType::MemoryUsage,
                    current_value: memory_percentage,
                    threshold: self.config.memory_threshold,
                    timestamp: metrics.timestamp,
                    count: 1,
                });
            }
        }

        // Check GPU usage
        for gpu_device in &metrics.gpu_devices {
            if gpu_device.cuda_usage > self.config.gpu_threshold as f32 {
                violations.push(ResourceViolation {
                    pod_identifier: metrics.pod_identifier.clone(),
                    namespace: metrics.namespace.clone(),
                    violation_type: ViolationType::GpuUsage {
                        device_idx: gpu_device.device_idx,
                    },
                    current_value: gpu_device.cuda_usage as f64,
                    threshold: self.config.gpu_threshold,
                    timestamp: metrics.timestamp,
                    count: 1,
                });
            }
        }

        if violations.is_empty() {
            None
        } else {
            Some(violations)
        }
    }

    /// Encode pod metrics to string format
    fn encode_pod_metrics(&self, metrics: &PodMetrics) -> String {
        // Use JSON format for now, can be extended to support other formats
        let mut gpu_devices_json = String::new();
        gpu_devices_json.push('[');
        for (idx, device) in metrics.gpu_devices.iter().enumerate() {
            if idx > 0 {
                gpu_devices_json.push(',');
            }
            gpu_devices_json.push_str(&format!(
                r#"{{"device_idx":{},"device_uuid":"{}","cuda_usage":{},"memory_usage":{},"active_processes":{}}}"#,
                device.device_idx, device.device_uuid, device.cuda_usage, device.memory_usage, device.active_processes
            ));
        }
        gpu_devices_json.push(']');

        format!(
            r#"{{"metric_type":"pod","pod_identifier":"{}","namespace":"{}","pod_name":"{}","timestamp":{},"cpu_usage":{},"memory_usage":{},"memory_limit":{},"worker_count":{},"healthy_workers":{},"gpu_devices":{}}}"#,
            metrics.pod_identifier,
            metrics.namespace,
            metrics.pod_name,
            metrics.timestamp.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.memory_limit,
            metrics.worker_count,
            metrics.healthy_workers,
            gpu_devices_json
        )
    }

    /// Encode resource violation to string format
    fn encode_violation(&self, violation: &ResourceViolation) -> String {
        let message = format!(
            "{:?} violation: current={}, threshold={}, count={}",
            violation.violation_type, violation.current_value, violation.threshold, violation.count
        );
        
        format!(
            r#"{{"metric_type":"pod_violation","pod_identifier":"{}","namespace":"{}","violation_type":"{}","message":"{}","timestamp":{}}}"#,
            violation.pod_identifier,
            violation.namespace,
            format!("{:?}", violation.violation_type),
            message,
            violation.timestamp.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
        )
    }
}

/// Start pod metrics collection task
pub async fn run_pod_metrics_collection(
    pod_manager: Arc<PodManager>,
    encoder: Box<dyn MetricsEncoder + Send + Sync>,
    config: PodMetricsConfig,
    cancellation_token: CancellationToken,
) {
    let collector = PodMetricsCollector::with_config(encoder, config.clone());
    let mut interval = tokio::time::interval(config.check_interval);

    info!("Starting pod metrics collection with interval: {:?}", config.check_interval);

    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                info!("Pod metrics collection shutdown requested");
                break;
            }
            _ = interval.tick() => {
                collector.collect_pod_metrics(&pod_manager).await;
            }
        }
    }

    info!("Pod metrics collection stopped");
}

/// Get current timestamp in milliseconds
pub fn current_time() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

/// Extract namespace and pod name from pod identifier
fn extract_namespace_and_name_from_identifier(pod_identifier: &str) -> (String, String) {
    // Pod identifier format is typically "namespace/pod_name" or custom format
    // Adjust this parsing logic based on actual format used by PodManager
    if let Some(pos) = pod_identifier.find('/') {
        let namespace = pod_identifier[..pos].to_string();
        let pod_name = pod_identifier[pos + 1..].to_string();
        (namespace, pod_name)
    } else {
        // Fallback for simple identifiers
        ("default".to_string(), pod_identifier.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::metrics::encoders::json::JsonEncoder;

    #[test]
    fn test_pod_metrics_encoding() {
        let encoder = JsonEncoder::new();
        let collector = PodMetricsCollector::new(Box::new(encoder));
        
        let metrics = PodMetrics {
            pod_identifier: "test-namespace/test-pod".to_string(),
            namespace: "test-namespace".to_string(),
            pod_name: "test-pod".to_string(),
            timestamp: SystemTime::now(),
            cpu_usage: 50.0,
            memory_usage: 1024 * 1024 * 1024, // 1GB
            memory_limit: 2 * 1024 * 1024 * 1024, // 2GB
            worker_count: 2,
            healthy_workers: 2,
            gpu_devices: vec![],
        };
        
        let result = collector.encode_pod_metrics(&metrics);
        assert!(result.contains("metric_type\":\"pod"));
        assert!(result.contains("test-namespace"));
        assert!(result.contains("test-pod"));
    }

    #[test]
    fn test_extract_namespace_and_name() {
        let (namespace, pod_name) = extract_namespace_and_name_from_identifier("test-ns/test-pod");
        assert_eq!(namespace, "test-ns");
        assert_eq!(pod_name, "test-pod");
        
        let (namespace, pod_name) = extract_namespace_and_name_from_identifier("simple-pod");
        assert_eq!(namespace, "default");
        assert_eq!(pod_name, "simple-pod");
    }
} 