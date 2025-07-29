//! Prometheus metrics integration for pod management

use std::sync::Arc;
use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramVec,
    IntCounter, IntCounterVec, IntGauge, IntGaugeVec,
    Registry, Opts, HistogramOpts,
};
use lazy_static::lazy_static;

use super::resource_monitor::{MetricsCollector, PodMetrics, ResourceViolation, ViolationType};

lazy_static! {
    /// Global Prometheus registry for pod management metrics
    pub static ref REGISTRY: Registry = Registry::new();
    
    // Pod-level metrics
    static ref POD_COUNT: IntGauge = IntGauge::new(
        "pod_management_pods_total",
        "Total number of managed pods"
    ).unwrap();
    
    static ref WORKER_COUNT: IntGaugeVec = IntGaugeVec::new(
        Opts::new("pod_management_workers_total", "Total number of workers per pod"),
        &["pod_id", "namespace"]
    ).unwrap();
    
    static ref HEALTHY_WORKER_COUNT: IntGaugeVec = IntGaugeVec::new(
        Opts::new("pod_management_healthy_workers", "Number of healthy workers per pod"),
        &["pod_id", "namespace"]
    ).unwrap();
    
    // Resource usage metrics
    static ref CPU_USAGE: GaugeVec = GaugeVec::new(
        Opts::new("pod_management_cpu_usage_percent", "CPU usage percentage per pod"),
        &["pod_id", "namespace"]
    ).unwrap();
    
    static ref MEMORY_USAGE_BYTES: GaugeVec = GaugeVec::new(
        Opts::new("pod_management_memory_usage_bytes", "Memory usage in bytes per pod"),
        &["pod_id", "namespace"]
    ).unwrap();
    
    static ref MEMORY_LIMIT_BYTES: GaugeVec = GaugeVec::new(
        Opts::new("pod_management_memory_limit_bytes", "Memory limit in bytes per pod"),
        &["pod_id", "namespace"]
    ).unwrap();
    
    // GPU metrics
    static ref GPU_USAGE: GaugeVec = GaugeVec::new(
        Opts::new("pod_management_gpu_usage_percent", "GPU usage percentage per device"),
        &["pod_id", "namespace", "device_idx", "device_uuid"]
    ).unwrap();
    
    static ref GPU_MEMORY_USAGE_BYTES: GaugeVec = GaugeVec::new(
        Opts::new("pod_management_gpu_memory_usage_bytes", "GPU memory usage in bytes"),
        &["pod_id", "namespace", "device_idx", "device_uuid"]
    ).unwrap();
    
    static ref GPU_ACTIVE_PROCESSES: IntGaugeVec = IntGaugeVec::new(
        Opts::new("pod_management_gpu_active_processes", "Number of active processes on GPU"),
        &["pod_id", "namespace", "device_idx", "device_uuid"]
    ).unwrap();
    
    // Violation metrics
    static ref RESOURCE_VIOLATIONS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("pod_management_resource_violations_total", "Total number of resource violations"),
        &["pod_id", "namespace", "violation_type"]
    ).unwrap();
    
    static ref VIOLATION_DURATION_SECONDS: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "pod_management_violation_duration_seconds",
            "Duration of resource violations"
        ),
        &["pod_id", "namespace", "violation_type"]
    ).unwrap();
    
    // Operational metrics
    static ref MONITORING_CYCLES_TOTAL: IntCounter = IntCounter::new(
        "pod_management_monitoring_cycles_total",
        "Total number of monitoring cycles completed"
    ).unwrap();
    
    static ref MONITORING_ERRORS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("pod_management_monitoring_errors_total", "Total number of monitoring errors"),
        &["error_type"]
    ).unwrap();
    
    static ref MONITORING_DURATION_SECONDS: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "pod_management_monitoring_duration_seconds",
            "Duration of monitoring cycles"
        )
    ).unwrap();
}

/// Initialize Prometheus metrics for pod management
pub fn init_metrics() -> Result<(), prometheus::Error> {
    // Register all metrics
    REGISTRY.register(Box::new(POD_COUNT.clone()))?;
    REGISTRY.register(Box::new(WORKER_COUNT.clone()))?;
    REGISTRY.register(Box::new(HEALTHY_WORKER_COUNT.clone()))?;
    REGISTRY.register(Box::new(CPU_USAGE.clone()))?;
    REGISTRY.register(Box::new(MEMORY_USAGE_BYTES.clone()))?;
    REGISTRY.register(Box::new(MEMORY_LIMIT_BYTES.clone()))?;
    REGISTRY.register(Box::new(GPU_USAGE.clone()))?;
    REGISTRY.register(Box::new(GPU_MEMORY_USAGE_BYTES.clone()))?;
    REGISTRY.register(Box::new(GPU_ACTIVE_PROCESSES.clone()))?;
    REGISTRY.register(Box::new(RESOURCE_VIOLATIONS_TOTAL.clone()))?;
    REGISTRY.register(Box::new(VIOLATION_DURATION_SECONDS.clone()))?;
    REGISTRY.register(Box::new(MONITORING_CYCLES_TOTAL.clone()))?;
    REGISTRY.register(Box::new(MONITORING_ERRORS_TOTAL.clone()))?;
    REGISTRY.register(Box::new(MONITORING_DURATION_SECONDS.clone()))?;
    
    Ok(())
}

/// Prometheus metrics collector that implements the MetricsCollector trait
#[derive(Debug)]
pub struct PrometheusMetricsCollector;

impl PrometheusMetricsCollector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PrometheusMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector for PrometheusMetricsCollector {
    fn collect_metrics(&self, metrics: &PodMetrics) {
        let namespace = extract_namespace_from_pod_id(&metrics.pod_id);
        let pod_name = extract_pod_name_from_pod_id(&metrics.pod_id);
        
        // Update pod-level metrics
        WORKER_COUNT
            .with_label_values(&[&metrics.pod_id.to_string(), &namespace])
            .set(metrics.worker_count as i64);
        
        HEALTHY_WORKER_COUNT
            .with_label_values(&[&metrics.pod_id.to_string(), &namespace])
            .set(metrics.healthy_workers as i64);
        
        CPU_USAGE
            .with_label_values(&[&metrics.pod_id.to_string(), &namespace])
            .set(metrics.cpu_usage);
        
        MEMORY_USAGE_BYTES
            .with_label_values(&[&metrics.pod_id.to_string(), &namespace])
            .set(metrics.memory_usage as f64);
        
        MEMORY_LIMIT_BYTES
            .with_label_values(&[&metrics.pod_id.to_string(), &namespace])
            .set(metrics.memory_limit as f64);
        
        // Update GPU metrics
        for (device_idx, device_usage) in &metrics.device_metrics {
            let device_idx_str = device_idx.to_string();
            
            GPU_USAGE
                .with_label_values(&[
                    &metrics.pod_id.to_string(),
                    &namespace,
                    &device_idx_str,
                    &device_usage.device_uuid,
                ])
                .set(device_usage.cuda_usage as f64);
            
            GPU_MEMORY_USAGE_BYTES
                .with_label_values(&[
                    &metrics.pod_id.to_string(),
                    &namespace,
                    &device_idx_str,
                    &device_usage.device_uuid,
                ])
                .set(device_usage.memory_usage as f64);
            
            GPU_ACTIVE_PROCESSES
                .with_label_values(&[
                    &metrics.pod_id.to_string(),
                    &namespace,
                    &device_idx_str,
                    &device_usage.device_uuid,
                ])
                .set(device_usage.active_processes as i64);
        }
        
        // Increment monitoring cycle counter
        MONITORING_CYCLES_TOTAL.inc();
    }
    
    fn report_violation(&self, violation: &ResourceViolation) {
        let namespace = extract_namespace_from_pod_id(&violation.pod_id);
        let violation_type = format!("{:?}", violation.violation_type);
        
        // Increment violation counter
        RESOURCE_VIOLATIONS_TOTAL
            .with_label_values(&[
                &violation.pod_id.to_string(),
                &namespace,
                &violation_type,
            ])
            .inc();
        
        // Record violation duration (simplified - would need to track start/end times)
        let duration = std::time::SystemTime::now()
            .duration_since(violation.timestamp)
            .unwrap_or_default()
            .as_secs_f64();
        
        VIOLATION_DURATION_SECONDS
            .with_label_values(&[
                &violation.pod_id.to_string(),
                &namespace,
                &violation_type,
            ])
            .observe(duration);
    }
}

/// Update global pod count metric
pub fn update_pod_count(count: i64) {
    POD_COUNT.set(count);
}

/// Record monitoring error
pub fn record_monitoring_error(error_type: &str) {
    MONITORING_ERRORS_TOTAL
        .with_label_values(&[error_type])
        .inc();
}

/// Record monitoring duration
pub fn record_monitoring_duration(duration: std::time::Duration) {
    MONITORING_DURATION_SECONDS.observe(duration.as_secs_f64());
}

/// Get metrics as Prometheus exposition format
pub fn get_metrics() -> Result<String, prometheus::Error> {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let metric_families = REGISTRY.gather();
    encoder.encode_to_string(&metric_families)
}

// Helper functions

/// Extract namespace from PodId
fn extract_namespace_from_pod_id(pod_id: &crate::domain::pod_management::types::PodId) -> String {
    let parts: Vec<&str> = pod_id.as_str().split('_').collect();
    if parts.len() >= 3 && parts[0] == "tf" && parts[1] == "shm" {
        parts[2].to_string()
    } else {
        "unknown".to_string()
    }
}

/// Extract pod name from PodId
fn extract_pod_name_from_pod_id(pod_id: &crate::domain::pod_management::types::PodId) -> String {
    let parts: Vec<&str> = pod_id.as_str().split('_').collect();
    if parts.len() >= 4 && parts[0] == "tf" && parts[1] == "shm" {
        parts[3..].join("_")
    } else {
        "unknown".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::pod_management::types::{PodId, DeviceUsage};
    use std::collections::HashMap;
    
    #[test]
    fn test_prometheus_metrics_collector() {
        let collector = PrometheusMetricsCollector::new();
        
        let pod_id = PodId::new("test-namespace", "test-pod");
        let mut device_metrics = HashMap::new();
        device_metrics.insert(0, DeviceUsage::new(0, "test-uuid".to_string()));
        
        let metrics = PodMetrics {
            pod_id,
            timestamp: std::time::SystemTime::now(),
            cpu_usage: 50.0,
            memory_usage: 1024 * 1024 * 1024, // 1GB
            memory_limit: 2 * 1024 * 1024 * 1024, // 2GB
            device_metrics,
            worker_count: 2,
            healthy_workers: 2,
        };
        
        collector.collect_metrics(&metrics);
        
        // Verify metrics were recorded (this would need a proper test setup)
        assert!(WORKER_COUNT.with_label_values(&["tf_shm_test-namespace_test-pod", "test-namespace"]).get() == 2);
    }
    
    #[test]
    fn test_extract_namespace() {
        let pod_id = PodId::new("test-namespace", "test-pod");
        let namespace = extract_namespace_from_pod_id(&pod_id);
        assert_eq!(namespace, "test-namespace");
    }
    
    #[test]
    fn test_extract_pod_name() {
        let pod_id = PodId::new("test-namespace", "test-pod");
        let pod_name = extract_pod_name_from_pod_id(&pod_id);
        assert_eq!(pod_name, "test-pod");
    }
}