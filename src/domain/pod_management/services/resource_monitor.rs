//! Resource monitoring service with comprehensive metrics collection

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio::time::interval;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::domain::pod_management::{
    core::{registry::PodRegistry, error::{PodManagementError, Result}},
    types::{PodId, DeviceUsage, WorkerStatus},
    services::DeviceService,
};

/// Resource monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Monitoring interval
    pub check_interval: Duration,
    /// CPU usage threshold for alerts (percentage)
    pub cpu_threshold: f64,
    /// Memory usage threshold for alerts (percentage)
    pub memory_threshold: f64,
    /// GPU usage threshold for alerts (percentage)
    pub gpu_threshold: f64,
    /// Maximum number of violations before marking as unhealthy
    pub max_violations: u32,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(5),
            cpu_threshold: 95.0,
            memory_threshold: 90.0,
            gpu_threshold: 95.0,
            max_violations: 3,
        }
    }
}

/// Resource usage metrics for a pod
#[derive(Debug, Clone)]
pub struct PodMetrics {
    pub pod_id: PodId,
    pub timestamp: SystemTime,
    pub cpu_usage: f64,
    pub memory_usage: u64,
    pub memory_limit: u64,
    pub device_metrics: HashMap<u32, DeviceUsage>,
    pub worker_count: usize,
    pub healthy_workers: usize,
}

/// Resource violation information
#[derive(Debug, Clone)]
pub struct ResourceViolation {
    pub pod_id: PodId,
    pub violation_type: ViolationType,
    pub current_value: f64,
    pub threshold: f64,
    pub timestamp: SystemTime,
    pub count: u32,
}

#[derive(Debug, Clone)]
pub enum ViolationType {
    CpuUsage,
    MemoryUsage,
    GpuUsage { device_idx: u32 },
    WorkerUnhealthy { worker_pid: u32 },
}

/// Metrics collection callback trait
pub trait MetricsCollector: Send + Sync {
    fn collect_metrics(&self, metrics: &PodMetrics);
    fn report_violation(&self, violation: &ResourceViolation);
}

/// Service for monitoring pod and worker resource usage
#[derive(Debug)]
pub struct ResourceMonitor {
    config: MonitorConfig,
    device_service: Arc<DeviceService>,
    metrics_collectors: Vec<Arc<dyn MetricsCollector>>,
    violations: Arc<RwLock<HashMap<PodId, Vec<ResourceViolation>>>>,
    monitoring_task: RwLock<Option<JoinHandle<()>>>,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new(device_service: Arc<DeviceService>) -> Self {
        Self {
            config: MonitorConfig::default(),
            device_service,
            metrics_collectors: Vec::new(),
            violations: Arc::new(RwLock::new(HashMap::new())),
            monitoring_task: RwLock::new(None),
        }
    }

    /// Create with custom configuration
    pub fn with_config(device_service: Arc<DeviceService>, config: MonitorConfig) -> Self {
        Self {
            config,
            device_service,
            metrics_collectors: Vec::new(),
            violations: Arc::new(RwLock::new(HashMap::new())),
            monitoring_task: RwLock::new(None),
        }
    }

    /// Add a metrics collector
    pub fn add_metrics_collector(&mut self, collector: Arc<dyn MetricsCollector>) {
        self.metrics_collectors.push(collector);
    }

    /// Start monitoring resources for all pods
    pub async fn start_monitoring(
        &self,
        registry: PodRegistry,
        cancellation_token: CancellationToken,
    ) {
        info!("Starting resource monitoring with interval {:?}", self.config.check_interval);

        let device_service = self.device_service.clone();
        let config = self.config.clone();
        let collectors = self.metrics_collectors.clone();
        let violations = self.violations.clone();

        let task = tokio::spawn(async move {
            Self::monitor_loop(
                registry,
                device_service,
                config,
                collectors,
                violations,
                cancellation_token,
            ).await;
        });

        let mut monitoring_task = self.monitoring_task.write().await;
        *monitoring_task = Some(task);
    }

    /// Stop monitoring
    pub async fn stop_monitoring(&self) {
        let mut monitoring_task = self.monitoring_task.write().await;
        if let Some(task) = monitoring_task.take() {
            task.abort();
            info!("Resource monitoring stopped");
        }
    }

    /// Get current violations for a pod
    pub async fn get_violations(&self, pod_id: &PodId) -> Vec<ResourceViolation> {
        let violations = self.violations.read().await;
        violations.get(pod_id).cloned().unwrap_or_default()
    }

    /// Get all violations
    pub async fn get_all_violations(&self) -> HashMap<PodId, Vec<ResourceViolation>> {
        let violations = self.violations.read().await;
        violations.clone()
    }

    /// Clear violations for a pod
    pub async fn clear_violations(&self, pod_id: &PodId) {
        let mut violations = self.violations.write().await;
        violations.remove(pod_id);
    }

    /// Get monitoring statistics
    pub async fn get_stats(&self) -> MonitoringStats {
        let violations = self.violations.read().await;
        let total_violations = violations.values().map(|v| v.len()).sum();
        let pods_with_violations = violations.len();

        MonitoringStats {
            monitored_pods: violations.keys().len(),
            total_violations,
            pods_with_violations,
            monitoring_active: self.monitoring_task.read().await.is_some(),
        }
    }

    // Private implementation

    /// Main monitoring loop
    async fn monitor_loop(
        registry: PodRegistry,
        device_service: Arc<DeviceService>,
        config: MonitorConfig,
        collectors: Vec<Arc<dyn MetricsCollector>>,
        violations: Arc<RwLock<HashMap<PodId, Vec<ResourceViolation>>>>,
        cancellation_token: CancellationToken,
    ) {
        let mut interval = interval(config.check_interval);
        
        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    info!("Resource monitoring cancelled");
                    break;
                }
                _ = interval.tick() => {
                    if let Err(e) = Self::check_all_pods(
                        &registry,
                        &device_service,
                        &config,
                        &collectors,
                        &violations,
                    ).await {
                        error!("Error during resource monitoring: {}", e);
                    }
                }
            }
        }
    }

    /// Check all pods for resource violations
    async fn check_all_pods(
        registry: &PodRegistry,
        device_service: &DeviceService,
        config: &MonitorConfig,
        collectors: &[Arc<dyn MetricsCollector>],
        violations: &Arc<RwLock<HashMap<PodId, Vec<ResourceViolation>>>>,
    ) -> Result<()> {
        let pods = registry.list_pods().await;
        debug!("Checking {} pods for resource violations", pods.len());

        for pod in pods {
            if let Err(e) = Self::check_pod_resources(
                &pod.id,
                &pod,
                device_service,
                config,
                collectors,
                violations,
            ).await {
                warn!("Failed to check resources for pod {}: {}", pod.id, e);
            }
        }

        Ok(())
    }

    /// Check resource usage for a specific pod
    async fn check_pod_resources(
        pod_id: &PodId,
        pod: &crate::domain::pod_management::types::Pod,
        device_service: &DeviceService,
        config: &MonitorConfig,
        collectors: &[Arc<dyn MetricsCollector>],
        violations: &Arc<RwLock<HashMap<PodId, Vec<ResourceViolation>>>>,
    ) -> Result<()> {
        let timestamp = SystemTime::now();

        // Collect device metrics
        let mut device_metrics = HashMap::new();
        let mut total_gpu_usage = 0.0;
        let mut gpu_device_count = 0;

        for device_idx in pod.device_allocation.device_indices() {
            match device_service.get_device_usage(device_idx).await {
                Ok(usage) => {
                    total_gpu_usage += usage.cuda_usage as f64;
                    gpu_device_count += 1;
                    device_metrics.insert(device_idx, usage);
                }
                Err(e) => {
                    warn!("Failed to get device usage for device {}: {}", device_idx, e);
                }
            }
        }

        // Calculate system metrics (simplified - in real implementation would use procfs)
        let cpu_usage = Self::get_pod_cpu_usage(pod).await?;
        let (memory_usage, memory_limit) = Self::get_pod_memory_usage(pod).await?;
        
        // Count healthy workers
        let worker_count = pod.containers.values()
            .map(|c| c.workers.len())
            .sum();
        let healthy_workers = worker_count; // Simplified - would check actual health

        let metrics = PodMetrics {
            pod_id: pod_id.clone(),
            timestamp,
            cpu_usage,
            memory_usage,
            memory_limit,
            device_metrics: device_metrics.clone(),
            worker_count,
            healthy_workers,
        };

        // Report metrics to collectors
        for collector in collectors {
            collector.collect_metrics(&metrics);
        }

        // Check for violations
        let mut new_violations = Vec::new();

        // Check CPU usage
        if cpu_usage > config.cpu_threshold {
            new_violations.push(ResourceViolation {
                pod_id: pod_id.clone(),
                violation_type: ViolationType::CpuUsage,
                current_value: cpu_usage,
                threshold: config.cpu_threshold,
                timestamp,
                count: 1,
            });
        }

        // Check memory usage
        let memory_usage_pct = (memory_usage as f64 / memory_limit as f64) * 100.0;
        if memory_usage_pct > config.memory_threshold {
            new_violations.push(ResourceViolation {
                pod_id: pod_id.clone(),
                violation_type: ViolationType::MemoryUsage,
                current_value: memory_usage_pct,
                threshold: config.memory_threshold,
                timestamp,
                count: 1,
            });
        }

        // Check GPU usage
        for (device_idx, usage) in &device_metrics {
            if usage.cuda_usage as f64 > config.gpu_threshold {
                new_violations.push(ResourceViolation {
                    pod_id: pod_id.clone(),
                    violation_type: ViolationType::GpuUsage { device_idx: *device_idx },
                    current_value: usage.cuda_usage as f64,
                    threshold: config.gpu_threshold,
                    timestamp,
                    count: 1,
                });
            }
        }

        // Update violations and report
        if !new_violations.is_empty() {
            let mut violations_map = violations.write().await;
            let pod_violations = violations_map.entry(pod_id.clone()).or_insert_with(Vec::new);
            
            for violation in new_violations {
                // Report violation to collectors
                for collector in collectors {
                    collector.report_violation(&violation);
                }
                
                pod_violations.push(violation);
            }

            // Keep only recent violations (last hour)
            let one_hour_ago = timestamp - Duration::from_secs(3600);
            pod_violations.retain(|v| v.timestamp > one_hour_ago);
        }

        Ok(())
    }

    /// Get CPU usage for a pod (simplified implementation)
    async fn get_pod_cpu_usage(
        _pod: &crate::domain::pod_management::types::Pod
    ) -> Result<f64> {
        // In a real implementation, this would read from /proc/*/stat for all processes
        // in the pod and calculate CPU usage percentage
        Ok(0.0) // Placeholder
    }

    /// Get memory usage for a pod (simplified implementation)
    async fn get_pod_memory_usage(
        pod: &crate::domain::pod_management::types::Pod
    ) -> Result<(u64, u64)> {
        // In a real implementation, this would read from /proc/*/status for all processes
        // and sum up memory usage
        let memory_limit = pod.device_allocation.configs
            .iter()
            .map(|c| c.mem_limit)
            .sum();
        
        Ok((0, memory_limit)) // Placeholder: (used, limit)
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        // This requires dependencies, so we can't provide a meaningful default
        panic!("ResourceMonitor requires dependencies and cannot be created with default()")
    }
}

/// Monitoring statistics
#[derive(Debug, Clone)]
pub struct MonitoringStats {
    pub monitored_pods: usize,
    pub total_violations: usize,
    pub pods_with_violations: usize,
    pub monitoring_active: bool,
}

/// Simple console metrics collector for development/debugging
#[derive(Debug)]
pub struct ConsoleMetricsCollector;

impl MetricsCollector for ConsoleMetricsCollector {
    fn collect_metrics(&self, metrics: &PodMetrics) {
        debug!(
            pod_id = %metrics.pod_id,
            cpu_usage = metrics.cpu_usage,
            memory_usage = metrics.memory_usage,
            memory_limit = metrics.memory_limit,
            workers = metrics.worker_count,
            "Pod metrics collected"
        );
    }

    fn report_violation(&self, violation: &ResourceViolation) {
        warn!(
            pod_id = %violation.pod_id,
            violation_type = ?violation.violation_type,
            current = violation.current_value,
            threshold = violation.threshold,
            "Resource violation detected"
        );
    }
}