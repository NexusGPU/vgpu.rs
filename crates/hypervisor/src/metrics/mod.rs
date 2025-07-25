use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use tokio_util::sync::CancellationToken;

use crate::config::GPU_CAPACITY_MAP;
use crate::gpu_observer::GpuObserver;
use crate::process::GpuResources;

use crate::pod_management::PodManager;

pub mod encoders;
use encoders::create_encoder;

// Wrapper struct for Vec<u8> that implements Display
pub struct BytesWrapper(Vec<u8>);

impl fmt::Display for BytesWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "");
        }

        // Format as UTF-8 string if valid, otherwise as hex
        match std::str::from_utf8(&self.0) {
            Ok(s) => write!(f, "{s}"),
            Err(_) => {
                tracing::error!(
                    target: "metrics",
                    msg = "Failed to convert bytes to string",
                );
                Err(fmt::Error)
            }
        }
    }
}

impl From<Vec<u8>> for BytesWrapper {
    fn from(bytes: Vec<u8>) -> Self {
        BytesWrapper(bytes)
    }
}

#[derive(Default)]
struct AccumulatedGpuMetrics {
    rx: f64,
    tx: f64,
    temperature: f64,
    memory_bytes: u64,
    compute_percentage: f64,
    compute_tflops: f64,
    count: usize,
}

#[derive(Default)]
struct AccumulatedWorkerMetrics {
    memory_bytes: u64,
    compute_percentage: f64,
    compute_tflops: f64,
    count: usize,
}

/// Run metrics collection asynchronously
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_metrics(
    gpu_observer: Arc<GpuObserver>,
    metrics_batch_size: usize,
    node_name: &str,
    gpu_pool: Option<&str>,
    pod_mgr: Arc<PodManager>,
    metrics_format: &str,
    metrics_extra_labels: Option<&str>,
    cancellation_token: CancellationToken,
) {
    let gpu_pool = gpu_pool.unwrap_or("unknown");
    let encoder = create_encoder(metrics_format);

    let mut gpu_acc: HashMap<String, AccumulatedGpuMetrics> = HashMap::new();

    // level 1 key is gpu_uuid, level 2 key is pod_ns/pod_name, value is GPU usage metrics
    let mut worker_acc: HashMap<String, HashMap<String, AccumulatedWorkerMetrics>> = HashMap::new();
    let mut counter = 0;

    let metrics_extra_labels: Vec<_> = metrics_extra_labels
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let has_dynamic_metrics_labels = !metrics_extra_labels.is_empty();

    let mut receiver = gpu_observer.subscribe();

    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                tracing::info!("Metrics collection shutdown requested");
                break;
            }
            recv_result = receiver.recv() => {
                match recv_result {
                    Some(()) => {
                        counter += 1;
                        // Accumulate GPU metrics
                        for (gpu_uuid, gpu) in gpu_observer
                            .metrics
                            .read()
                            .expect("poisoned")
                            .gpu_metrics
                            .iter()
                        {
                            let acc = gpu_acc.entry(gpu_uuid.clone()).or_default();
                            acc.rx += gpu.rx as f64;
                            acc.tx += gpu.tx as f64;
                            acc.temperature += gpu.temperature as f64;
                            acc.memory_bytes += gpu.resources.memory_bytes;
                            acc.compute_percentage += gpu.resources.compute_percentage as f64;

                            // Estimation of TFlops (not accurate because of
                            // a. MFU won't be 100%
                            // b. memory operations also treat as utilization in NVML
                            // c. nvml result is overestimated at some extent
                            acc.compute_tflops += gpu.resources.compute_percentage as f64
                                * GPU_CAPACITY_MAP
                                    .read()
                                    .expect("poisoned")
                                    .get(gpu_uuid)
                                    .unwrap_or(&0.0);
                            acc.count += 1;
                        }

                        // Accumulate process metrics
                        // First, collect all the process metrics data to avoid holding the lock across await points
                        let process_metrics_snapshot: Vec<(String, Vec<(u32, GpuResources)>)> = {
                            let metrics_guard = gpu_observer.metrics.read().expect("poisoned");
                            metrics_guard
                                .process_metrics
                                .iter()
                                .map(|(gpu_uuid, (process_metrics, _))| {
                                    let processes: Vec<(u32, GpuResources)> = process_metrics
                                        .iter()
                                        .map(|(pid, resources)| (*pid, resources.clone()))
                                        .collect();
                                    (gpu_uuid.clone(), processes)
                                })
                                .collect()
                        }; // RwLockReadGuard is dropped here

                        // Now process the collected data with async operations
                        for (gpu_uuid, process_metrics) in process_metrics_snapshot {
                            let worker_acc = worker_acc.entry(gpu_uuid.clone()).or_default();
                            for (pid, resources) in process_metrics.iter() {
                                let pod_entry = pod_mgr.find_pod_by_worker_pid(*pid).await;
                                if pod_entry.is_none() {
                                    tracing::debug!(
                                        msg = "Failed to find worker, GPU may used by unknown process not managed by TensorFusion",
                                        pid = *pid,
                                    );
                                    continue;
                                }
                                let pod_entry = pod_entry.unwrap();
                                let pod_identifier = pod_mgr.generate_pod_identifier_for_info(&pod_entry.info);
                                let acc = worker_acc.entry(pod_identifier).or_default();
                                acc.memory_bytes += resources.memory_bytes;
                                acc.compute_percentage += resources.compute_percentage as f64;
                                acc.compute_tflops += resources.compute_percentage as f64
                                    * GPU_CAPACITY_MAP
                                        .read()
                                        .expect("poisoned")
                                        .get(&gpu_uuid)
                                        .unwrap_or(&0.0);
                                acc.count += 1;
                            }
                        }

                        if counter >= metrics_batch_size {
                            let timestamp = current_time();
                            // Output averaged PCIE metrics
                            for (gpu_uuid, acc) in &gpu_acc {
                                if acc.count > 0 {
                                    let metrics_str = encoder.encode_gpu_metrics(
                                        gpu_uuid,
                                        node_name,
                                        gpu_pool,
                                        acc.rx / acc.count as f64,
                                        acc.tx / acc.count as f64,
                                        acc.temperature / acc.count as f64,
                                        acc.memory_bytes / acc.count as u64,
                                        acc.compute_percentage / acc.count as f64,
                                        acc.compute_tflops / acc.count as f64,
                                        timestamp,
                                    );
                                    tracing::info!(
                                        target: "metrics",
                                        msg = %metrics_str,
                                    );
                                }
                            }

                            // Output averaged worker metrics
                            let pod_registry = pod_mgr.registry().read().await;
                            for (gpu_uuid, pod_metrics) in &worker_acc {
                                for (pod_identifier, acc) in pod_metrics {
                                    let pod_entry = pod_registry.get(pod_identifier).unwrap();
                                    let labels = &pod_entry.info.labels;

                                    if acc.count > 0 {
                                        let mut extra_labels = HashMap::new();
                                        if has_dynamic_metrics_labels {
                                            for label in &metrics_extra_labels {
                                                extra_labels.insert(
                                                    label.clone(),
                                                    labels
                                                        .get(label)
                                                        .cloned()
                                                        .unwrap_or_else(|| "unknown".to_string()),
                                                );
                                            }
                                        }

                                        let metrics_str = encoder.encode_worker_metrics(
                                            gpu_uuid,
                                            node_name,
                                            gpu_pool,
                                            pod_identifier,
                                                                                    &pod_entry.info.namespace,
                                        pod_entry
                                                .info
                                                .workload_name
                                                .as_deref()
                                                .unwrap_or("unknown"),
                                            acc.memory_bytes / acc.count as u64,
                                            acc.compute_percentage / acc.count as f64,
                                            acc.compute_tflops / acc.count as f64,
                                            (acc.memory_bytes as f64 / acc.count as f64)
                                                / pod_entry.info.vram_limit.unwrap_or(0) as f64,
                                            timestamp,
                                            &extra_labels,
                                        );
                                        tracing::info!(
                                            target: "metrics",
                                            msg = %metrics_str,
                                        );
                                    }
                                }
                            }

                            // Reset accumulators and counter
                            gpu_acc.clear();
                            worker_acc.clear();
                            counter = 0;
                        }
                    }
                    None => {
                        tracing::info!("Metrics receiver closed");
                        break;
                    }
                }
            }
        }
    }
}

pub fn current_time() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}
