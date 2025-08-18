use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use tokio_util::sync::CancellationToken;

use crate::config::GPU_CAPACITY_MAP;
use crate::gpu_observer::GpuObserver;
use crate::metrics::encoders::GpuMetricsParams;
use crate::metrics::encoders::WorkerMetricsParams;
use crate::process::GpuResources;

use crate::pod_management::PodManager;

pub mod encoders;
use encoders::create_encoder;
use encoders::MetricsEncoder as _;

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
                tracing::error!(msg = "Failed to convert bytes to string",);
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
    memory_bytes: u64,
    memory_percentage: f64,
    compute_percentage: f64,
    compute_tflops: f64,

    rx: f64,
    tx: f64,
    temperature: f64,
    graphics_clock_mhz: f64,
    sm_clock_mhz: f64,
    memory_clock_mhz: f64,
    video_clock_mhz: f64,
    power_usage: i64,
    nvlink_rx_bandwidth: i64,
    nvlink_tx_bandwidth: i64,

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
pub(crate) async fn run_metrics<M, P, D, T>(
    gpu_observer: Arc<GpuObserver>,
    metrics_batch_size: usize,
    node_name: &str,
    gpu_pool: Option<&str>,
    pod_mgr: Arc<PodManager<M, P, D, T>>,
    metrics_format: &str,
    metrics_extra_labels: Option<&str>,
    cancellation_token: CancellationToken,
) {
    let gpu_pool = gpu_pool.unwrap_or("unknown");
    let encoder = create_encoder(metrics_format);

    let mut gpu_acc: HashMap<String, AccumulatedGpuMetrics> = HashMap::new();

    let mut worker_acc: HashMap<
        // gpu_uuid
        String,
        // pod_identifier -> pid -> metrics
        HashMap<String, HashMap<u32, AccumulatedWorkerMetrics>>,
    > = HashMap::new();
    let mut counter = 0;

    let metrics_extra_labels: HashMap<String, String> = metrics_extra_labels
        .map(|labels_json| {
            if labels_json == "null" {
                HashMap::new()
            } else {
                serde_json::from_str(labels_json).unwrap_or_else(|e| {
                    tracing::warn!(
                        "Failed to parse metrics_extra_labels JSON: {}, using empty map",
                        e
                    );
                    HashMap::new()
                })
            }
        })
        .unwrap_or_default();
    let has_dynamic_metrics_labels = !metrics_extra_labels.is_empty();

    let mut receiver = gpu_observer.subscribe().await;

    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                tracing::info!("Metrics collection shutdown requested");
                break;
            }
            recv_result = receiver.recv() => {
                if recv_result.is_none() {
                    tracing::info!("Metrics receiver closed");
                    break;
                }

                counter += 1;

                // Accumulate GPU metrics
                for (gpu_uuid, gpu) in gpu_observer
                    .metrics
                    .read()
                    .await
                    .gpu_metrics
                    .iter()
                {
                    let acc = gpu_acc.entry(gpu_uuid.to_string()).or_default();
                    acc.rx += gpu.rx as f64;
                    acc.tx += gpu.tx as f64;
                    acc.temperature += gpu.temperature as f64;
                    acc.graphics_clock_mhz += gpu.graphics_clock_mhz as f64;
                    acc.sm_clock_mhz += gpu.sm_clock_mhz as f64;
                    acc.memory_clock_mhz += gpu.memory_clock_mhz as f64;
                    acc.video_clock_mhz += gpu.video_clock_mhz as f64;
                    acc.memory_bytes += gpu.resources.memory_bytes;
                    acc.memory_percentage += gpu.memory_percentage;
                    acc.compute_percentage += gpu.resources.compute_percentage as f64;
                    acc.power_usage += gpu.power_usage as i64;

                    // Estimation of TFlops (approximate)
                    acc.compute_tflops += gpu.resources.compute_percentage as f64
                        * GPU_CAPACITY_MAP
                            .read()
                            .expect("should not be poisoned")
                            .get(gpu_uuid)
                            .unwrap_or(&0.0) / 100.0;
                    acc.count += 1;
                }

                // Accumulate process metrics
                // First, collect all the process metrics data to avoid holding the lock across await points
                let process_metrics_snapshot: Vec<(String, Vec<(u32, GpuResources)>)> = {
                    let metrics_guard = gpu_observer.metrics.read().await;
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
                    let worker_acc = worker_acc.entry(gpu_uuid.to_string()).or_default();
                    for (pid, resources) in process_metrics.iter() {
                        let Some(pod_identifier) = pod_mgr.find_pod_by_worker_pid(*pid) else {
                            tracing::debug!(
                                msg = "Failed to find worker, GPU may used by unknown process not managed by TensorFusion",
                                pid = *pid,
                            );
                            continue;
                        };
                        let acc = worker_acc.entry(pod_identifier).or_default();
                        let acc = acc.entry(*pid).or_default();
                        acc.memory_bytes += resources.memory_bytes;
                        acc.compute_percentage += resources.compute_percentage as f64;
                        acc.compute_tflops += resources.compute_percentage as f64
                            * GPU_CAPACITY_MAP
                                .read()
                                .expect("should not be poisoned")
                                .get(&gpu_uuid)
                                .unwrap_or(&0.0) / 100.0;
                        acc.count += 1;
                    }
                }

                // Not enough samples yet, keep accumulating
                if counter < metrics_batch_size {
                    continue;
                }

                let timestamp = current_time();

                // Output averaged GPU metrics
                for (gpu_uuid, acc) in &gpu_acc {
                    if acc.count == 0 { continue; }
                    let metrics_str = encoder.encode_gpu_metrics_with_params(&GpuMetricsParams {
                        gpu_uuid,
                        node_name,
                        gpu_pool,
                        rx: acc.rx / acc.count as f64,
                        tx: acc.tx / acc.count as f64,
                        nvlink_rx_bandwidth: acc.nvlink_rx_bandwidth / acc.count as i64,
                        nvlink_tx_bandwidth: acc.nvlink_tx_bandwidth / acc.count as i64,
                        temperature: acc.temperature / acc.count as f64,
                        graphics_clock_mhz: acc.graphics_clock_mhz / acc.count as f64,
                        sm_clock_mhz: acc.sm_clock_mhz / acc.count as f64,
                        memory_clock_mhz: acc.memory_clock_mhz / acc.count as f64,
                        video_clock_mhz: acc.video_clock_mhz / acc.count as f64,
                        memory_bytes: acc.memory_bytes / acc.count as u64,
                        memory_percentage: acc.memory_percentage / acc.count as f64,
                        compute_percentage: acc.compute_percentage / acc.count as f64,
                        compute_tflops: acc.compute_tflops / acc.count as f64,
                        power_usage: acc.power_usage / acc.count as i64,
                        timestamp,
                    });
                    tracing::info!(
                        target: "metrics",
                        msg = %metrics_str,
                    );
                }

                // Output averaged worker metrics
                for (gpu_uuid, pod_metrics) in &worker_acc {
                    for (pod_identifier, acc_map) in pod_metrics {
                        let Some(pod_state) = pod_mgr.pod_state_store().get_pod(pod_identifier) else {
                            tracing::warn!(
                                msg = "Failed to find pod",
                                pod_identifier = %pod_identifier,
                            );
                            continue;
                        };

                        let labels = &pod_state.info.labels;

                        let mut memory_bytes = 0;
                        let mut compute_percentage = 0.0;
                        let mut compute_tflops = 0.0;
                        let mut memory_percentage = 0.0;
                        for (_pid, acc) in acc_map.iter() {
                            memory_bytes += acc.memory_bytes / acc.count as u64;
                            compute_percentage += acc.compute_percentage / acc.count as f64;
                            compute_tflops += acc.compute_tflops / acc.count as f64;
                            memory_percentage += {
                                let avg_memory_bytes = acc.memory_bytes as f64 / acc.count as f64;
                                let vram_limit = pod_state.info.vram_limit.unwrap_or(0) as f64;
                                if vram_limit > 0.0 { avg_memory_bytes / vram_limit * 100.0 } else { 0.0 }
                            }
                        }

                        let mut extra_labels = HashMap::new();
                        if has_dynamic_metrics_labels {
                            for (label, value) in &metrics_extra_labels {
                                extra_labels.insert(
                                    value.clone(),
                                    labels
                                        .get(label)
                                        .cloned()
                                        .unwrap_or_else(|| "unknown".to_string()),
                                );
                            }
                        }

                        let metrics_str = encoder.encode_worker_metrics_with_params(&WorkerMetricsParams {
                            gpu_uuid,
                            node_name,
                            gpu_pool,
                            pod_name: &pod_state.info.pod_name,
                            namespace: &pod_state.info.namespace,
                            workload: pod_state
                                .info
                                .workload_name
                                .as_deref()
                                .unwrap_or("unknown"),
                            memory_bytes,
                            compute_percentage,
                            compute_tflops,
                            memory_percentage,
                            timestamp,
                            extra_labels: &extra_labels,
                        });
                        tracing::info!(
                            target: "metrics",
                            msg = %metrics_str,
                        );
                    }
                }

                // Reset accumulators and counter
                gpu_acc.clear();
                worker_acc.clear();
                counter = 0;
            }
        }
    }
}

pub fn current_time() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("should compute duration since UNIX_EPOCH")
        .as_millis() as i64
}
