use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use crate::config::GPU_CAPACITY_MAP;
use crate::gpu_observer::GpuObserver;
use crate::worker_manager::WorkerManager;

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
pub(crate) async fn run_metrics<AddCB, RemoveCB>(
    gpu_observer: Arc<GpuObserver>,
    metrics_batch_size: usize,
    node_name: String,
    gpu_pool: Option<String>,
    worker_mgr: Arc<WorkerManager<AddCB, RemoveCB>>,
    metrics_format: String,
    metrics_extra_labels: Option<String>,
) {
    let gpu_pool = gpu_pool.unwrap_or("unknown".to_string());
    let encoder = create_encoder(&metrics_format);

    let mut gpu_acc: HashMap<String, AccumulatedGpuMetrics> = HashMap::new();

    // level 1 key is gpu_uuid, level 2 key is pod_ns/pod_name, value is GPU usage metrics
    let mut worker_acc: HashMap<String, HashMap<String, AccumulatedWorkerMetrics>> = HashMap::new();
    let mut counter = 0;

    let metrics_extra_labels = metrics_extra_labels.unwrap_or_default();
    let metrics_extra_labels = metrics_extra_labels.split(',').collect::<Vec<&str>>();
    let has_dynamic_metrics_labels = metrics_extra_labels.len() > 0;

    let mut receiver = gpu_observer.subscribe();
    while receiver.recv().await.is_some() {
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
                    .unwrap();
            acc.count += 1;
        }

        // Accumulate process metrics
        // First, collect all the process metrics data to avoid holding the lock across await points
        let process_metrics_snapshot: Vec<(String, Vec<(u32, crate::GpuResources)>)> = {
            let metrics_guard = gpu_observer.metrics.read().expect("poisoned");
            metrics_guard
                .process_metrics
                .iter()
                .map(|(gpu_uuid, (process_metrics, _))| {
                    let processes: Vec<(u32, crate::GpuResources)> = process_metrics
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
                let worker_entry = worker_mgr.find_worker_by_pid(*pid).await;
                if worker_entry.is_none() {
                    tracing::debug!(
                        msg = "Failed to find worker, GPU may used by unknown process not managed by TensorFusion",
                        pid = *pid,
                    );
                    continue;
                }
                let worker_info = worker_entry.unwrap().info;
                let pod_identifier = format!("{}:{}", worker_info.namespace, worker_info.pod_name);
                let acc = worker_acc.entry(pod_identifier).or_default();
                acc.memory_bytes += resources.memory_bytes;
                acc.compute_percentage += resources.compute_percentage as f64;
                acc.compute_tflops += resources.compute_percentage as f64
                    * GPU_CAPACITY_MAP
                        .read()
                        .expect("poisoned")
                        .get(&gpu_uuid)
                        .unwrap();
                acc.count += 1;
            }
        }

        // Output averaged metrics every metrics_batch_size iterations
        if counter >= metrics_batch_size {
            let timestamp = current_time();
            // Output averaged PCIE metrics
            for (gpu_uuid, acc) in &gpu_acc {
                if acc.count > 0 {
                    let metrics_str = encoder.encode_gpu_metrics(
                        gpu_uuid,
                        &node_name,
                        &gpu_pool,
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
            let worker_registry = worker_mgr.registry().read().await;
            for (gpu_uuid, pod_metrics) in &worker_acc {
                for (pod_identifier, acc) in pod_metrics {
                    let worker_entry = worker_registry.get(pod_identifier).unwrap();
                    let labels = worker_entry.info.labels.clone();

                    if acc.count > 0 {
                        let mut extra_labels = HashMap::new();
                        if has_dynamic_metrics_labels {
                            for label in &metrics_extra_labels {
                                extra_labels.insert(
                                    label.to_string(),
                                    labels
                                        .get(*label)
                                        .unwrap_or(&String::from("unknown"))
                                        .clone(),
                                );
                            }
                        }

                        let metrics_str = encoder.encode_worker_metrics(
                            gpu_uuid,
                            &node_name,
                            &gpu_pool,
                            pod_identifier,
                            &worker_entry.info.namespace,
                            worker_entry
                                .info
                                .workload_name
                                .as_deref()
                                .unwrap_or("unknown"),
                            acc.memory_bytes / acc.count as u64,
                            acc.compute_percentage / acc.count as f64,
                            acc.compute_tflops / acc.count as f64,
                            (acc.memory_bytes as f64 / acc.count as f64)
                                / worker_entry.info.vram_limit.unwrap_or(0) as f64,
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
}

pub fn current_time() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}
