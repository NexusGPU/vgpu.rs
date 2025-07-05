use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use influxdb_line_protocol::LineProtocolBuilder;

use crate::config::GPU_CAPACITY_MAP;
use crate::gpu_observer::GpuObserver;

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
pub(crate) async fn run_metrics(
    gpu_observer: Arc<GpuObserver>,
    metrics_batch_size: usize,
    node_name: Option<String>,
    gpu_pool: Option<String>,
) {
    let node_name = node_name.unwrap_or("unknown".to_string());
    let gpu_pool = gpu_pool.unwrap_or("unknown".to_string());

    let mut gpu_acc: HashMap<String, AccumulatedGpuMetrics> = HashMap::new();
    let mut worker_acc: HashMap<String, HashMap<u32, AccumulatedWorkerMetrics>> = HashMap::new();
    let mut counter = 0;

    let mut receiver = gpu_observer.subscribe();
    while receiver.recv().await.is_some() {
        counter += 1;
        let metrics = gpu_observer.metrics.read().expect("poisoned");
        // Accumulate GPU metrics
        for (gpu_uuid, gpu) in metrics.gpu_metrics.iter() {
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
        for (gpu_uuid, (process_metrics, _)) in metrics.process_metrics.iter() {
            let gpu_acc = worker_acc.entry(gpu_uuid.clone()).or_default();
            for (pid, resources) in process_metrics.iter() {
                let acc = gpu_acc.entry(*pid).or_default();
                acc.memory_bytes += resources.memory_bytes;
                acc.compute_percentage += resources.compute_percentage as f64;
                acc.compute_tflops += resources.compute_percentage as f64
                    * GPU_CAPACITY_MAP
                        .read()
                        .expect("poisoned")
                        .get(gpu_uuid)
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
                    let lp = LineProtocolBuilder::new()
                        .measurement("tf_gpu_usage")
                        .tag("node_name", node_name.as_str())
                        .tag("pool", gpu_pool.as_str())
                        .tag("uuid", gpu_uuid.as_str())
                        .field("rx", acc.rx / acc.count as f64)
                        .field("tx", acc.tx / acc.count as f64)
                        .field("temperature", acc.temperature / acc.count as f64)
                        .field("memory_bytes", acc.memory_bytes / acc.count as u64)
                        .field(
                            "compute_percentage",
                            acc.compute_percentage / acc.count as f64,
                        )
                        .field("compute_tflops", acc.compute_tflops / acc.count as f64)
                        .timestamp(timestamp)
                        .close_line()
                        .build();
                    // Convert BytesWrapper to string first
                    let lp_str = BytesWrapper::from(lp).to_string();
                    tracing::info!(
                        target: "metrics",
                        msg = %lp_str,
                    );
                }
            }

            // Output averaged worker metrics
            for (gpu_uuid, pid_metrics) in &worker_acc {
                for (pid, acc) in pid_metrics {
                    if acc.count > 0 {
                        let lp = LineProtocolBuilder::new()
                            .measurement("tf_worker_usage")
                            .tag("node_name", node_name.as_str())
                            .tag("pool", gpu_pool.as_str())
                            .tag("uuid", gpu_uuid.as_str())
                            .tag("pid", &pid.to_string())
                            .field("memory_bytes", acc.memory_bytes / acc.count as u64)
                            .field(
                                "compute_percentage",
                                acc.compute_percentage / acc.count as f64,
                            )
                            .field("compute_tflops", acc.compute_tflops / acc.count as f64)
                            .timestamp(timestamp)
                            .close_line()
                            .build();
                        // Convert BytesWrapper to string first
                        let lp_str = BytesWrapper::from(lp).to_string();
                        tracing::info!(
                            target: "metrics",
                            msg = %lp_str,
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
        .as_nanos() as i64
}
