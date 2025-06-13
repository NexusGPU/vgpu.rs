use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use influxdb_line_protocol::LineProtocolBuilder;

use crate::config::GPU_CAPACITY_MAP;
use crate::gpu_observer::GpuObserver;

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

/// Run metrics collection in the current thread
pub(crate) fn run_metrics(
    gpu_observer: Arc<GpuObserver>,
    worker_pid_mapping: Arc<RwLock<HashMap<u32, (String, String)>>>,
    metrics_batch_size: usize,
    gpu_node: String,
    gpu_pool: String,
) {
    let mut gpu_acc: HashMap<String, AccumulatedGpuMetrics> = HashMap::new();
    let mut worker_acc: HashMap<String, HashMap<u32, AccumulatedWorkerMetrics>> = HashMap::new();
    let mut counter = 0;

    for _ in gpu_observer.subscribe().iter() {
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
                        .tag("node_name", gpu_node.as_str())
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
                    let lp_str = std::str::from_utf8(&lp).unwrap();
                    tracing::info!(
                        target: "metrics",
                        msg=lp_str,
                    );
                }
            }

            // Output averaged worker metrics
            let worker_pid_mapping = worker_pid_mapping.read().expect("poisoning");
            for (gpu_uuid, pid_metrics) in &worker_acc {
                for (pid, acc) in pid_metrics {
                    if acc.count > 0 {
                        let (worker_name, workload) = worker_pid_mapping
                            .get(pid)
                            .cloned()
                            .unwrap_or((String::from("unknown"), String::from("unknown")));

                        let lp = LineProtocolBuilder::new()
                            .measurement("tf_worker_usage")
                            .tag("node_name", gpu_node.as_str())
                            .tag("pool", gpu_pool.as_str())
                            .tag("uuid", gpu_uuid.as_str())
                            .tag("worker", worker_name.as_str())
                            .tag("workload", workload.as_str())
                            .field("memory_bytes", acc.memory_bytes / acc.count as u64)
                            .field(
                                "compute_percentage",
                                acc.compute_percentage / acc.count as f64,
                            )
                            .field("compute_tflops", acc.compute_tflops / acc.count as f64)
                            .timestamp(timestamp)
                            .close_line()
                            .build();
                        let lp_str = std::str::from_utf8(&lp).unwrap();
                        tracing::info!(
                            target: "metrics",
                            msg=lp_str,
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
