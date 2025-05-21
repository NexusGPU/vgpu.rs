use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use crate::gpu_observer::GpuObserver;

#[derive(Default)]
struct AccumulatedGpuMetrics {
    rx: f64,
    tx: f64,
    temperature: f64,
    memory_bytes: u64,
    compute_percentage: f64,
    count: usize,
}

#[derive(Default)]
struct AccumulatedWorkerMetrics {
    memory_bytes: u64,
    compute_percentage: f64,
    count: usize,
}

/// Run metrics collection in the current thread
/// This is meant to be called from a crossbeam scope thread
pub(crate) fn run_metrics(
    gpu_observer: Arc<GpuObserver>,
    worker_pid_mapping: Arc<RwLock<HashMap<u32, String>>>,
    metrics_batch_size: usize,
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
            acc.count += 1;
        }

        // Accumulate process metrics
        for (gpu_uuid, process_metrics) in metrics.process_metrics.iter() {
            let gpu_acc = worker_acc.entry(gpu_uuid.clone()).or_default();
            for (pid, resources) in process_metrics.iter() {
                let acc = gpu_acc.entry(*pid).or_default();
                acc.memory_bytes += resources.memory_bytes;
                acc.compute_percentage += resources.compute_percentage as f64;
                acc.count += 1;
            }
        }

        // Output averaged metrics every metrics_batch_size iterations
        if counter >= metrics_batch_size {
            // Output averaged PCIE metrics
            for (gpu_uuid, acc) in &gpu_acc {
                if acc.count > 0 {
                    tracing::info!(
                        target: "metrics.gpu_metrics_avg",
                        tag_uuid=gpu_uuid,
                        rx=acc.rx / acc.count as f64,
                        tx=acc.tx / acc.count as f64,
                        temperature=acc.temperature / acc.count as f64,
                        memory_bytes=acc.memory_bytes / acc.count as u64,
                        compute_percentage=acc.compute_percentage / acc.count as f64
                    );
                }
            }

            // Output averaged worker metrics
            let worker_pid_mapping = worker_pid_mapping.read().expect("poisoning");
            for (gpu_uuid, pid_metrics) in &worker_acc {
                for (pid, acc) in pid_metrics {
                    if acc.count > 0 {
                        let name = worker_pid_mapping.get(pid);
                        tracing::info!(
                            target: "metrics.worker_metrics_avg",
                            tag_uuid=gpu_uuid,
                            tag_worker=name.unwrap_or(&"unknown".to_string()),
                            memory_bytes=acc.memory_bytes / acc.count as u64,
                            compute_percentage=acc.compute_percentage / acc.count as f64
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
