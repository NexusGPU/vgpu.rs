use std::{collections::HashMap, sync::Arc};

use crate::{gpu_observer::GpuObserver, hypervisor::Hypervisor};

#[derive(Default)]
struct AccumulatedPcie {
    rx: f64,
    tx: f64,
    count: usize,
}

#[derive(Default)]
struct AccumulatedWorkerMetrics {
    memory_bytes: u64,
    compute_percentage: f64,
    count: usize,
}

pub(crate) fn output_metrics(gpu_observer: Arc<GpuObserver>, hypervisor: Arc<Hypervisor>) {
    let receiver = gpu_observer.subscribe();
    let _ = std::thread::Builder::new()
        .name("output metrics".into())
        .spawn({
            let gpu_observer = gpu_observer.clone();
            move || {
                let mut pcie_acc: HashMap<String, AccumulatedPcie> = HashMap::new();
                let mut worker_acc: HashMap<String, HashMap<u32, AccumulatedWorkerMetrics>> =
                    HashMap::new();
                let mut counter = 0;

                for _ in receiver.iter() {
                    counter += 1;
                    let metrics = gpu_observer.metrics.read().expect("poisoned");
                    // Accumulate PCIE metrics
                    for (gpu_uuid, pcie) in metrics.pcie_throughput.iter() {
                        let acc = pcie_acc.entry(gpu_uuid.clone()).or_default();
                        acc.rx += pcie.rx as f64;
                        acc.tx += pcie.tx as f64;
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

                    // Output averaged metrics every 10 iterations
                    if counter >= 10 {
                        // Output averaged PCIE metrics
                        for (gpu_uuid, acc) in &pcie_acc {
                            if acc.count > 0 {
                                tracing::info!(
                                    target: "metrics.pcie_throughput_avg",
                                    tag_uuid=gpu_uuid,
                                    rx=acc.rx / acc.count as f64,
                                    tx=acc.tx / acc.count as f64,
                                );
                            }
                        }

                        // Output averaged worker metrics
                        let worker_pid_mapping =
                            hypervisor.worker_pid_mapping.read().expect("poisoned");
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
                        pcie_acc.clear();
                        worker_acc.clear();
                        counter = 0;
                    }
                }
            }
        });
}
