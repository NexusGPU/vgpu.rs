use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use anyhow::Result;
use nvml_wrapper::enum_wrappers::device::PcieUtilCounter;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::Nvml;
use tokio::sync::mpsc;

use crate::process::GpuResources;

type ProcessId = u32;
type GpuUuid = String;
type ProcessMetrics = HashMap<ProcessId, GpuResources>;

#[derive(Debug, Default)]
pub(crate) struct GpuMetrics {
    // KB/s
    pub rx: u32,
    // KB/s
    pub tx: u32,

    pub temperature: u32,

    pub resources: GpuResources,
}

type LastSeenTimestamp = u64;
#[derive(Debug, Default)]
pub(crate) struct Metrics {
    pub process_metrics: HashMap<GpuUuid, (ProcessMetrics, LastSeenTimestamp)>,
    pub gpu_metrics: HashMap<GpuUuid, GpuMetrics>,
}

#[derive(Debug)]
pub(crate) struct GpuObserver {
    nvml: Arc<Nvml>,
    pub metrics: RwLock<Metrics>,
    senders: RwLock<Vec<mpsc::Sender<()>>>,
}

// Explicitly implement Send and Sync for GpuObserver
unsafe impl Send for GpuObserver {}
unsafe impl Sync for GpuObserver {}

impl GpuObserver {
    pub(crate) fn create(nvml: Arc<Nvml>) -> Arc<Self> {
        Arc::new(Self {
            nvml,
            metrics: Default::default(),
            senders: Default::default(),
        })
    }

    /// Run the GPU observer loop asynchronously
    pub(crate) async fn run(&self, update_interval: Duration) {
        loop {
            // Query new metrics without holding the lock across .await
            let metrics_result = {
                let last_metrics_guard = self.metrics.read().expect("poisoned");
                self.query_metrics(&last_metrics_guard)
            };

            match metrics_result {
                Ok(metrics) => {
                    // Update the metrics – obtain write lock briefly
                    *self.metrics.write().expect("poisoned") = metrics;

                    // Clone sender list while holding read lock, then drop guard before await
                    let sender_list = {
                        let senders_guard = self.senders.read().expect("poisoned");
                        senders_guard.clone()
                    };

                    // Notify subscribers concurrently (sender::send is async)
                    for sender in sender_list {
                        if let Err(e) = sender.send(()).await {
                            tracing::error!("Failed to send update signal: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to update GPU metrics: {}", e);
                }
            }

            tokio::time::sleep(update_interval).await;
        }
    }

    fn query_metrics(&self, last_metrics: &Metrics) -> Result<Metrics> {
        let mut gpu_metrics: HashMap<String, GpuMetrics> = HashMap::new();
        let mut gpu_process_metrics = HashMap::new();

        for i in 0..self.nvml.device_count()? {
            let device = self.nvml.device_by_index(i)?;
            let gpu_uuid = device.uuid()?.to_lowercase();

            let last_seen_timestamp = last_metrics
                .process_metrics
                .get(&gpu_uuid)
                .map(|(_, ts)| *ts)
                .unwrap_or(0);
            let mut process_metrics = HashMap::new();
            let mut newest_timestamp_candidate = last_seen_timestamp;
            let mut utilizations = HashMap::new();
            for sample in device
                .process_utilization_stats(last_seen_timestamp)
                .unwrap_or_default()
            {
                if sample.timestamp > newest_timestamp_candidate {
                    newest_timestamp_candidate = sample.timestamp;
                }
                utilizations
                    .entry(sample.pid)
                    .or_insert_with(Vec::new)
                    .push(sample);
            }

            let running_compute_processes = device.running_compute_processes()?;

            for process_info in running_compute_processes {
                if let UsedGpuMemory::Used(used) = process_info.used_gpu_memory {
                    process_metrics.insert(process_info.pid, GpuResources {
                        memory_bytes: used,
                        compute_percentage: match utilizations.get(&process_info.pid) {
                            Some(utilization_samples) if !utilization_samples.is_empty() => {
                                // Calculate average utilization across all samples
                                let total: u32 = utilization_samples
                                    .iter()
                                    .filter_map(|sample| {
                                        if sample.sm_util > 100
                                            || sample.enc_util > 100
                                            || sample.dec_util > 100
                                            || sample.timestamp < last_seen_timestamp
                                        {
                                            None
                                        } else {
                                            Some(sample.sm_util + sample.enc_util + sample.dec_util)
                                        }
                                    })
                                    .sum();

                                total / utilization_samples.len() as u32
                            }
                            _ => 0,
                        },
                        tflops_request: None,
                        tflops_limit: None,
                        memory_limit: None,
                    });
                }
            }

            let tx = device.pcie_throughput(PcieUtilCounter::Send)?;
            let rx = device.pcie_throughput(PcieUtilCounter::Receive)?;

            // Get GPU temperature
            let temperature = device.temperature(TemperatureSensor::Gpu)?;

            // Get GPU memory info
            let memory_info = device.memory_info()?;
            // Get GPU utilization info
            let utilization = device.utilization_rates()?;

            gpu_metrics.insert(gpu_uuid.clone(), GpuMetrics {
                rx,
                tx,
                temperature,
                resources: GpuResources {
                    memory_bytes: memory_info.used,
                    compute_percentage: utilization.gpu,
                    tflops_request: None,
                    tflops_limit: None,
                    memory_limit: None,
                },
            });
            gpu_process_metrics.insert(gpu_uuid, (process_metrics, newest_timestamp_candidate));
        }

        Ok(Metrics {
            process_metrics: gpu_process_metrics,
            gpu_metrics,
        })
    }

    pub(crate) fn get_process_resources(
        &self,
        gpu_uuid: &str,
        process_id: u32,
    ) -> Option<GpuResources> {
        self.metrics
            .read()
            .expect("poisoned")
            .process_metrics
            .get(gpu_uuid)
            .and_then(|(processes, _)| processes.get(&process_id))
            .cloned()
    }

    pub(crate) fn subscribe(&self) -> mpsc::Receiver<()> {
        let (sender, receiver) = mpsc::channel(32);
        self.senders.write().expect("poisoned").push(sender);
        receiver
    }
}
