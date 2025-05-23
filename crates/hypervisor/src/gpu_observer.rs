use std::collections::HashMap;
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use nvml_wrapper::enum_wrappers::device::{PcieUtilCounter, TemperatureSensor};
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::Nvml;

use super::GpuResources;

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

#[derive(Debug, Default)]
pub(crate) struct Metrics {
    pub process_metrics: HashMap<GpuUuid, ProcessMetrics>,
    pub gpu_metrics: HashMap<GpuUuid, GpuMetrics>,
}

pub(crate) struct GpuObserver {
    nvml: Arc<Nvml>,
    pub metrics: RwLock<Metrics>,
    senders: RwLock<Vec<mpsc::Sender<()>>>,
}

impl GpuObserver {
    pub(crate) fn create(nvml: Arc<Nvml>) -> Arc<Self> {
        Arc::new(Self {
            nvml,
            metrics: Default::default(),
            senders: Default::default(),
        })
    }

    /// Run the GPU observer loop in the current thread
    /// This is meant to be called from a crossbeam scope thread
    pub(crate) fn run(&self, update_interval: Duration) {
        loop {
            let last_seen_timestamp = unix_as_millis()
                .saturating_mul(1000)
                .saturating_sub(update_interval.as_micros() as u64);

            match self.query_metrics(last_seen_timestamp) {
                Ok(metrics) => {
                    *self.metrics.write().expect("poisoned") = metrics;
                    let senders = self.senders.read().expect("poisoned");
                    for sender in senders.iter() {
                        if let Err(e) = sender.send(()) {
                            tracing::error!("Failed to send update signal: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to update GPU metrics: {}", e);
                }
            }
            thread::sleep(update_interval);
        }
    }

    fn query_metrics(&self, last_seen_timestamp: u64) -> Result<Metrics> {
        let mut gpu_metrics = HashMap::new();
        let mut gpu_process_metrics = HashMap::new();

        for i in 0..self.nvml.device_count()? {
            let device = self.nvml.device_by_index(i)?;
            let gpu_uuid = device.uuid()?.to_string();

            let mut process_metrics = HashMap::new();

            let utilizations = device
                .process_utilization_stats(last_seen_timestamp)
                .unwrap_or_default()
                .into_iter()
                .map(|p| (p.pid, p))
                .collect::<HashMap<_, _>>();

            let running_compute_processes = device.running_compute_processes()?;

            for process_info in running_compute_processes {
                if let UsedGpuMemory::Used(used) = process_info.used_gpu_memory {
                    process_metrics.insert(
                        process_info.pid,
                        GpuResources {
                            memory_bytes: used,
                            compute_percentage: match utilizations.get(&process_info.pid) {
                                Some(utilization) => {
                                    utilization.sm_util
                                        + utilization.enc_util
                                        + utilization.dec_util
                                }
                                None => 0,
                            },
                        },
                    );
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

            gpu_metrics.insert(
                gpu_uuid.clone(),
                GpuMetrics {
                    rx,
                    tx,
                    temperature,
                    resources: GpuResources {
                        memory_bytes: memory_info.used,
                        compute_percentage: utilization.gpu,
                    },
                },
            );
            gpu_process_metrics.insert(gpu_uuid, process_metrics);
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
            .and_then(|processes| processes.get(&process_id))
            .cloned()
    }

    pub(crate) fn subscribe(&self) -> mpsc::Receiver<()> {
        let (sender, receiver) = mpsc::channel();
        self.senders.write().expect("poisoned").push(sender);
        receiver
    }
}

pub(crate) fn unix_as_millis() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
