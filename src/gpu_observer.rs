use std::collections::HashMap;
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use nvml_wrapper::enum_wrappers::device::PcieUtilCounter;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::Nvml;
use tracing;

use super::GpuResources;

type ProcessId = u32;
type GpuUuid = String;
type ProcessMetrics = HashMap<ProcessId, GpuResources>;

#[derive(Debug, Default)]
pub struct PcieThroughput {
    // KB/s
    pub rx: u32,
    // KB/s
    pub tx: u32,
}

#[derive(Debug, Default)]
pub struct Metrics {
    pub process_metrics: HashMap<GpuUuid, ProcessMetrics>,
    pub pcie_throughput: HashMap<GpuUuid, PcieThroughput>,
}

pub struct GpuObserver {
    nvml: Arc<Nvml>,
    pub metrics: RwLock<Metrics>,
    senders: RwLock<Vec<mpsc::Sender<()>>>,
}

impl GpuObserver {
    pub fn create(nvml: Arc<Nvml>, update_interval: Duration) -> Arc<Self> {
        let self_arc = Arc::new(Self {
            nvml,
            metrics: Default::default(),
            senders: Default::default(),
        });
        let self_cloned = self_arc.clone();
        let _ = std::thread::Builder::new()
            .name("gpu_observer".into())
            .spawn(move || loop {
                let last_seen_timestamp =
                    unix_as_millis().saturating_sub(update_interval.as_millis() as u64);

                match self_cloned.query_metrics(last_seen_timestamp) {
                    Ok(metrics) => {
                        *self_cloned.metrics.write().expect("poisoned") = metrics;
                        let senders = self_cloned.senders.read().expect("poisoned");
                        for sender in senders.iter() {
                            if let Err(e) = sender.send(()) {
                                tracing::error!("Failed to send update signal: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to update GPU metrics: {}", e);
                    }
                }
                thread::sleep(update_interval);
            });

        self_arc
    }

    fn query_metrics(&self, last_seen_timestamp: u64) -> Result<Metrics> {
        let mut gpu_process_metrics = HashMap::new();
        let mut pcie_throughput = HashMap::new();

        for i in 0..self.nvml.device_count()? {
            let device = self.nvml.device_by_index(i)?;
            let gpu_uuid = device.uuid()?.to_string();

            let mut process_metrics = HashMap::new();

            let utilizations = device.process_utilization_stats(last_seen_timestamp)?;
            let running_compute_processes = device.running_compute_processes().map(|p| {
                p.into_iter()
                    .filter_map(|p| {
                        if let UsedGpuMemory::Used(used) = p.used_gpu_memory {
                            Some((p.pid, used))
                        } else {
                            None
                        }
                    })
                    .collect::<HashMap<_, _>>()
            })?;

            for utilization in utilizations {
                process_metrics.insert(
                    utilization.pid,
                    GpuResources {
                        memory_bytes: running_compute_processes[&utilization.pid],
                        compute_percentage: utilization.sm_util
                            + utilization.enc_util
                            + utilization.dec_util,
                    },
                );
            }

            let tx = device.pcie_throughput(PcieUtilCounter::Send)?;
            let rx = device.pcie_throughput(PcieUtilCounter::Receive)?;
            pcie_throughput.insert(gpu_uuid.clone(), PcieThroughput { rx, tx });
            gpu_process_metrics.insert(gpu_uuid, process_metrics);
        }

        Ok(Metrics {
            process_metrics: gpu_process_metrics,
            pcie_throughput,
        })
    }

    pub fn get_process_resources(&self, gpu_uuid: &str, process_id: u32) -> Option<GpuResources> {
        self.metrics
            .read()
            .expect("poisoned")
            .process_metrics
            .get(gpu_uuid)
            .and_then(|processes| processes.get(&process_id))
            .cloned()
    }

    pub fn subscribe(&self) -> mpsc::Receiver<()> {
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
