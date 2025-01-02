use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use anyhow::Result;
use nvml_wrapper::Nvml;
use tracing;

use super::GpuResources;

type ProcessId = u32;
type GpuUuid = String;
type ProcessMetrics = HashMap<ProcessId, GpuResources>;
type GpuMetrics = HashMap<GpuUuid, ProcessMetrics>;

pub struct GpuObserver {
    metrics: Arc<RwLock<GpuMetrics>>,
}

impl GpuObserver {
    pub fn new(nvml: Arc<Nvml>, update_interval: Duration) -> Self {
        let metrics = Arc::new(RwLock::new(HashMap::new()));
        let metrics_clone = metrics.clone();

        thread::spawn(move || {
            loop {
                if let Err(e) = Self::update_metrics(&nvml, &metrics_clone) {
                    tracing::error!("Failed to update GPU metrics: {}", e);
                }
                thread::sleep(update_interval);
            }
        });

        Self { metrics }
    }

    fn update_metrics(nvml: &Nvml, metrics: &RwLock<GpuMetrics>) -> Result<()> {
        let mut new_metrics = HashMap::new();

        for device in nvml.devices()? {
            let gpu_uuid = device.uuid()?.to_string();
            let mut process_metrics = HashMap::new();

            for process in device.running_compute_processes()? {
                let memory = process.used_memory();
                let utilization = device.utilization_rates()?;

                process_metrics.insert(
                    process.pid(),
                    GpuResources {
                        memory_bytes: memory,
                        compute_percentage: utilization.gpu,
                    },
                );
            }

            new_metrics.insert(gpu_uuid, process_metrics);
        }

        *metrics.write().unwrap() = new_metrics;
        Ok(())
    }

    pub fn get_process_resources(&self, gpu_uuid: &str, process_id: u32) -> Option<GpuResources> {
        self.metrics
            .read()
            .unwrap()
            .get(gpu_uuid)
            .and_then(|processes| processes.get(&process_id))
            .cloned()
    }
}
