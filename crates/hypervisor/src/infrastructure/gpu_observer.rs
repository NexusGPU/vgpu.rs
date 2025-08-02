use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use anyhow::Result;
use nvml_wrapper::enum_wrappers::device::PcieUtilCounter;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::Nvml;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::pod_management::PodStateStore;
use crate::process::GpuResources;

type ProcessId = u32;
type GpuUuid = String;
type ProcessMetrics = HashMap<ProcessId, GpuResources>;

#[derive(Debug, Default)]
pub struct GpuMetrics {
    // KB/s
    pub rx: u32,
    // KB/s
    pub tx: u32,

    pub temperature: u32,

    pub resources: GpuResources,
}

type LastSeenTimestamp = u64;
#[derive(Debug, Default)]
pub struct Metrics {
    pub process_metrics: HashMap<GpuUuid, (ProcessMetrics, LastSeenTimestamp)>,
    pub gpu_metrics: HashMap<GpuUuid, GpuMetrics>,
}

#[derive(Debug)]
pub struct GpuObserver {
    nvml: Arc<Nvml>,
    pub metrics: RwLock<Metrics>,
    senders: RwLock<Vec<mpsc::Sender<()>>>,
    pod_state_store: Arc<PodStateStore>,
}

// Explicitly implement Send and Sync for GpuObserver
unsafe impl Send for GpuObserver {}
unsafe impl Sync for GpuObserver {}

impl GpuObserver {
    pub(crate) fn create(nvml: Arc<Nvml>, pod_state_store: Arc<PodStateStore>) -> Arc<Self> {
        Arc::new(Self {
            nvml,
            metrics: Default::default(),
            senders: Default::default(),
            pod_state_store,
        })
    }

    /// Run the GPU observer loop asynchronously
    pub(crate) async fn run(
        &self,
        update_interval: Duration,
        cancellation_token: CancellationToken,
    ) {
        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => {
                    tracing::info!("GPU observer shutdown requested");
                    break;
                }
                _ = async {
                    // Query new metrics without holding the lock across .await
                    let metrics_result = {
                        let last_metrics_guard = self.metrics.read().await;
                        self.query_metrics(&last_metrics_guard)
                    };

                    match metrics_result {
                        Ok(metrics) => {
                            // Update the metrics – obtain write lock briefly
                            *self.metrics.write().await = metrics;

                            // Notify subscribers and clean up closed channels
                            self.notify_subscribers().await;
                        }
                        Err(e) => {
                            tracing::warn!("Failed to update GPU metrics: {}", e);
                        }
                    }

                    tokio::time::sleep(update_interval).await;
                } => {
                    // Continue the loop
                }
            }
        }
    }

    /// Notify all subscribers and clean up closed channels
    async fn notify_subscribers(&self) {
        // Clone sender list while holding read lock
        let sender_list = {
            let senders_guard = self.senders.read().await;
            senders_guard.clone()
        };

        // Track which senders are still valid
        let mut valid_senders = Vec::new();

        // Notify subscribers and identify closed channels
        for sender in sender_list {
            match sender.send(()).await {
                Ok(()) => {
                    // Channel is still open, keep it
                    valid_senders.push(sender);
                }
                Err(_) => {
                    // Channel is closed, don't keep it
                    // We don't log this as an error since it's normal for subscribers to disconnect
                    tracing::debug!("Subscriber disconnected, cleaning up channel");
                }
            }
        }

        // Update the senders list with only valid senders
        {
            let mut senders_guard = self.senders.write().await;
            *senders_guard = valid_senders;
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

            for process_info in running_compute_processes.iter() {
                if let UsedGpuMemory::Used(used) = process_info.used_gpu_memory {
                    process_metrics.insert(
                        process_info.pid,
                        GpuResources {
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
                                                Some(
                                                    sample.sm_util
                                                        + sample.enc_util
                                                        + sample.dec_util,
                                                )
                                            }
                                        })
                                        .sum();

                                    total / utilization_samples.len() as u32
                                }
                                _ => 0,
                            },
                        },
                    );
                }
            }

            let nvml_pids = running_compute_processes
                .iter()
                .map(|p| p.pid)
                .collect::<HashSet<_>>();
            let mut all_pids = HashSet::new();
            for pod_identifier in self.pod_state_store.list_pod_identifiers() {
                let pod_processes = self.pod_state_store.get_pod_processes(&pod_identifier);
                all_pids.extend(pod_processes.iter().map(|p| p.host_pid));
            }

            let idle_pids = all_pids.difference(&nvml_pids);

            for pid in idle_pids {
                process_metrics.insert(
                    *pid,
                    GpuResources {
                        memory_bytes: 0,
                        compute_percentage: 0,
                    },
                );
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
            gpu_process_metrics.insert(gpu_uuid, (process_metrics, newest_timestamp_candidate));
        }

        Ok(Metrics {
            process_metrics: gpu_process_metrics,
            gpu_metrics,
        })
    }

    pub(crate) async fn get_process_resources(
        &self,
        gpu_uuid: &str,
        process_id: u32,
    ) -> Option<GpuResources> {
        self.metrics
            .read()
            .await
            .process_metrics
            .get(gpu_uuid)
            .and_then(|(processes, _)| processes.get(&process_id))
            .cloned()
    }

    pub(crate) async fn subscribe(&self) -> mpsc::Receiver<()> {
        let (sender, receiver) = mpsc::channel(32);
        self.senders.write().await.push(sender);
        receiver
    }
}
