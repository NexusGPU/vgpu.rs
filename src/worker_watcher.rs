use crate::hypervisor::Hypervisor;
use crate::process::worker::TensorFusionWorker;
use crate::process::GpuResources;
use notify::{Error, Event, Watcher};
use nvml_wrapper::Nvml;
use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

pub struct WorkerWatcher {
    rx: Receiver<Result<Event, Error>>,
    hypervisor: Arc<Hypervisor>,
}

impl WorkerWatcher {
    pub fn new<P: AsRef<Path>>(path: P, hypervisor: Arc<Hypervisor>) -> Result<Self, Error> {
        let (tx, rx) = mpsc::channel::<Result<Event, Error>>();
        let mut watcher = notify::recommended_watcher(tx)?;
        watcher.watch(path.as_ref(), notify::RecursiveMode::NonRecursive)?;
        Ok(WorkerWatcher { rx, hypervisor })
    }

    pub fn run(&self, nvml: Arc<Nvml>) {
        for res in self.rx.iter() {
            match res {
                Ok(event) => match event.kind {
                    notify::EventKind::Create(_) => {
                        if let Some(path) = event.paths.first() {
                            // The sock file name is {pid}.sock
                            let id = parse_pid_from_sockpath(path);
                            let pid = if let Some(pid) = id {
                                pid
                            } else {
                                tracing::warn!("invaild sock file: {:?}, skipped", path);
                                continue;
                            };

                            let worker = TensorFusionWorker::new(
                                pid,
                                path.clone(),
                                GpuResources {
                                    memory_bytes: 0,
                                    compute_percentage: 0,
                                },
                                nvml.clone(),
                            );

                            self.hypervisor.add_process(Arc::new(worker));
                        }
                    }
                    notify::EventKind::Remove(_) => {
                        if let Some(path) = event.paths.first() {
                            let id = parse_pid_from_sockpath(path);
                            let pid = if let Some(pid) = id {
                                pid
                            } else {
                                tracing::warn!("invaild sock file: {:?}, skipped", path);
                                continue;
                            };

                            self.hypervisor.remove_process(pid);
                        }
                    }
                    _ => {}
                },
                Err(e) => tracing::error!("watch error: {:?}", e),
            }
        }
    }
}

fn parse_pid_from_sockpath(path: &Path) -> Option<u32> {
    path.file_name()
        .and_then(|file_name| file_name.to_str().and_then(|s| s.split('.').next()))
        .and_then(|pid| pid.parse::<u32>().map(Some).unwrap_or_default())
}
