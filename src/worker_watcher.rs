use crate::gpu_observer::GpuObserver;
use crate::hypervisor::Hypervisor;
use crate::process::worker::TensorFusionWorker;
use crate::process::GpuResources;
use notify::{Error, Event, Watcher};
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::{fs, io};

pub struct WorkerWatcher {
    rx: Receiver<Result<Event, Error>>,
    hypervisor: Arc<Hypervisor>,
}

impl WorkerWatcher {
    pub fn new<P: AsRef<Path>>(path: P, hypervisor: Arc<Hypervisor>) -> Result<Self, Error> {
        let channel = mpsc::channel::<Result<Event, Error>>();
        let (tx, rx) = channel;

        let entries = fs::read_dir(path.as_ref())?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect::<Vec<_>>();

        for entry in entries {
            let event = Event::new(notify::event::EventKind::Create(
                notify::event::CreateKind::File,
            ))
            .add_path(entry);
            let _ = tx.send(Ok(event));
        }

        let mut watcher = notify::recommended_watcher(tx)?;
        watcher.watch(path.as_ref(), notify::RecursiveMode::NonRecursive)?;
        Ok(WorkerWatcher { rx, hypervisor })
    }

    pub fn run(&self, gpu_observer: Arc<GpuObserver>) {
        for res in self.rx.iter() {
            match res {
                Ok(event) => match event.kind {
                    notify::EventKind::Create(_) => {
                        if let Some(path) = event.paths.first() {
                            let (pid, worker_name) = match extract_pid_worker_name_from_path(path) {
                                Ok(pid_worker_name) => pid_worker_name,
                                Err(msg) => {
                                    tracing::warn!("{}", msg);
                                    continue;
                                }
                            };

                            let uuid = match read_process_env_vars(pid) {
                                Ok(mut env) => {
                                    if let Some(uuid) = env.remove("NVIDIA_VISIBLE_DEVICES") {
                                        uuid
                                    } else {
                                        tracing::warn!(
                                            "no visible device for worker: {:?}, skipped",
                                            path
                                        );
                                        continue;
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "cannot read env vars for worker: {:?}, err: {:?} skipped",
                                        path,
                                        e
                                    );
                                    continue;
                                }
                            };

                            let worker = TensorFusionWorker::new(
                                pid,
                                path.clone(),
                                GpuResources {
                                    memory_bytes: 0,
                                    compute_percentage: 0,
                                },
                                uuid,
                                gpu_observer.clone(),
                            );

                            self.hypervisor.add_process(worker_name, Arc::new(worker));
                        }
                    }
                    notify::EventKind::Remove(_) => {
                        if let Some(path) = event.paths.first() {
                            let (pid, _) = match extract_pid_worker_name_from_path(path) {
                                Ok(pid_worker_name) => pid_worker_name,
                                Err(msg) => {
                                    tracing::warn!("{}", msg);
                                    continue;
                                }
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

fn extract_pid_worker_name_from_path(path: &std::path::Path) -> Result<(u32, String), String> {
    // Extract PID from filename
    let pid = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("could not extract PID from path: {:?}, skipped", path))
        .and_then(|s| {
            s.parse::<u32>().map_err(|e| {
                format!(
                    "failed to parse PID from path: {:?}, error: {:?}, skipped",
                    path, e
                )
            })
        })?;

    // Extract worker name from parent directory
    let worker_name = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .map(String::from)
        .ok_or_else(|| {
            format!(
                "could not extract worker name from path: {:?}, skipped",
                path
            )
        })?;

    Ok((pid, worker_name))
}

fn read_process_env_vars(pid: u32) -> Result<HashMap<String, String>, io::Error> {
    let environ_path = format!("/proc/{}/environ", pid);
    let content = fs::read(&environ_path)?;

    let mut env_vars = HashMap::new();
    for var in content.split(|&b| b == 0) {
        if var.is_empty() {
            continue;
        }
        if let Ok(var_str) = String::from_utf8(var.to_vec()) {
            if let Some((key, value)) = var_str.split_once('=') {
                env_vars.insert(key.to_string(), value.to_string());
            }
        }
    }

    Ok(env_vars)
}
