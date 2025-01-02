use crate::gpu_observer::GpuObserver;
use crate::hypervisor::Hypervisor;
use crate::process::worker::TensorFusionWorker;
use crate::process::GpuResources;
use notify::{Error, Event, Watcher};
use nvml_wrapper::Nvml;
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
        let mut watcher = notify::recommended_watcher(tx)?;
        watcher.watch(path.as_ref(), notify::RecursiveMode::NonRecursive)?;
        Ok(WorkerWatcher { rx, hypervisor })
    }

    pub fn run(&self, nvml: Arc<Nvml>, gpu_observer: Arc<GpuObserver>) {
        for res in self.rx.iter() {
            match res {
                Ok(event) => match event.kind {
                    notify::EventKind::Create(_) => {
                        if let Some(path) = event.paths.first() {
                            let id = find_socket_listener_pid(path);
                            let pid = match id {
                                Ok(pid) => pid,
                                Err(e) => {
                                    tracing::warn!(
                                        "invaild sock file: {:?}, err: {:?}, skipped",
                                        path,
                                        e
                                    );
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
                                nvml.clone(),
                                uuid,
                                gpu_observer.clone(),
                            );

                            self.hypervisor.add_process(Arc::new(worker));
                        }
                    }
                    notify::EventKind::Remove(_) => {
                        if let Some(path) = event.paths.first() {
                            let id = find_socket_listener_pid(path);
                            let pid = match id {
                                Ok(pid) => pid,
                                Err(e) => {
                                    tracing::warn!(
                                        "invaild sock file: {:?}, err: {:?},skipped",
                                        path,
                                        e
                                    );
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

fn find_socket_listener_pid(socket_path: &Path) -> Result<u32, io::Error> {
    let proc_dir = fs::read_dir("/proc")?;
    let socket_canonical = socket_path.canonicalize()?;

    for entry in proc_dir {
        let entry = entry?;
        // Skip if not a directory or not a number (PID)
        let file_name = entry.file_name();
        let pid_str = file_name.to_string_lossy();
        if !pid_str.chars().all(|c| c.is_digit(10)) {
            continue;
        }

        let fd_dir = format!("/proc/{}/fd", pid_str);
        if let Ok(fd_entries) = fs::read_dir(fd_dir) {
            for fd_entry in fd_entries {
                if let Ok(fd_entry) = fd_entry {
                    if let Ok(target) = fs::read_link(fd_entry.path()) {
                        if target == socket_canonical {
                            return Ok(pid_str.parse().unwrap());
                        }
                    }
                }
            }
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "No process found listening on the socket",
    ))
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
