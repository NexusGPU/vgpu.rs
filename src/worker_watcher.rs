use crate::gpu_observer::GpuObserver;
use crate::hypervisor::Hypervisor;
use crate::process::worker::TensorFusionWorker;
use crate::process::GpuResources;
use notify::{Error, Event, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
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

                            let worker_name = path.file_name().expect("file_name");
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

                            self.hypervisor.add_process(
                                worker_name
                                    .to_str()
                                    .expect("invaild worker name")
                                    .to_string(),
                                Arc::new(worker),
                            );
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

    // query sock file inode
    // cat /proc/net/unix | grep $socket_path
    let unix_sockets = fs::read_to_string("/proc/net/unix")?;
    let socket_name = socket_canonical.to_string_lossy();
    let socket_inode = unix_sockets
        .lines()
        .find(|line| line.contains(&*socket_name))
        .and_then(|line| line.split_whitespace().nth(6))
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Socket inode not found"))?;
    let sock = format!("socket:[{}]", socket_inode);
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
                        if target == PathBuf::from(&sock) {
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
