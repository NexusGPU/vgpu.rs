use crate::gpu_observer::GpuObserver;
use crate::hypervisor::Hypervisor;
use crate::process::worker::TensorFusionWorker;
use crate::process::{GpuResources, QosLevel};
use crate::scheduler::GpuScheduler;
use notify::{Error, Event, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use std::{fs, io, thread};

pub(crate) struct WorkerWatcher<Sched: GpuScheduler<TensorFusionWorker>> {
    rx: Mutex<Receiver<Result<Event, Error>>>,
    tx: Sender<Result<Event, Error>>,
    hypervisor: Arc<Hypervisor<TensorFusionWorker, Sched>>,
    worker_pid_mapping: Arc<RwLock<HashMap<u32, String>>>,
    #[cfg(target_os = "linux")]
    _watcher: notify::INotifyWatcher,
}

impl<Sched: GpuScheduler<TensorFusionWorker>> WorkerWatcher<Sched> {
    pub(crate) fn new<P: AsRef<Path>>(
        path: P,
        hypervisor: Arc<Hypervisor<TensorFusionWorker, Sched>>,
        worker_pid_mapping: Arc<RwLock<HashMap<u32, String>>>,
    ) -> Result<Self, Error> {
        let (tx, rx) = mpsc::channel::<Result<Event, Error>>();

        #[cfg(target_os = "linux")]
        let watcher = {
            // Create a watcher using the recommended watcher implementation for the current platform
            let mut watcher = notify::recommended_watcher(tx.clone())?;
            watcher.watch(path.as_ref(), notify::RecursiveMode::NonRecursive)?;
            watcher
        };
        tracing::info!("watching worker sock files at: {:?}", path.as_ref());
        Ok(WorkerWatcher {
            rx: Mutex::new(rx),
            tx,
            hypervisor,
            worker_pid_mapping,
            #[cfg(target_os = "linux")]
            _watcher: watcher,
        })
    }

    /// Run the worker watcher loop that periodically checks the directory for new files
    /// This is meant to be called from a dedicated thread in the crossbeam scope
    pub(crate) fn run_watcher_loop(&self, path: impl AsRef<Path>) {
        let tx = self.tx.clone();
        let path = PathBuf::from(path.as_ref());

        loop {
            // Read directory contents
            let entries = match fs::read_dir(&path) {
                Ok(entries) => entries,
                Err(e) => {
                    tracing::error!("failed to read directory: {:?}", e);
                    thread::sleep(Duration::from_secs(3));
                    continue;
                }
            };

            // Process directory entries
            let entries = entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .collect::<Vec<_>>();

            // Send events for each entry
            for entry in entries {
                let event = Event::new(notify::event::EventKind::Create(
                    notify::event::CreateKind::File,
                ))
                .add_path(entry);
                let _ = tx.send(Ok(event));
            }

            // Sleep before next check
            thread::sleep(Duration::from_secs(3));
        }
    }

    pub(crate) fn run(&self, gpu_observer: Arc<GpuObserver>) {
        let rx = self.rx.lock().expect("Failed to lock receiver");
        for res in rx.iter() {
            match res {
                Ok(event) => match event.kind {
                    notify::EventKind::Create(_) => {
                        if let Some(path) = event.paths.first() {
                            let (pid, worker_name) = match extract_pid_worker_name_from_path(path) {
                                Ok(pid_worker_name) => pid_worker_name,
                                Err(msg) => {
                                    tracing::warn!("{}", msg);
                                    // remove invalid pid file
                                    if let Err(e) = fs::remove_file(path) {
                                        tracing::warn!(
                                            "cannot remove invalid pid file: {:?}, err: {:?} skipped",
                                            path,
                                            e
                                        );
                                    }
                                    tracing::info!("removed invalid pid file: {:?}", path);
                                    continue;
                                }
                            };

                            let env = match read_process_env_vars(pid) {
                                Ok(env) => env,
                                Err(e) => {
                                    if e.kind() == io::ErrorKind::NotFound {
                                        // remove orphan pid file
                                        if let Err(e) = fs::remove_file(path) {
                                            tracing::warn!(
                                                "cannot remove orphan pid file: {:?}, err: {:?} skipped",
                                                path,
                                                e
                                            );
                                        }
                                        tracing::info!("removed orphan pid file: {:?}", path);
                                    } else {
                                        tracing::warn!(
                                            "cannot read env vars for worker: {:?}, err: {:?} skipped",
                                            path,
                                            e
                                        );
                                    }
                                    continue;
                                }
                            };

                            // Get GPU UUID
                            let uuid = if let Some(uuid) = env.get("NVIDIA_VISIBLE_DEVICES") {
                                uuid.clone()
                            } else {
                                tracing::warn!(
                                    "no visible device for worker: {:?}, skipped",
                                    path
                                );
                                continue;
                            };

                            if self.hypervisor.process_exists(pid) {
                                continue;
                            }

                            // Get QoS level
                            let qos_level = match env.get("TENSOR_FUSION_QOS_LEVEL").map(String::as_str) {
                                Some("HIGH") | Some("high") => QosLevel::High,
                                Some("LOW") | Some("low") => QosLevel::Low,
                                _ => QosLevel::Medium,
                            };

                            let worker = TensorFusionWorker::new(
                                pid,
                                path.clone(),
                                GpuResources {
                                    memory_bytes: 0,
                                    compute_percentage: 0,
                                },
                                qos_level,
                                uuid,
                                gpu_observer.clone(),
                            );

                            tracing::info!("new worker added: {:?}", worker_name);
                            self.worker_pid_mapping
                                .write()
                                .expect("poisoning")
                                .insert(pid, worker_name);
                            self.hypervisor.add_process(worker);
                        }
                    }
                    notify::EventKind::Remove(_) => {
                        tracing::info!("worker sock file removed: {:?}", event.paths);
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
        .file_name()
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

/// Read environment variables from a process by its PID
/// 
/// This function reads from /proc/{pid}/environ to get the process environment variables.
/// Important environment variables:
/// - NVIDIA_VISIBLE_DEVICES: Required. Specifies the GPU UUID for the worker.
/// - TENSOR_FUSION_QOS_LEVEL: Optional. Sets the QoS level for the worker.
///   Possible values: "HIGH", "LOW" (case insensitive). Defaults to "MEDIUM" if not set.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_extract_pid_worker_name_from_path() {
        // Define test cases for extracting PID and worker name from path
        let cases = [
            // Test case: Valid PID
            (
                "/some/path/12345",
                true,
                Some((12345, "12345".to_string())),
                None,
                "Valid PID",
            ),
            // Test case: Invalid PID (non-numeric filename)
            (
                "/some/path/not_a_number",
                false,
                None,
                Some("failed to parse PID"),
                "Invalid PID (non-numeric filename)",
            ),
            // Test case: No filename in path
            (
                "/some/path/",
                false,
                None,
                Some("failed to parse PID"), // This path also produces a parse error
                "No filename in path",
            ),
            // Test case: Empty path
            ("", false, None, Some("could not extract PID"), "Empty path"),
            // Test case: Edge case - PID is 0
            (
                "/some/path/0",
                true,
                Some((0, "0".to_string())),
                None,
                "Edge case - PID is 0",
            ),
            // Test case: Edge case - Maximum u32 value
            (
                &format!("/some/path/{}", u32::MAX),
                true,
                Some((u32::MAX, u32::MAX.to_string())),
                None,
                "Edge case - Maximum u32 value",
            ),
        ];

        // Run all test cases
        for (path_str, should_succeed, expected_result, error_fragment, description) in cases {
            println!("Testing case: {}", description);
            let path = PathBuf::from(path_str);
            let result = extract_pid_worker_name_from_path(&path);

            if should_succeed {
                assert!(
                    result.is_ok(),
                    "Case '{}' failed: expected Ok but got Err: {}",
                    description,
                    result.unwrap_err()
                );

                if let Some((expected_pid, expected_worker_name)) = expected_result {
                    let (pid, worker_name) = result.unwrap();
                    assert_eq!(
                        pid, expected_pid,
                        "Case '{}' failed: PID mismatch",
                        description
                    );
                    assert_eq!(
                        worker_name, expected_worker_name,
                        "Case '{}' failed: worker name mismatch",
                        description
                    );
                }
            } else {
                assert!(
                    result.is_err(),
                    "Case '{}' failed: expected Err but got Ok: {:?}",
                    description,
                    result.unwrap()
                );

                if let Some(fragment) = error_fragment {
                    let actual_error = result.unwrap_err();
                    assert!(
                        actual_error.contains(fragment),
                        "Case '{}' failed: error message '{}' doesn't contain expected text '{}'",
                        description,
                        actual_error,
                        fragment
                    );
                }
            }
        }
    }
}
