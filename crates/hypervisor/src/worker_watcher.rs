use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::thread;
use std::time::Duration;

use notify::Error;
use notify::Event;
use notify::Watcher;

use crate::gpu_observer::GpuObserver;
use crate::process::worker::TensorFusionWorker;
use crate::process::GpuResources;
use crate::process::QosLevel;

pub(crate) struct WorkerWatcher<AddCB, RemoveCB> {
    rx: Mutex<Receiver<Result<Event, Error>>>,
    tx: Sender<Result<Event, Error>>,
    add_callback: AddCB,
    remove_callback: RemoveCB,
    worker_pid_mapping: Arc<RwLock<HashMap<u32, (String, String)>>>,
    #[cfg(target_os = "linux")]
    _watcher: notify::INotifyWatcher,
}

impl<AddCB: Fn(u32, TensorFusionWorker), RemoveCB: Fn(u32)> WorkerWatcher<AddCB, RemoveCB> {
    pub(crate) fn new<P: AsRef<Path>>(
        path: P,
        add_callback: AddCB,
        remove_callback: RemoveCB,
        worker_pid_mapping: Arc<RwLock<HashMap<u32, (String, String)>>>,
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
            add_callback,
            remove_callback,
            worker_pid_mapping,
            #[cfg(target_os = "linux")]
            _watcher: watcher,
        })
    }

    /// Run the worker watcher loop that periodically checks the directory for new files
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
            let entries = entries.filter_map(Result::ok).map(|entry| entry.path());
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
                            let pid = match extract_pid_from_path(path) {
                                Ok(pid) => pid,
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

                            let worker_name = env
                                .get("POD_NAME")
                                .cloned()
                                .unwrap_or_else(|| String::from("unknown"));
                            let workload_name = env
                                .get("TENSOR_FUSION_WORKLOAD_NAME")
                                .cloned()
                                .unwrap_or_else(|| String::from("unknown"));

                            // Get GPU UUID
                            let uuid = if let Some(uuid) = env.get("NVIDIA_VISIBLE_DEVICES") {
                                uuid.clone()
                            } else {
                                tracing::warn!("no visible device for worker: {:?}, skipped", path);
                                continue;
                            };

                            // Get QoS level
                            let qos_level =
                                match env.get("TENSOR_FUSION_QOS_LEVEL").map(String::as_str) {
                                    Some("High") | Some("high") => QosLevel::High,
                                    Some("Low") | Some("low") => QosLevel::Low,
                                    Some("Medium") | Some("medium") => QosLevel::Medium,
                                    Some("Critical") | Some("critical") => QosLevel::Critical,
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

                            self.worker_pid_mapping
                                .write()
                                .expect("poisoning")
                                .insert(pid, (worker_name, workload_name));
                            (self.add_callback)(pid, worker);
                        }
                    }
                    notify::EventKind::Remove(_) => {
                        tracing::info!("worker sock file removed: {:?}", event.paths);
                        if let Some(path) = event.paths.first() {
                            let pid = match extract_pid_from_path(path) {
                                Ok(pid_worker_name) => pid_worker_name,
                                Err(msg) => {
                                    tracing::warn!("{}", msg);
                                    continue;
                                }
                            };
                            (self.remove_callback)(pid)
                        }
                    }
                    _ => {}
                },
                Err(e) => tracing::error!("watch error: {:?}", e),
            }
        }
    }
}

fn extract_pid_from_path(path: &std::path::Path) -> Result<u32, String> {
    // Extract PID from filename
    let pid = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("could not extract PID from path: {path:?}, skipped"))
        .and_then(|s| {
            s.parse::<u32>().map_err(|e| {
                format!("failed to parse PID from path: {path:?}, error: {e:?}, skipped")
            })
        })?;

    Ok(pid)
}

/// Read environment variables from a process by its PID
///
/// This function reads from /proc/{pid}/environ to get the process environment variables.
/// Important environment variables:
/// - NVIDIA_VISIBLE_DEVICES: Required. Specifies the GPU UUID for the worker.
/// - TENSOR_FUSION_QOS_LEVEL: Optional. Sets the QoS level for the worker.
///   Possible values: "HIGH", "LOW" (case insensitive). Defaults to "MEDIUM" if not set.
fn read_process_env_vars(pid: u32) -> Result<HashMap<String, String>, io::Error> {
    let environ_path = format!("/proc/{pid}/environ");
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
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_extract_pid_worker_name_from_path() {
        // Define test cases for extracting PID and worker name from path
        let cases = [
            // Test case: Valid PID
            ("/some/path/12345", true, Some(12345), None, "Valid PID"),
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
            ("/some/path/0", true, Some(0), None, "Edge case - PID is 0"),
            // Test case: Edge case - Maximum u32 value
            (
                &format!("/some/path/{}", u32::MAX),
                true,
                Some(u32::MAX),
                None,
                "Edge case - Maximum u32 value",
            ),
        ];

        // Run all test cases
        for (path_str, should_succeed, expected_result, error_fragment, description) in cases {
            println!("Testing case: {}", description);
            let path = PathBuf::from(path_str);
            let result = extract_pid_from_path(&path);

            if should_succeed {
                assert!(
                    result.is_ok(),
                    "Case '{}' failed: expected Ok but got Err: {}",
                    description,
                    result.unwrap_err()
                );

                if let Some(expected_pid) = expected_result {
                    let pid = result.unwrap();
                    assert_eq!(
                        pid, expected_pid,
                        "Case '{}' failed: PID mismatch",
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
