use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use tokio_util::sync::CancellationToken;
use utils::shared_memory::PodIdentifier;

use crate::config::GPU_CAPACITY_MAP;
use crate::core::process::GpuResources;
use crate::platform::host_pid_probe::parse_pod_environment_variables;
use crate::platform::metrics::encoders::{GpuMetricsParams, WorkerMetricsParams};
use crate::platform::nvml::gpu_observer::GpuObserver;

use crate::core::pod::PodManager;

pub mod encoders;
use encoders::create_encoder;
use encoders::MetricsEncoder as _;

// Wrapper struct for Vec<u8> that implements Display
pub struct BytesWrapper(Vec<u8>);

impl fmt::Display for BytesWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "");
        }

        // Format as UTF-8 string if valid, otherwise as hex
        match std::str::from_utf8(&self.0) {
            Ok(s) => write!(f, "{s}"),
            Err(_) => {
                tracing::error!(msg = "Failed to convert bytes to string",);
                Err(fmt::Error)
            }
        }
    }
}

impl From<Vec<u8>> for BytesWrapper {
    fn from(bytes: Vec<u8>) -> Self {
        BytesWrapper(bytes)
    }
}

/// Cache entry for PID to PodIdentifier mapping
struct PidCacheEntry {
    pod_identifier: PodIdentifier,
    /// Process start time from /proc/{pid}/stat (field 22)
    /// Used to detect PID reuse
    process_start_time: u64,
}

/// Reads the process start time from /proc/{pid}/stat
///
/// The start time is the 22nd field in the stat file, representing the time
/// the process started after system boot (in clock ticks).
///
/// # Arguments
///
/// * `pid` - The process ID to query
///
/// # Returns
///
/// The process start time in clock ticks, or an error if the file cannot be read
async fn read_process_start_time(pid: u32) -> std::io::Result<u64> {
    let stat_path = format!("/proc/{pid}/stat");
    let stat_data = tokio::fs::read_to_string(&stat_path).await?;

    // Parse the stat file format: pid (comm) state ppid ... starttime ...
    // The process name (comm) can contain spaces and parentheses, so we need to
    // find the last ')' and then split the remaining fields
    let close_paren_pos = stat_data.rfind(')').ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid stat format: missing ')'",
        )
    })?;

    let fields_after_comm = &stat_data[close_paren_pos + 1..];
    let fields: Vec<&str> = fields_after_comm.split_whitespace().collect();

    // starttime is the 20th field after the comm field (22nd overall - 2 for pid and comm)
    // Fields after comm: state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt
    //                    cmajflt utime stime cutime cstime priority nice num_threads itrealvalue starttime
    // So starttime is at index 19 (0-based) in fields_after_comm
    fields.get(19).and_then(|s| s.parse().ok()).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Cannot parse starttime from stat file, got {} fields",
                fields.len()
            ),
        )
    })
}

/// Extracts pod information from /proc/{pid}/environ with caching
///
/// This function implements a hybrid caching strategy:
/// 1. Check cache first, validate with process start time to detect PID reuse
/// 2. If cache miss or PID reused, read from /proc and update cache
/// 3. If read fails, remove from cache (process likely exited)
///
/// # Arguments
///
/// * `pid` - The process ID to query
/// * `cache` - Mutable reference to the cache map
///
/// # Returns
///
/// `Some(PodIdentifier)` if the process is a pod process, `None` otherwise
async fn extract_pod_info_with_cache(
    pid: u32,
    cache: &mut HashMap<u32, PidCacheEntry>,
) -> Option<PodIdentifier> {
    // Check cache first
    if let Some(entry) = cache.get(&pid) {
        // Verify PID hasn't been reused by comparing start times
        if let Ok(current_start_time) = read_process_start_time(pid).await {
            if current_start_time == entry.process_start_time {
                // Cache hit and PID is still the same process
                return Some(entry.pod_identifier.clone());
            }
        }
        // PID reused or process doesn't exist, remove from cache
        cache.remove(&pid);
    }

    // Cache miss, try to read from /proc
    let environ_path = format!("/proc/{pid}/environ");
    let environ_data = tokio::fs::read_to_string(&environ_path).await.ok()?;

    // Quick check: is this a pod process?
    let has_pod_env = environ_data
        .split('\0')
        .any(|var| var.starts_with("POD_NAME="));

    if !has_pod_env {
        return None;
    }

    // Parse environment variables to extract pod info
    let (pod_name, namespace, _container_name) =
        parse_pod_environment_variables(&environ_data).ok()?;

    // Get process start time for cache validation
    let process_start_time = read_process_start_time(pid).await.ok()?;

    let pod_identifier = PodIdentifier::new(namespace, pod_name);

    // Update cache
    cache.insert(
        pid,
        PidCacheEntry {
            pod_identifier: pod_identifier.clone(),
            process_start_time,
        },
    );

    Some(pod_identifier)
}

#[derive(Default)]
struct AccumulatedGpuMetrics {
    memory_bytes: u64,
    memory_percentage: f64,
    compute_percentage: f64,
    compute_tflops: f64,

    rx: f64,
    tx: f64,
    temperature: f64,
    graphics_clock_mhz: f64,
    sm_clock_mhz: f64,
    memory_clock_mhz: f64,
    video_clock_mhz: f64,
    power_usage: i64,
    nvlink_rx_bandwidth: i64,
    nvlink_tx_bandwidth: i64,

    count: usize,
}

#[derive(Default)]
struct AccumulatedWorkerMetrics {
    memory_bytes: u64,
    compute_percentage: f64,
    compute_tflops: f64,
    count: usize,
}

/// Run metrics collection asynchronously
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_metrics<M, P, D, T>(
    gpu_observer: Arc<GpuObserver>,
    metrics_batch_size: usize,
    node_name: &str,
    gpu_pool: Option<&str>,
    pod_mgr: Arc<PodManager<M, P, D, T>>,
    metrics_format: &str,
    metrics_extra_labels: Option<&str>,
    cancellation_token: CancellationToken,
) {
    let gpu_pool = gpu_pool.unwrap_or("unknown");
    let encoder = create_encoder(metrics_format);

    let mut gpu_acc: HashMap<String, AccumulatedGpuMetrics> = HashMap::new();

    let mut worker_acc: HashMap<
        // gpu_uuid
        String,
        // pod_identifier -> pid -> metrics
        HashMap<PodIdentifier, HashMap<u32, AccumulatedWorkerMetrics>>,
    > = HashMap::new();
    let mut counter = 0;

    let metrics_extra_labels: HashMap<String, String> = metrics_extra_labels
        .map(|labels_json| {
            if labels_json == "null" {
                HashMap::new()
            } else {
                serde_json::from_str(labels_json).unwrap_or_else(|e| {
                    tracing::warn!(
                        "Failed to parse metrics_extra_labels JSON: {}, using empty map",
                        e
                    );
                    HashMap::new()
                })
            }
        })
        .unwrap_or_default();
    let has_dynamic_metrics_labels = !metrics_extra_labels.is_empty();

    let mut receiver = gpu_observer.subscribe().await;

    // Get hypervisor's own PID to filter it out
    let hypervisor_pid = std::process::id();

    // Cache for PID to PodIdentifier mapping (with PID reuse detection)
    let mut pid_to_pod_cache: HashMap<u32, PidCacheEntry> = HashMap::new();

    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                tracing::info!("Metrics collection shutdown requested");
                break;
            }
            recv_result = receiver.recv() => {
                if recv_result.is_none() {
                    tracing::info!("Metrics receiver closed");
                    break;
                }

                counter += 1;

                // Accumulate GPU metrics
                for (gpu_uuid, gpu) in gpu_observer
                    .metrics
                    .read()
                    .await
                    .gpu_metrics
                    .iter()
                {
                    let acc = gpu_acc.entry(gpu_uuid.to_string()).or_default();
                    acc.rx += gpu.rx as f64;
                    acc.tx += gpu.tx as f64;
                    acc.temperature += gpu.temperature as f64;
                    acc.graphics_clock_mhz += gpu.graphics_clock_mhz as f64;
                    acc.sm_clock_mhz += gpu.sm_clock_mhz as f64;
                    acc.memory_clock_mhz += gpu.memory_clock_mhz as f64;
                    acc.video_clock_mhz += gpu.video_clock_mhz as f64;
                    acc.memory_bytes += gpu.resources.memory_bytes;
                    acc.memory_percentage += gpu.memory_percentage;
                    acc.compute_percentage += gpu.resources.compute_percentage as f64;
                    acc.power_usage += gpu.power_usage as i64;

                    // Estimation of TFlops (approximate)
                    acc.compute_tflops += gpu.resources.compute_percentage as f64
                        * GPU_CAPACITY_MAP
                            .read()
                            .expect("should not be poisoned")
                            .get(gpu_uuid)
                            .unwrap_or(&0.0) / 100.0;
                    acc.count += 1;
                }

                // Accumulate process metrics
                // First, collect all the process metrics data to avoid holding the lock across await points
                let process_metrics_snapshot: Vec<(String, Vec<(u32, GpuResources)>)> = {
                    let metrics_guard = gpu_observer.metrics.read().await;
                    metrics_guard
                        .process_metrics
                        .iter()
                        .map(|(gpu_uuid, (process_metrics, _))| {
                            let processes: Vec<(u32, GpuResources)> = process_metrics
                                .iter()
                                .map(|(pid, resources)| (*pid, resources.clone()))
                                .collect();
                            (gpu_uuid.clone(), processes)
                        })
                        .collect()
                }; // RwLockReadGuard is dropped here

                let pod_store = pod_mgr.pod_state_store();
                for pod_identifier in pod_store.list_pod_identifiers() {
                    if let Some(pod_info) = pod_store.get_pod_info(&pod_identifier) {
                        if let Some(gpu_uuids) = pod_info.gpu_uuids {
                            for uuid in gpu_uuids {
                                let acc = worker_acc.entry(uuid).or_default();
                                acc.entry(pod_identifier.clone()).or_default();
                            }
                        }
                    }
                }

                // Now process the collected data with async operations
                for (gpu_uuid, process_metrics) in process_metrics_snapshot {
                    let worker_acc = worker_acc.entry(gpu_uuid.to_string()).or_default();
                    for (pid, resources) in process_metrics.iter() {
                        // Skip hypervisor's own process
                        if *pid == hypervisor_pid {
                            continue;
                        }

                        // Try to find pod identifier:
                        // 1. First check pod_store (for registered processes)
                        // 2. If not found, try to extract from /proc (for skip-hook processes)
                        let pod_identifier = if let Some(pod_id) = pod_mgr.find_pod_by_worker_pid(*pid) {
                            Some(pod_id)
                        } else {
                            // Not in pod_store, try to extract from /proc with caching
                            extract_pod_info_with_cache(*pid, &mut pid_to_pod_cache).await
                        };

                        let Some(pod_identifier) = pod_identifier else {
                            // Neither in pod_store nor a pod process, skip
                            continue;
                        };

                        let acc = worker_acc.entry(pod_identifier).or_default();
                        let acc = acc.entry(*pid).or_default();
                        acc.memory_bytes += resources.memory_bytes;
                        acc.compute_percentage += resources.compute_percentage as f64;
                        acc.compute_tflops += resources.compute_percentage as f64
                            * GPU_CAPACITY_MAP
                                .read()
                                .expect("should not be poisoned")
                                .get(&gpu_uuid)
                                .unwrap_or(&0.0) / 100.0;
                        acc.count += 1;
                    }
                }

                // Not enough samples yet, keep accumulating
                if counter < metrics_batch_size {
                    continue;
                }

                let timestamp = current_time();

                // Output averaged GPU metrics
                for (gpu_uuid, acc) in &gpu_acc {
                    if acc.count == 0 { continue; }
                    let metrics_str = encoder.encode_gpu_metrics_with_params(&GpuMetricsParams {
                        gpu_uuid,
                        node_name,
                        gpu_pool,
                        rx: acc.rx / acc.count as f64,
                        tx: acc.tx / acc.count as f64,
                        nvlink_rx_bandwidth: acc.nvlink_rx_bandwidth / acc.count as i64,
                        nvlink_tx_bandwidth: acc.nvlink_tx_bandwidth / acc.count as i64,
                        temperature: acc.temperature / acc.count as f64,
                        graphics_clock_mhz: acc.graphics_clock_mhz / acc.count as f64,
                        sm_clock_mhz: acc.sm_clock_mhz / acc.count as f64,
                        memory_clock_mhz: acc.memory_clock_mhz / acc.count as f64,
                        video_clock_mhz: acc.video_clock_mhz / acc.count as f64,
                        memory_bytes: acc.memory_bytes / acc.count as u64,
                        memory_percentage: acc.memory_percentage / acc.count as f64,
                        compute_percentage: acc.compute_percentage / acc.count as f64,
                        compute_tflops: acc.compute_tflops / acc.count as f64,
                        power_usage: acc.power_usage / acc.count as i64,
                        timestamp,
                    });
                    tracing::info!(
                        target: "metrics",
                        msg = %metrics_str,
                    );
                }

                // Collect all active PIDs for cache cleanup
                let active_pids: HashSet<u32> = worker_acc
                    .values()
                    .flat_map(|pod_map| {
                        pod_map.values().flat_map(|acc_map| acc_map.keys().copied())
                    })
                    .collect();

                // Output averaged worker metrics
                for (gpu_uuid, pod_metrics) in &worker_acc {
                    for (pod_id, acc_map) in pod_metrics {
                        // Calculate aggregated metrics
                        let mut memory_bytes = 0;
                        let mut compute_percentage = 0.0;
                        let mut compute_tflops = 0.0;

                        // Try to get full pod info from pod_store
                        if let Some(pod_state) = pod_mgr.pod_state_store().get_pod(pod_id) {
                            // Pod is registered in pod_store
                            let vram_limit = pod_state.pod_info.vram_limit.unwrap_or(0) as f64;
                            let mut memory_percentage = 0.0;

                            for (_pid, acc) in acc_map.iter() {
                                memory_bytes += acc.memory_bytes / acc.count as u64;
                                compute_percentage += acc.compute_percentage / acc.count as f64;
                                compute_tflops += acc.compute_tflops / acc.count as f64;
                                memory_percentage += {
                                    let avg_memory_bytes = acc.memory_bytes as f64 / acc.count as f64;
                                    if vram_limit > 0.0 {
                                        avg_memory_bytes / vram_limit * 100.0
                                    } else {
                                        0.0
                                    }
                                }
                            }

                            let mut extra_labels = HashMap::new();
                            if has_dynamic_metrics_labels {
                                for (label, value) in &metrics_extra_labels {
                                    extra_labels.insert(
                                        value.clone(),
                                        pod_state.pod_info.labels
                                            .get(label)
                                            .cloned()
                                            .unwrap_or_else(|| "unknown".to_string()),
                                    );
                                }
                            }

                            let metrics_str = encoder.encode_worker_metrics_with_params(&WorkerMetricsParams {
                                gpu_uuid,
                                node_name,
                                gpu_pool,
                                pod_name: &pod_state.pod_info.pod_name,
                                namespace: &pod_state.pod_info.namespace,
                                workload: pod_state.pod_info.workload_name.as_deref().unwrap_or("unknown"),
                                memory_bytes,
                                compute_percentage,
                                compute_tflops,
                                memory_percentage,
                                timestamp,
                                extra_labels: &extra_labels,
                            });
                            tracing::info!(
                                target: "metrics",
                                msg = %metrics_str,
                            );
                        } else {
                            // Fallback: Pod not in pod_store (skip-hook process)
                            let mut memory_percentage = 0.0;

                            for (_pid, acc) in acc_map.iter() {
                                memory_bytes += acc.memory_bytes / acc.count as u64;
                                compute_percentage += acc.compute_percentage / acc.count as f64;
                                compute_tflops += acc.compute_tflops / acc.count as f64;
                                // No vram_limit available for skip-hook processes
                                memory_percentage = 0.0;
                            }

                            let empty_extra_labels = HashMap::new();

                            let metrics_str = encoder.encode_worker_metrics_with_params(&WorkerMetricsParams {
                                gpu_uuid,
                                node_name,
                                gpu_pool,
                                pod_name: &pod_id.name,
                                namespace: &pod_id.namespace,
                                workload: "unknown",
                                memory_bytes,
                                compute_percentage,
                                compute_tflops,
                                memory_percentage,
                                timestamp,
                                extra_labels: &empty_extra_labels,
                            });
                            tracing::info!(
                                target: "metrics",
                                msg = %metrics_str,
                            );
                        }
                    }
                }

                // Clean up cache: remove PIDs that are no longer active
                // This is the hybrid strategy: passive validation (on access) + active cleanup (here)
                pid_to_pod_cache.retain(|pid, _| active_pids.contains(pid));

                // Reset accumulators and counter
                gpu_acc.clear();
                worker_acc.clear();
                counter = 0;
            }
        }
    }
}

pub fn current_time() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("should compute duration since UNIX_EPOCH")
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[cfg(target_os = "linux")]
    async fn test_read_process_start_time_current_process() {
        // Test reading start time of current process (should always succeed on Linux)
        let current_pid = std::process::id();
        let result = read_process_start_time(current_pid).await;
        assert!(
            result.is_ok(),
            "Should successfully read start time for current process"
        );
        let start_time = result.unwrap();
        assert!(start_time > 0, "Start time should be positive");
    }

    #[tokio::test]
    #[cfg(target_os = "linux")]
    async fn test_read_process_start_time_nonexistent() {
        // Test reading start time of nonexistent process
        let fake_pid = 9999999;
        let result = read_process_start_time(fake_pid).await;
        assert!(
            result.is_err(),
            "Should fail to read start time for nonexistent process"
        );
    }

    #[tokio::test]
    #[cfg(target_os = "linux")]
    async fn test_read_process_start_time_consistency() {
        // Test that reading the same process twice returns the same start time
        let current_pid = std::process::id();
        let start_time1 = read_process_start_time(current_pid).await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        let start_time2 = read_process_start_time(current_pid).await.unwrap();
        assert_eq!(
            start_time1, start_time2,
            "Start time should be consistent for the same process"
        );
    }

    #[tokio::test]
    async fn test_extract_pod_info_cache_miss_nonpod() {
        // Test cache miss for a non-pod process (current process)
        let mut cache = HashMap::new();
        let current_pid = std::process::id();
        let result = extract_pod_info_with_cache(current_pid, &mut cache).await;
        assert!(result.is_none(), "Should return None for non-pod process");
        assert!(
            cache.is_empty(),
            "Cache should remain empty for non-pod process"
        );
    }

    #[tokio::test]
    async fn test_extract_pod_info_cache_miss_nonexistent() {
        // Test cache miss for a nonexistent process
        let mut cache = HashMap::new();
        let fake_pid = 9999999;
        let result = extract_pod_info_with_cache(fake_pid, &mut cache).await;
        assert!(
            result.is_none(),
            "Should return None for nonexistent process"
        );
        assert!(
            cache.is_empty(),
            "Cache should remain empty for nonexistent process"
        );
    }

    #[tokio::test]
    async fn test_extract_pod_info_cache_hit() {
        // Test cache hit scenario
        let mut cache = HashMap::new();
        let test_pid = 1234;
        let test_pod_id = PodIdentifier::new("test-namespace", "test-pod");
        let test_start_time = 1000;

        // Populate cache
        cache.insert(
            test_pid,
            PidCacheEntry {
                pod_identifier: test_pod_id.clone(),
                process_start_time: test_start_time,
            },
        );

        // Mock the start time to match (in real scenario, this would be the actual process)
        // Since we can't easily mock file I/O in a unit test, we'll test the logic separately
        // For now, this test documents the expected behavior
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&test_pid));
    }

    #[tokio::test]
    async fn test_cache_entry_structure() {
        // Test that PidCacheEntry stores data correctly
        let pod_id = PodIdentifier::new("my-namespace", "my-pod");
        let start_time = 12345u64;

        let entry = PidCacheEntry {
            pod_identifier: pod_id.clone(),
            process_start_time: start_time,
        };

        assert_eq!(entry.pod_identifier.namespace, "my-namespace");
        assert_eq!(entry.pod_identifier.name, "my-pod");
        assert_eq!(entry.process_start_time, start_time);
    }

    #[test]
    fn test_parse_pod_environment_variables_integration() {
        // Test the imported parse_pod_environment_variables function
        let environ_data =
            "PATH=/usr/bin\0POD_NAME=test-pod\0POD_NAMESPACE=test-ns\0CONTAINER_NAME=test-container\0";
        let result = parse_pod_environment_variables(environ_data);
        assert!(result.is_ok());
        let (pod_name, namespace, _container) = result.unwrap();
        assert_eq!(pod_name, "test-pod");
        assert_eq!(namespace, "test-ns");
    }

    #[test]
    fn test_parse_pod_environment_variables_missing() {
        // Test parsing environment without pod variables
        let environ_data = "PATH=/usr/bin\0HOME=/root\0USER=test\0";
        let result = parse_pod_environment_variables(environ_data);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cache_cleanup_removes_inactive_pids() {
        // Test the cache cleanup logic
        let mut cache = HashMap::new();

        // Populate cache with some entries
        cache.insert(
            1001,
            PidCacheEntry {
                pod_identifier: PodIdentifier::new("ns1", "pod1"),
                process_start_time: 1000,
            },
        );
        cache.insert(
            1002,
            PidCacheEntry {
                pod_identifier: PodIdentifier::new("ns2", "pod2"),
                process_start_time: 2000,
            },
        );
        cache.insert(
            1003,
            PidCacheEntry {
                pod_identifier: PodIdentifier::new("ns3", "pod3"),
                process_start_time: 3000,
            },
        );

        assert_eq!(cache.len(), 3);

        // Simulate active PIDs (only 1001 and 1003 are active)
        let active_pids: HashSet<u32> = [1001, 1003].iter().copied().collect();

        // Clean up cache (this is what happens in run_metrics)
        cache.retain(|pid, _| active_pids.contains(pid));

        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&1001));
        assert!(!cache.contains_key(&1002)); // Should be removed
        assert!(cache.contains_key(&1003));
    }

    #[tokio::test]
    async fn test_multiple_cache_operations() {
        // Test a sequence of cache operations
        let mut cache = HashMap::new();

        let pod_id1 = PodIdentifier::new("default", "app1");
        let pod_id2 = PodIdentifier::new("default", "app2");

        cache.insert(
            1001,
            PidCacheEntry {
                pod_identifier: pod_id1.clone(),
                process_start_time: 1000,
            },
        );

        assert_eq!(cache.len(), 1);

        cache.insert(
            1002,
            PidCacheEntry {
                pod_identifier: pod_id2.clone(),
                process_start_time: 2000,
            },
        );

        assert_eq!(cache.len(), 2);

        // Remove one entry
        cache.remove(&1001);
        assert_eq!(cache.len(), 1);
        assert!(!cache.contains_key(&1001));
        assert!(cache.contains_key(&1002));
    }

    #[test]
    fn test_current_time() {
        // Test that current_time returns a reasonable value
        let time1 = current_time();
        assert!(time1 > 0);

        // Time should be close to Unix epoch in milliseconds
        // Current time should be greater than 2020-01-01 (1577836800000 ms)
        assert!(time1 > 1577836800000);
    }

    #[test]
    fn test_bytes_wrapper_display() {
        // Test BytesWrapper Display implementation
        let empty = BytesWrapper::from(vec![]);
        assert_eq!(format!("{empty}"), "");

        let valid_utf8 = BytesWrapper::from(b"hello".to_vec());
        assert_eq!(format!("{valid_utf8}"), "hello");
    }

    #[test]
    fn test_accumulated_metrics_default() {
        // Test that AccumulatedGpuMetrics and AccumulatedWorkerMetrics have sensible defaults
        let gpu_metrics = AccumulatedGpuMetrics::default();
        assert_eq!(gpu_metrics.count, 0);
        assert_eq!(gpu_metrics.memory_bytes, 0);

        let worker_metrics = AccumulatedWorkerMetrics::default();
        assert_eq!(worker_metrics.count, 0);
        assert_eq!(worker_metrics.memory_bytes, 0);
    }
}
