#[cfg(test)]
mod integration_test;

mod test_setup;
use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::process::Output;
use std::process::Stdio;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;
use serde_json::json;
pub use test_setup::init_test_logging;

/// Get UUID of the GPU at the specified index using nvidia-smi
pub fn get_gpu_uuid(gpu_index: usize) -> anyhow::Result<String> {
    let nvml = test_setup::global_nvml();
    let dev = nvml.device_by_index(gpu_index as u32)?;
    Ok(dev.uuid()?)
}

pub fn get_gpu_utilization(
    nvml: &Nvml,
    gpu_index: usize,
    pid: u32,
    last_seen_timestamp: u64,
) -> anyhow::Result<(u32, u64)> {
    // Get the device by index
    let device = nvml.device_by_index(gpu_index as u32)?;

    // Get process utilization samples
    let process_utilization_samples = match device.process_utilization_stats(last_seen_timestamp) {
        Ok(samples) => samples,
        Err(NvmlError::NotFound) => return Ok((0, last_seen_timestamp)),
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to get process utilization stats: {e}"
            ))
        }
    };

    let mut util = 0;
    let mut count = 0;
    let mut newest_timestamp_candidate = last_seen_timestamp;
    // Find the sample for the requested PID
    for sample in process_utilization_samples {
        // Skip old samples
        if sample.timestamp < last_seen_timestamp {
            continue;
        }

        if sample.timestamp > newest_timestamp_candidate {
            newest_timestamp_candidate = sample.timestamp;
        }

        if sample.pid == pid && sample.sm_util <= 100 {
            // Sum the different utilization components
            util += sample.sm_util;
            count += 1;
        }
    }

    if count == 0 {
        Ok((0, newest_timestamp_candidate))
    } else {
        Ok((util / count, newest_timestamp_candidate))
    }
}

pub fn run_cuda_test_program(
    memory_bytes: u64,
    duration_seconds: u64,
    gpu_index: usize,
    // (utilization_limit, memory_limit)
    limit: Option<(u64, u64)>,
) -> anyhow::Result<(u32, impl FnOnce() -> anyhow::Result<Output>)> {
    let cuda_program_path = env!("CUDA_TEST_PROGRAM_PATH");
    let mut cmd = Command::new(cuda_program_path);
    cmd.arg(memory_bytes.to_string())
        .arg(duration_seconds.to_string())
        .arg(gpu_index.to_string());

    // If using limiter, set up LD_PRELOAD
    if let Some((utilization_limit, memory_limit)) = limit {
        let limiter_path = get_limiter_library_path()?;
        cmd.env("LD_PRELOAD", &limiter_path);

        // Convert UUID to lowercase for the test
        let uuid_lowercase = get_gpu_uuid(gpu_index)?.to_lowercase();
        // Set memory limit as JSON
        cmd.env(
            "TENSOR_FUSION_CUDA_MEM_LIMIT",
            json!({
                &uuid_lowercase: memory_limit,
            })
            .to_string(),
        );

        // Set utilization limit as JSON
        cmd.env(
            "TENSOR_FUSION_CUDA_UP_LIMIT",
            json!({
                &uuid_lowercase: utilization_limit,
            })
            .to_string(),
        );
    }

    // Set up stdio
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    // Spawn the process
    let child = cmd
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to execute CUDA test program: {e}"))?;

    let child_id = child.id();
    let wait_fn = move || {
        // Wait for the process and get output
        let output = child
            .wait_with_output()
            .map_err(|e| anyhow::anyhow!("Failed to get output from CUDA test program: {e}"))?;

        Ok(output)
    };

    Ok((child_id, wait_fn))
}

/// Get the path to the compiled cuda-limiter library
fn get_limiter_library_path() -> anyhow::Result<PathBuf> {
    let bin_dir = env::current_dir()
        .map_err(|e| anyhow::anyhow!("Failed to get current directory: {e}"))?
        .join("../../target/debug");

    // Look for the library file (platform-dependent extension)
    let lib_name = "libcuda_limiter.so";

    let lib_path = bin_dir.join(lib_name);
    if !lib_path.exists() {
        return Err(anyhow::anyhow!(
            "Limiter library not found at {lib_path:?}. Did you build it?"
        ));
    }

    Ok(lib_path)
}

/// Monitor GPU metrics (utilization and memory) for a given duration
pub fn monitor_gpu_metrics(
    nvml: &Nvml,
    pid: u32,
    gpu_index: usize,
    duration_seconds: u64,
    sample_interval_ms: u64,
) -> anyhow::Result<Vec<u32>> {
    let mut metrics = Vec::new();
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(duration_seconds);

    let mut last_seen_timestamp = 0;

    while Instant::now() < end_time {
        let (utilization, newest_timestamp) =
            get_gpu_utilization(nvml, gpu_index, pid, last_seen_timestamp)?;
        metrics.push(utilization);
        last_seen_timestamp = newest_timestamp;
        // Sleep for the sample interval
        thread::sleep(Duration::from_millis(sample_interval_ms));
    }

    Ok(metrics)
}

/// Check if CUDA is available on the system
pub fn is_cuda_available() -> bool {
    Command::new("nvidia-smi").output().is_ok()
}

pub fn calculate_avg_utilization(metrics: &[u32]) -> u32 {
    if metrics.is_empty() {
        return 0;
    }

    let sum: u32 = metrics.iter().sum();
    sum / metrics.len() as u32
}
