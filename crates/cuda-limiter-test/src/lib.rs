#[cfg(test)]
mod integration_test;

mod test_setup;
pub use test_setup::init_test_logging;

use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::process::Output;
use std::process::Stdio;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use serde_json::json;

/// Get UUID of the GPU at the specified index using nvidia-smi
pub fn get_gpu_uuid(gpu_index: usize) -> Result<String, String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=index,uuid", "--format=csv,noheader"])
        .output()
        .map_err(|e| format!("Failed to execute nvidia-smi: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let output_str = String::from_utf8_lossy(&output.stdout);

    // Parse the output to find the UUID for the requested GPU index
    for line in output_str.lines() {
        let parts: Vec<&str> = line.trim().split(',').collect();
        if parts.len() == 2 {
            let index_str = parts[0].trim();
            let uuid = parts[1].trim().to_string();

            if let Ok(index) = index_str.parse::<usize>() {
                if index == gpu_index {
                    return Ok(uuid);
                }
            }
        }
    }

    Err(format!("GPU with index {} not found", gpu_index))
}

/// Get current GPU utilization using nvidia-smi
pub fn get_gpu_utilization(gpu_index: usize) -> Result<u32, String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
            &format!("--id={}", gpu_index),
        ])
        .output()
        .map_err(|e| format!("Failed to execute nvidia-smi: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let output_str = String::from_utf8_lossy(&output.stdout);
    let utilization = output_str
        .trim()
        .parse::<u32>()
        .map_err(|e| format!("Failed to parse GPU utilization: {}", e))?;

    Ok(utilization)
}

/// Get current GPU memory usage using nvidia-smi
pub fn get_gpu_memory_usage(gpu_index: usize) -> Result<u64, String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            &format!("--id={}", gpu_index),
        ])
        .output()
        .map_err(|e| format!("Failed to execute nvidia-smi: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let output_str = String::from_utf8_lossy(&output.stdout);
    // nvidia-smi returns memory in MiB, convert to bytes
    let memory_mib = output_str
        .trim()
        .parse::<u64>()
        .map_err(|e| format!("Failed to parse GPU memory usage: {}", e))?;

    Ok(memory_mib * 1024 * 1024) // Convert MiB to bytes
}

/// Set environment variables for the CUDA limiter
pub fn set_limiter_env_vars(gpu_uuid: &str, mem_limit: u64, utilization_limit: u32) {
    // Convert UUID to lowercase for the test
    let uuid_lowercase = gpu_uuid.to_lowercase();

    // Set memory limit as JSON
    let mem_limit_json = json!({
        &uuid_lowercase: mem_limit,
    })
    .to_string();
    env::set_var("TENSOR_FUSION_CUDA_MEM_LIMIT", mem_limit_json);

    // Set utilization limit as JSON
    let up_limit_json = json!({
        &uuid_lowercase: utilization_limit,
    })
    .to_string();
    env::set_var("TENSOR_FUSION_CUDA_UP_LIMIT", up_limit_json);
}

/// Set environment variables with uppercase GPU UUID to test case-insensitivity
pub fn set_limiter_env_vars_uppercase(gpu_uuid: &str, mem_limit: u64, utilization_limit: u32) {
    // Convert UUID to uppercase to test case-insensitivity
    let uuid_uppercase = gpu_uuid.to_uppercase();

    // Set memory limit as JSON
    let mem_limit_json = json!({
        &uuid_uppercase: mem_limit,
    })
    .to_string();
    env::set_var("TENSOR_FUSION_CUDA_MEM_LIMIT", mem_limit_json);

    // Set utilization limit as JSON
    let up_limit_json = json!({
        &uuid_uppercase: utilization_limit,
    })
    .to_string();
    env::set_var("TENSOR_FUSION_CUDA_UP_LIMIT", up_limit_json);
}

/// Clear limiter environment variables
pub fn clear_limiter_env_vars() {
    env::remove_var("TENSOR_FUSION_CUDA_MEM_LIMIT");
    env::remove_var("TENSOR_FUSION_CUDA_UP_LIMIT");
}

/// Run the CUDA test program with or without the limiter
pub fn run_cuda_test_program(
    memory_bytes: u64,
    utilization_percent: u32,
    duration_seconds: u64,
    gpu_index: usize,
    use_limiter: bool,
) -> Result<Output, String> {
    // 获取CUDA程序路径 (从环境变量或查找编译后的二进制文件)
    let cuda_program_path = env!("CUDA_TEST_PROGRAM_PATH");

    // 设置命令
    let mut cmd = Command::new(cuda_program_path);

    // 我们的C语言CUDA测试程序使用位置参数而不是命名参数
    cmd.arg(memory_bytes.to_string())
        .arg(utilization_percent.to_string())
        .arg(duration_seconds.to_string())
        .arg(gpu_index.to_string());

    // If using limiter, set up LD_PRELOAD
    if use_limiter {
        let limiter_path = get_limiter_library_path()?;
        cmd.env("LD_PRELOAD", &limiter_path);
    }

    // Run the command
    let output = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to execute CUDA test program: {}", e))?;

    Ok(output)
}

/// Get the path to the compiled cuda-limiter library
fn get_limiter_library_path() -> Result<PathBuf, String> {
    let bin_dir = env::current_dir()
        .map_err(|e| format!("Failed to get current directory: {}", e))?
        .join("../../target/debug");

    // Look for the library file (platform-dependent extension)
    let lib_name = "libcuda_limiter.so";

    let lib_path = bin_dir.join(lib_name);
    if !lib_path.exists() {
        return Err(format!(
            "Limiter library not found at {:?}. Did you build it?",
            lib_path
        ));
    }

    Ok(lib_path)
}

/// Monitor GPU metrics (utilization and memory) for a given duration
pub fn monitor_gpu_metrics(
    gpu_index: usize,
    duration_seconds: u64,
    sample_interval_ms: u64,
) -> Result<Vec<(u32, u64)>, String> {
    let mut metrics = Vec::new();
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(duration_seconds);

    while Instant::now() < end_time {
        let utilization = get_gpu_utilization(gpu_index)?;
        let memory = get_gpu_memory_usage(gpu_index)?;

        metrics.push((utilization, memory));

        // Sleep for the sample interval
        thread::sleep(Duration::from_millis(sample_interval_ms));
    }

    Ok(metrics)
}

/// Calculate the average of a series of GPU utilization measurements
pub fn calculate_avg_utilization(metrics: &[(u32, u64)]) -> u32 {
    if metrics.is_empty() {
        return 0;
    }

    let sum: u32 = metrics.iter().map(|(util, _)| *util).sum();
    sum / metrics.len() as u32
}

/// Calculate the average of a series of GPU memory usage measurements
pub fn calculate_avg_memory(metrics: &[(u32, u64)]) -> u64 {
    if metrics.is_empty() {
        return 0;
    }

    let sum: u64 = metrics.iter().map(|(_, mem)| *mem).sum();
    sum / metrics.len() as u64
}

/// Calculate the maximum of a series of GPU utilization measurements
pub fn calculate_max_utilization(metrics: &[(u32, u64)]) -> u32 {
    metrics.iter().map(|(util, _)| *util).max().unwrap_or(0)
}

/// Calculate the maximum of a series of GPU memory usage measurements
pub fn calculate_max_memory(metrics: &[(u32, u64)]) -> u64 {
    metrics.iter().map(|(_, mem)| *mem).max().unwrap_or(0)
}

/// Check if CUDA is available on the system
pub fn is_cuda_available() -> bool {
    match Command::new("nvidia-smi").output() {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}
