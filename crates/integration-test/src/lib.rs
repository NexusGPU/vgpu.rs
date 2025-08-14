use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

use nvml_wrapper::Nvml;
use serde_json::json;
use std::ffi;

pub fn global_nvml() -> &'static Nvml {
    static INIT: OnceLock<Nvml> = OnceLock::new();
    INIT.get_or_init(|| {
        let nvml = match Nvml::init() {
            Ok(nvml) => nvml,
            Err(_) => Nvml::builder()
                .lib_path(ffi::OsStr::new("libnvidia-ml.so.1"))
                .init()
                .expect("Failed to initialize NVML"),
        };
        nvml
    })
}

pub fn run_cuda_test_program(
    memory_bytes: u64,
    duration_seconds: u64,
    gpu_index: usize,
    // (utilization_limit, memory_limit)
    limit: Option<(u64, u64)>,
    // If provided, run a fixed number of outer iterations to allow runtime comparisons
    fixed_iterations: Option<u64>,
) -> anyhow::Result<(u32, impl FnOnce() -> anyhow::Result<Output>)> {
    let cuda_program_path = env!("CUDA_TEST_PROGRAM_PATH");
    let mut cmd = Command::new(cuda_program_path);
    // Use the flag-based CLI supported by the CUDA test binary
    cmd.arg("--memory-bytes")
        .arg(memory_bytes.to_string())
        .arg("--duration")
        .arg(duration_seconds.to_string())
        .arg("--gpu-index")
        .arg(gpu_index.to_string());

    if let Some(iters) = fixed_iterations {
        cmd.arg("--fixed-iterations").arg(iters.to_string());
    }

    // If using limiter, set up LD_PRELOAD
    if let Some((utilization_limit, memory_limit)) = limit {
        let limiter_path = build_and_find_cuda_limiter()?;
        cmd.env("LD_PRELOAD", &limiter_path);
        // Enable mock/test mode in limiter to avoid contacting hypervisor
        cmd.env("CUDA_LIMITER_MOCK_MODE", "1");

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

        // Also wire a deterministic shared memory identifier for tests, so limiter can find it
        // Format: tf_shm_<namespace>_<podname>
        let shm_identifier = std::env::var("TF_SHM_IDENTIFIER").unwrap_or_else(|_| {
            // default to a single-test identifier
            "tf_shm_default_ns_integration_test".to_string()
        });
        cmd.env("TF_SHM_IDENTIFIER", shm_identifier);
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

/// Convenience helper to run the CUDA test program and measure wall-clock time.
pub fn run_cuda_and_measure(
    memory_bytes: u64,
    gpu_index: usize,
    limit: Option<(u64, u64)>,
    fixed_iterations: u64,
) -> anyhow::Result<(u128, Output)> {
    let start = std::time::Instant::now();
    let (_pid, wait) = run_cuda_test_program(
        memory_bytes,
        1, // duration is ignored when fixed_iterations is used
        gpu_index,
        limit,
        Some(fixed_iterations),
    )?;
    let output = wait()?;
    let elapsed_ms = start.elapsed().as_millis();
    Ok((elapsed_ms, output))
}

fn build_and_find_cuda_limiter() -> anyhow::Result<String> {
    // Allow override via env for flexibility
    if let Ok(explicit) = std::env::var("TF_CUDA_LIMITER_PATH") {
        if std::path::Path::new(&explicit).exists() {
            return Ok(explicit);
        }
    }

    // Compute workspace root: integration-test crate lives at <root>/crates/integration-test
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(manifest_dir.as_path())
        .to_path_buf();

    // Candidate file names and directories (Linux)
    let candidate_names = ["libcuda_limiter.so", "libcuda-limiter.so"]; // hyphen vs underscore safety
    let candidate_dirs = [
        workspace_root.join("target/debug"),
        workspace_root.join("target/release"),
    ];

    for dir in &candidate_dirs {
        for name in &candidate_names {
            let path = dir.join(name);
            if path.exists() {
                return Ok(path.display().to_string());
            }
        }
    }

    // If not found, try building the crate, then retry
    let status = Command::new("cargo")
        .args(["build", "--package", "cuda-limiter"])
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to invoke cargo build for cuda-limiter: {e}"))?;
    if !status.success() {
        return Err(anyhow::anyhow!("Building cuda-limiter failed"));
    }

    for dir in &candidate_dirs {
        for name in &candidate_names {
            let path = dir.join(name);
            if path.exists() {
                return Ok(path.display().to_string());
            }
        }
    }

    Err(anyhow::anyhow!(
        "Unable to locate cuda-limiter shared library in target directories"
    ))
}

fn get_gpu_uuid(gpu_index: usize) -> anyhow::Result<String> {
    let nvml = global_nvml();
    let device = nvml
        .device_by_index(gpu_index as u32)
        .map_err(|e| anyhow::anyhow!("Failed to get NVML device by index {gpu_index}: {e}"))?;
    let uuid = device
        .uuid()
        .map_err(|e| anyhow::anyhow!("Failed to get NVML UUID for device {gpu_index}: {e}"))?;
    Ok(uuid)
}

#[cfg(test)]
mod limiter_tests {
    use super::*;

    // in the same fixed workload, different up_limit should significantly affect runtime: the smaller the limit, the longer the runtime
    #[test]
    fn test_cuda_limiter_runtime_scaling() -> anyhow::Result<()> {
        let memory_bytes = 512 * 1024 * 1024; // 512MB
        let gpu_index = 0usize;
        let fixed_iters = 600u64; // large enough to stabilize the difference

        // 80%
        let (t80, o80) = run_cuda_and_measure(
            memory_bytes,
            gpu_index,
            Some((80, memory_bytes)),
            fixed_iters,
        )?;
        assert!(
            o80.status.success(),
            "80% run failed: status={:?}",
            o80.status
        );

        // 40%
        let (t40, o40) = run_cuda_and_measure(
            memory_bytes,
            gpu_index,
            Some((40, memory_bytes)),
            fixed_iters,
        )?;
        assert!(
            o40.status.success(),
            "40% run failed: status={:?}",
            o40.status
        );

        // 20%
        let (t20, o20) = run_cuda_and_measure(
            memory_bytes,
            gpu_index,
            Some((20, memory_bytes)),
            fixed_iters,
        )?;
        assert!(
            o20.status.success(),
            "20% run failed: status={:?}",
            o20.status
        );

        // assert that the runtime increases as the limit decreases (with some tolerance to avoid jitter)
        assert!(
            t40 >= t80 * 110 / 100,
            "t40({t40}ms) should be >= 1.10 * t80({t80}ms)"
        );
        assert!(
            t20 >= t40 * 115 / 100,
            "t20({t20}ms) should be >= 1.15 * t40({t40}ms)"
        );

        Ok(())
    }

    // baseline (unlimited) should not be slower than 80% limit
    #[test]
    fn test_cuda_limiter_vs_unlimited_baseline() -> anyhow::Result<()> {
        let memory_bytes = 512 * 1024 * 1024; // 512MB
        let gpu_index = 0usize;
        let fixed_iters = 500u64;

        let (t_unlimited, o0) = run_cuda_and_measure(memory_bytes, gpu_index, None, fixed_iters)?;
        assert!(
            o0.status.success(),
            "unlimited run failed: status={:?}",
            o0.status
        );

        let (t80, o80) = run_cuda_and_measure(
            memory_bytes,
            gpu_index,
            Some((80, memory_bytes)),
            fixed_iters,
        )?;
        assert!(
            o80.status.success(),
            "80% run failed: status={:?}",
            o80.status
        );

        // allow some noise, require 80% not faster than baseline's 95%
        assert!(
            t80 >= t_unlimited * 95 / 100,
            "t80({t80}ms) should be >= 0.95 * t_unlimited({t_unlimited}ms)"
        );
        Ok(())
    }
}
