use tracing::info;
use tracing::warn;

use crate::integration_framework::calculate_avg_utilization;
use crate::integration_framework::is_cuda_available;
use crate::integration_framework::monitor_gpu_metrics;
use crate::integration_framework::run_cuda_test_program;
use crate::test_setup;

// Define test constants
const TEST_DURATION_SECONDS: u64 = 30;
const GPU_INDEX: usize = 0;
const MEMORY_LIMIT_BYTES: u64 = 256 * 1024 * 1024; // 256 MB
const UTILIZATION_LIMIT_PERCENT: u64 = 40;
const MONITOR_INTERVAL_MS: u64 = 500; // 500ms interval for monitoring

#[test]
fn test_memory_limit_enforcement() {
    // Skip test if CUDA is not available
    if !is_cuda_available() {
        info!("CUDA not available, skipping test");
        return;
    }

    // First run without limiter to establish baseline
    let memory_to_allocate = MEMORY_LIMIT_BYTES * 2; // 2x the limit
    info!("Running baseline test without limiter...");
    let _ = run_and_monitor(memory_to_allocate, 3, None).expect("Failed to run baseline test");

    // Now run with limiter
    info!(
        "Running test with memory limit of {} bytes...",
        MEMORY_LIMIT_BYTES
    );
    let cuda_test_result = run_and_monitor(
        memory_to_allocate,
        3,
        Some((100, MEMORY_LIMIT_BYTES)), // 100% utilization allowed
    )
    .expect("Failed to run test with limiter");

    assert!(cuda_test_result.stdout.contains("out of memory"));
}

#[test]
fn test_utilization_limit_enforcement() {
    // Skip test if CUDA is not available
    if !is_cuda_available() {
        info!("CUDA not available, skipping test");
        return;
    }

    info!("Running baseline test without limiter...");
    let baseline_result = run_and_monitor(
        10 * 1024 * 1024, // 10 MB memory
        TEST_DURATION_SECONDS,
        None, // No limiter
    )
    .expect("Failed to run baseline test");

    let baseline_metrics = baseline_result.metrics;
    let baseline_avg_utilization = calculate_avg_utilization(&baseline_metrics);
    info!(
        "Baseline average utilization: {}%",
        baseline_avg_utilization
    );

    // Now run with utilization limit
    info!(
        "Running test with utilization limit of {}%...",
        UTILIZATION_LIMIT_PERCENT
    );
    let cuda_test_result = run_and_monitor(
        10 * 1024 * 1024, // 10 MB memory
        TEST_DURATION_SECONDS,
        Some((UTILIZATION_LIMIT_PERCENT, u64::MAX)), // No memory limit, just utilization limit
    )
    .expect("Failed to run test with limiter");

    let limited_metrics = cuda_test_result.metrics;
    let limited_avg_utilization = calculate_avg_utilization(&limited_metrics);
    info!("Limited average utilization: {}%", limited_avg_utilization);

    // Make sure the baseline was actually higher to ensure test validity
    assert!(
        baseline_avg_utilization > limited_avg_utilization,
        "Baseline utilization {baseline_avg_utilization}% should be higher than limited utilization {limited_avg_utilization}%",
    );
}

struct CudaTestResult {
    metrics: Vec<u32>,
    stdout: String,
    #[allow(dead_code)]
    stderr: String,
}

// Helper function to run the test program and monitor metrics
fn run_and_monitor(
    memory_bytes: u64,
    duration_seconds: u64,
    // (utilization_limit, memory_limit)
    limit: Option<(u64, u64)>,
) -> anyhow::Result<CudaTestResult> {
    // Start the test program in the background
    let (child_id, wait_output_fn) =
        run_cuda_test_program(memory_bytes, duration_seconds, GPU_INDEX, limit)?;

    let join_handle = std::thread::spawn(move || match wait_output_fn() {
        Err(err) => {
            warn!("Error running CUDA test program: {}", err);
            Err(err)
        }
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();

            if !output.status.success() {
                warn!("CUDA test program failed:");
                warn!("STDOUT: {}", stdout);
                warn!("STDERR: {}", stderr);
            }

            Ok((stdout, stderr))
        }
    });

    let nvml = test_setup::global_nvml();
    // Monitor metrics while the program runs
    let metrics = monitor_gpu_metrics(
        nvml,
        child_id,
        GPU_INDEX,
        duration_seconds - 2,
        MONITOR_INTERVAL_MS,
    )?;

    let thread_result = join_handle.join().unwrap().unwrap();

    Ok(CudaTestResult {
        metrics,
        stdout: thread_result.0,
        stderr: thread_result.1,
    })
}
