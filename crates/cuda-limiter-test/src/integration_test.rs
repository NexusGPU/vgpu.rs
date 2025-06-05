use std::time::Duration;

use tracing::info;
use tracing::warn;

use crate::calculate_avg_utilization;
use crate::calculate_max_utilization;
use crate::clear_limiter_env_vars;
use crate::get_gpu_uuid;
use crate::is_cuda_available;
use crate::monitor_gpu_metrics;
use crate::run_cuda_test_program;
use crate::set_limiter_env_vars;
use crate::test_setup;

// Define test constants
const TEST_DURATION_SECONDS: u64 = 10;
const GPU_INDEX: usize = 0;
const MEMORY_LIMIT_BYTES: u64 = 256 * 1024 * 1024; // 256 MB
const UTILIZATION_LIMIT_PERCENT: u32 = 30;
const MONITOR_INTERVAL_MS: u64 = 500; // 500ms interval for monitoring

#[test]
fn test_memory_limit_enforcement() {
    // Skip test if CUDA is not available
    if !is_cuda_available() {
        info!("CUDA not available, skipping test");
        return;
    }

    // Get GPU UUID
    let gpu_uuid = get_gpu_uuid(GPU_INDEX).expect("Failed to get GPU UUID");
    info!("Testing with GPU UUID: {}", gpu_uuid);

    // First run without limiter to establish baseline
    clear_limiter_env_vars();
    let memory_to_allocate = MEMORY_LIMIT_BYTES * 2; // 2x the limit
    info!("Running baseline test without limiter...");
    let _ = run_and_monitor(
        memory_to_allocate,
        false, // Don't use limiter,
        3,
    )
    .expect("Failed to run baseline test");

    // Now run with limiter
    set_limiter_env_vars(&gpu_uuid, MEMORY_LIMIT_BYTES, 100); // 100% utilization allowed
    info!(
        "Running test with memory limit of {} bytes...",
        MEMORY_LIMIT_BYTES
    );
    let cuda_test_result = run_and_monitor(
        memory_to_allocate,
        true, // Use limiter,
        3,
    )
    .expect("Failed to run test with limiter");

    assert!(cuda_test_result.stdout.contains("out of memory"));

    // Cleanup
    clear_limiter_env_vars();
}

#[test]
// TODO: ignore
#[ignore]
fn test_utilization_limit_enforcement() {
    // Skip test if CUDA is not available
    if !is_cuda_available() {
        info!("CUDA not available, skipping test");
        return;
    }

    // Get GPU UUID
    let gpu_uuid = get_gpu_uuid(GPU_INDEX).expect("Failed to get GPU UUID");
    info!("Testing with GPU UUID: {}", gpu_uuid);

    // First run without limiter to establish baseline
    clear_limiter_env_vars();
    info!("Running baseline test without limiter...");
    let baseline_result = run_and_monitor(
        100 * 1024 * 1024, // 100 MB memory
        false,             // Don't use limiter
        TEST_DURATION_SECONDS,
    )
    .expect("Failed to run baseline test");

    let baseline_metrics = baseline_result.metrics;
    let baseline_avg_utilization = calculate_avg_utilization(&baseline_metrics);
    info!(
        "Baseline average utilization: {}%",
        baseline_avg_utilization
    );

    // Now run with utilization limit
    set_limiter_env_vars(&gpu_uuid, u64::MAX, UTILIZATION_LIMIT_PERCENT); // No memory limit
    info!(
        "Running test with utilization limit of {}%...",
        UTILIZATION_LIMIT_PERCENT
    );
    let cuda_test_result = run_and_monitor(
        100 * 1024 * 1024, // 100 MB memory
        true,              // Use limiter,
        TEST_DURATION_SECONDS,
    )
    .expect("Failed to run test with limiter");

    let limited_metrics = cuda_test_result.metrics;
    let limited_avg_utilization = calculate_avg_utilization(&limited_metrics);
    let limited_max_utilization = calculate_max_utilization(&limited_metrics);
    info!("Limited average utilization: {}%", limited_avg_utilization);
    info!("Limited maximum utilization: {}%", limited_max_utilization);

    // Allow some overhead beyond the exact limit
    let allowable_overhead = 10; // 10% overhead allowed

    // Verification
    assert!(
        limited_max_utilization <= UTILIZATION_LIMIT_PERCENT + allowable_overhead,
        "Utilization limit not enforced: max utilization {}% exceeds limit {}% (with overhead {}%)",
        limited_max_utilization,
        UTILIZATION_LIMIT_PERCENT,
        allowable_overhead
    );

    // Make sure the baseline was actually higher to ensure test validity
    assert!(
        baseline_avg_utilization > limited_avg_utilization,
        "Baseline utilization {}% should be higher than limited utilization {}%",
        baseline_avg_utilization,
        limited_avg_utilization
    );

    // Cleanup
    clear_limiter_env_vars();
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
    use_limiter: bool,
    duration_seconds: u64,
) -> anyhow::Result<CudaTestResult> {
    // Start the test program in the background
    let (child_id, wait_output_fn) =
        run_cuda_test_program(memory_bytes, duration_seconds, GPU_INDEX, use_limiter)?;
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

    // Give the program a moment to start
    std::thread::sleep(Duration::from_secs(1));
    let nvml = test_setup::global_nvml();
    // Monitor metrics while the program runs
    let metrics = monitor_gpu_metrics(
        &nvml,
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
