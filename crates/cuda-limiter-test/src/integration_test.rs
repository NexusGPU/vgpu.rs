use std::time::Duration;

use tracing::info;
use tracing::warn;

use crate::calculate_avg_memory;
use crate::calculate_avg_utilization;
use crate::calculate_max_memory;
use crate::calculate_max_utilization;
use crate::clear_limiter_env_vars;
use crate::get_gpu_uuid;
use crate::is_cuda_available;
use crate::monitor_gpu_metrics;
use crate::run_cuda_test_program;
use crate::set_limiter_env_vars;
use crate::set_limiter_env_vars_uppercase;

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
    let baseline_metrics = run_and_monitor(
        memory_to_allocate,
        50,    // 50% utilization
        false, // Don't use limiter
    )
    .expect("Failed to run baseline test");

    let baseline_avg_memory = calculate_avg_memory(&baseline_metrics);
    info!(
        "Baseline average memory usage: {} bytes",
        baseline_avg_memory
    );

    // Now run with limiter
    set_limiter_env_vars(&gpu_uuid, MEMORY_LIMIT_BYTES, 100); // 100% utilization allowed
    info!(
        "Running test with memory limit of {} bytes...",
        MEMORY_LIMIT_BYTES
    );
    let limited_metrics = run_and_monitor(
        memory_to_allocate,
        50,   // 50% utilization
        true, // Use limiter
    )
    .expect("Failed to run test with limiter");

    let limited_avg_memory = calculate_avg_memory(&limited_metrics);
    let limited_max_memory = calculate_max_memory(&limited_metrics);
    info!("Limited average memory usage: {} bytes", limited_avg_memory);
    info!("Limited maximum memory usage: {} bytes", limited_max_memory);

    // Allow some overhead beyond the exact limit
    let allowable_overhead = MEMORY_LIMIT_BYTES / 10; // 10% overhead allowed

    // Verification
    assert!(
        limited_max_memory <= MEMORY_LIMIT_BYTES + allowable_overhead,
        "Memory limit not enforced: max memory {} exceeds limit {} (with overhead {})",
        limited_max_memory,
        MEMORY_LIMIT_BYTES,
        allowable_overhead
    );

    // Make sure the baseline was actually higher to ensure test validity
    assert!(
        baseline_avg_memory > limited_avg_memory,
        "Baseline memory usage {} should be higher than limited usage {}",
        baseline_avg_memory,
        limited_avg_memory
    );

    // Cleanup
    clear_limiter_env_vars();
}

#[test]
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
    let baseline_metrics = run_and_monitor(
        100 * 1024 * 1024, // 100 MB memory
        80,                // 80% utilization (higher than our limit)
        false,             // Don't use limiter
    )
    .expect("Failed to run baseline test");

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
    let limited_metrics = run_and_monitor(
        100 * 1024 * 1024, // 100 MB memory
        80,                // 80% utilization (higher than our limit)
        true,              // Use limiter
    )
    .expect("Failed to run test with limiter");

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

#[test]
fn test_gpu_uuid_case_insensitivity() {
    // Skip test if CUDA is not available
    if !is_cuda_available() {
        info!("CUDA not available, skipping test");
        return;
    }

    // Get GPU UUID
    let gpu_uuid = get_gpu_uuid(GPU_INDEX).expect("Failed to get GPU UUID");
    info!("Testing with GPU UUID: {}", gpu_uuid);

    // Test with uppercase UUID
    set_limiter_env_vars_uppercase(&gpu_uuid, MEMORY_LIMIT_BYTES, UTILIZATION_LIMIT_PERCENT);
    info!("Running test with uppercase UUID format...");
    let uppercase_metrics = run_and_monitor(
        MEMORY_LIMIT_BYTES * 2, // 2x the memory limit
        80,                     // 80% utilization (higher than our limit)
        true,                   // Use limiter
    )
    .expect("Failed to run test with uppercase UUID");

    let uppercase_avg_memory = calculate_avg_memory(&uppercase_metrics);
    let uppercase_max_memory = calculate_max_memory(&uppercase_metrics);
    let uppercase_avg_utilization = calculate_avg_utilization(&uppercase_metrics);
    let uppercase_max_utilization = calculate_max_utilization(&uppercase_metrics);

    info!("Uppercase UUID test results:");
    info!(
        "  Memory: Avg={} bytes, Max={} bytes",
        uppercase_avg_memory, uppercase_max_memory
    );
    info!(
        "  Utilization: Avg={}%, Max={}%",
        uppercase_avg_utilization, uppercase_max_utilization
    );

    // Clear and set with lowercase UUID
    clear_limiter_env_vars();
    set_limiter_env_vars(
        &gpu_uuid.to_lowercase(),
        MEMORY_LIMIT_BYTES,
        UTILIZATION_LIMIT_PERCENT,
    );
    info!("Running test with lowercase UUID format...");
    let lowercase_metrics = run_and_monitor(
        MEMORY_LIMIT_BYTES * 2, // 2x the memory limit
        80,                     // 80% utilization (higher than our limit)
        true,                   // Use limiter
    )
    .expect("Failed to run test with lowercase UUID");

    let lowercase_avg_memory = calculate_avg_memory(&lowercase_metrics);
    let lowercase_max_memory = calculate_max_memory(&lowercase_metrics);
    let lowercase_avg_utilization = calculate_avg_utilization(&lowercase_metrics);
    let lowercase_max_utilization = calculate_max_utilization(&lowercase_metrics);

    info!("Lowercase UUID test results:");
    info!(
        "  Memory: Avg={} bytes, Max={} bytes",
        lowercase_avg_memory, lowercase_max_memory
    );
    info!(
        "  Utilization: Avg={}%, Max={}%",
        lowercase_avg_utilization, lowercase_max_utilization
    );

    // Allow some overhead beyond the exact limits
    let memory_overhead = MEMORY_LIMIT_BYTES / 10; // 10% overhead allowed
    let utilization_overhead = 10; // 10% overhead allowed

    // Verification for uppercase
    assert!(
        uppercase_max_memory <= MEMORY_LIMIT_BYTES + memory_overhead,
        "Memory limit not enforced with uppercase UUID: max memory {} exceeds limit {}",
        uppercase_max_memory,
        MEMORY_LIMIT_BYTES
    );

    assert!(
        uppercase_max_utilization <= UTILIZATION_LIMIT_PERCENT + utilization_overhead,
        "Utilization limit not enforced with uppercase UUID: max utilization {}% exceeds limit {}%",
        uppercase_max_utilization,
        UTILIZATION_LIMIT_PERCENT
    );

    // Verification for lowercase
    assert!(
        lowercase_max_memory <= MEMORY_LIMIT_BYTES + memory_overhead,
        "Memory limit not enforced with lowercase UUID: max memory {} exceeds limit {}",
        lowercase_max_memory,
        MEMORY_LIMIT_BYTES
    );

    assert!(
        lowercase_max_utilization <= UTILIZATION_LIMIT_PERCENT + utilization_overhead,
        "Utilization limit not enforced with lowercase UUID: max utilization {}% exceeds limit {}%",
        lowercase_max_utilization,
        UTILIZATION_LIMIT_PERCENT
    );

    // Cleanup
    clear_limiter_env_vars();
}

// Helper function to run the test program and monitor metrics
fn run_and_monitor(
    memory_bytes: u64,
    utilization_percent: u32,
    use_limiter: bool,
) -> Result<Vec<(u32, u64)>, String> {
    // Start the test program in the background
    std::thread::spawn(move || {
        let output = run_cuda_test_program(
            memory_bytes,
            utilization_percent,
            TEST_DURATION_SECONDS,
            GPU_INDEX,
            use_limiter,
        );

        if let Err(err) = &output {
            warn!("Error running CUDA test program: {}", err);
        } else if let Ok(output) = output {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            if !output.status.success() {
                warn!("CUDA test program failed:");
                warn!("STDOUT: {}", stdout);
                warn!("STDERR: {}", stderr);
            }
        }
    });

    // Give the program a moment to start
    std::thread::sleep(Duration::from_secs(1));

    // Monitor metrics while the program runs
    monitor_gpu_metrics(GPU_INDEX, TEST_DURATION_SECONDS - 2, MONITOR_INTERVAL_MS)
}
