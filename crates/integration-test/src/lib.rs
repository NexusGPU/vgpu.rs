use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

use error_stack::{Report, ResultExt};
use nvml_wrapper::Nvml;
use serde_json::json;

// Type aliases for complex return types
type CudaTestResult = Result<
    (
        u32,
        Box<dyn FnOnce() -> Result<Output, Report<IntegrationTestError>>>,
    ),
    Report<IntegrationTestError>,
>;

#[cfg(test)]
use serial_test::serial;

/// Errors that can occur during integration testing.
#[derive(Debug, thiserror::Error)]
pub enum IntegrationTestError {
    #[error("Failed to initialize NVML")]
    NvmlInitFailed,
    #[error("GPU device at index {index} not found")]
    GpuNotFound { index: usize },
    #[error("Failed to get GPU UUID")]
    GpuUuidFailed,
    #[error("Failed to build or find cuda-limiter library")]
    CudaLimiterBuildFailed,
    #[error("CUDA test program not found at path")]
    CudaProgramNotFound,
    #[error("Failed to spawn CUDA test process")]
    ProcessSpawnFailed,
    #[error("Failed to get output from CUDA test process")]
    ProcessOutputFailed,
}

// Test configuration constants
const DEFAULT_SHM_IDENTIFIER: &str = "tf_shm_default_ns_integration_test";
const LIMITER_MOCK_MODE: &str = "CUDA_LIMITER_MOCK_MODE";
const TF_CUDA_MEM_LIMIT: &str = "TENSOR_FUSION_CUDA_MEM_LIMIT";
const TF_CUDA_UP_LIMIT: &str = "TENSOR_FUSION_CUDA_UP_LIMIT";
const TF_SHM_IDENTIFIER: &str = "TF_SHM_IDENTIFIER";

/// Gets a global NVML instance, initializing it if necessary.
///
/// # Panics
///
/// Panics if NVML cannot be initialized either through the default method
/// or by using the fallback library path.
pub fn global_nvml() -> &'static Nvml {
    static INIT: OnceLock<Nvml> = OnceLock::new();
    INIT.get_or_init(|| {
        let nvml = match Nvml::init() {
            Ok(nvml) => nvml,
            Err(_) => Nvml::builder()
                .lib_path(OsStr::new("libnvidia-ml.so.1"))
                .init()
                .expect("should initialize NVML with fallback library path"),
        };
        nvml
    })
}

/// Runs a CUDA test program with optional GPU limiting.
///
/// # Arguments
///
/// * `memory_bytes` - Amount of GPU memory to allocate in bytes
/// * `duration_seconds` - Duration to run the test (ignored if `fixed_iterations` is provided)
/// * `gpu_index` - GPU device index to use
/// * `limit` - Optional tuple of (utilization_limit, memory_limit) percentages
/// * `fixed_iterations` - Optional fixed number of iterations for consistent timing
///
/// # Errors
///
/// Returns [`IntegrationTestError::CudaLimiterBuildFailed`] if the cuda-limiter library cannot be built or found.
/// Returns [`IntegrationTestError::ProcessSpawnFailed`] if the CUDA test program fails to start.
/// Returns [`IntegrationTestError::GpuNotFound`] if the specified GPU index is invalid.
pub fn run_cuda_test_program(
    memory_bytes: u64,
    duration_seconds: u64,
    gpu_index: usize,
    limit: Option<(u64, u64)>,
    fixed_iterations: Option<u64>,
) -> CudaTestResult {
    let mut cmd = setup_cuda_command(memory_bytes, duration_seconds, gpu_index, fixed_iterations)?;

    if let Some((utilization_limit, memory_limit)) = limit {
        apply_limiter_config(&mut cmd, gpu_index, utilization_limit, memory_limit)?;
    }

    spawn_cuda_process(cmd)
}

/// Convenience helper to run the CUDA test program and measure wall-clock time.
///
/// # Arguments
///
/// * `memory_bytes` - Amount of GPU memory to allocate in bytes
/// * `gpu_index` - GPU device index to use  
/// * `limit` - Optional tuple of (utilization_limit, memory_limit) percentages
/// * `fixed_iterations` - Number of iterations to run for consistent timing
///
/// # Errors
///
/// Returns [`IntegrationTestError`] variants if the CUDA program fails to run or complete.
pub fn run_cuda_and_measure(
    memory_bytes: u64,
    gpu_index: usize,
    limit: Option<(u64, u64)>,
    fixed_iterations: u64,
) -> Result<(u128, Output), Report<IntegrationTestError>> {
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

/// Builds and locates the cuda-limiter shared library.
///
/// First checks for an explicit path via `TF_CUDA_LIMITER_PATH` environment variable.
/// If not found, searches standard build directories and attempts to build if necessary.
///
/// # Errors
///
/// Returns [`IntegrationTestError::CudaLimiterBuildFailed`] if the library cannot be built or found.
fn build_and_find_cuda_limiter() -> Result<String, Report<IntegrationTestError>> {
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

    // First search for existing library
    if let Some(path) = find_library_in_dirs(&candidate_dirs, &candidate_names) {
        return Ok(path);
    }

    // If not found, try building the crate, then retry
    let status = Command::new("cargo")
        .args(["build", "--package", "cuda-limiter"])
        .status()
        .change_context(IntegrationTestError::CudaLimiterBuildFailed)
        .attach_printable("Failed to invoke cargo build for cuda-limiter")?;

    if !status.success() {
        return Err(Report::new(IntegrationTestError::CudaLimiterBuildFailed)
            .attach_printable("Cargo build command returned non-zero exit status"));
    }

    // Search again after building
    find_library_in_dirs(&candidate_dirs, &candidate_names).ok_or_else(|| {
        Report::new(IntegrationTestError::CudaLimiterBuildFailed)
            .attach_printable("Library not found in target directories after build")
    })
}

/// Gets the UUID for a GPU device by index.
///
/// # Arguments
///
/// * `gpu_index` - The GPU device index
///
/// # Errors
///
/// Returns [`IntegrationTestError::GpuNotFound`] if the GPU index is invalid.
/// Returns [`IntegrationTestError::GpuUuidFailed`] if the UUID cannot be retrieved.
fn get_gpu_uuid(gpu_index: usize) -> Result<String, Report<IntegrationTestError>> {
    let nvml = global_nvml();
    let device = nvml
        .device_by_index(gpu_index as u32)
        .change_context(IntegrationTestError::GpuNotFound { index: gpu_index })
        .attach_printable(format!("NVML device lookup failed for index {gpu_index}"))?;
    let uuid = device
        .uuid()
        .change_context(IntegrationTestError::GpuUuidFailed)
        .attach_printable(format!("Failed to get UUID for GPU device {gpu_index}"))?;
    Ok(uuid)
}

/// Sets up the base CUDA command with standard arguments.
fn setup_cuda_command(
    memory_bytes: u64,
    duration_seconds: u64,
    gpu_index: usize,
    fixed_iterations: Option<u64>,
) -> Result<Command, Report<IntegrationTestError>> {
    let cuda_program_path = env!("CUDA_TEST_PROGRAM_PATH");

    // Verify CUDA program exists
    if !std::path::Path::new(cuda_program_path).exists() {
        return Err(
            Report::new(IntegrationTestError::CudaProgramNotFound).attach_printable(format!(
                "CUDA test program not found at: {cuda_program_path}"
            )),
        );
    }

    let mut cmd = Command::new(cuda_program_path);
    cmd.arg("--memory-bytes")
        .arg(memory_bytes.to_string())
        .arg("--duration")
        .arg(duration_seconds.to_string())
        .arg("--gpu-index")
        .arg(gpu_index.to_string());

    if let Some(iters) = fixed_iterations {
        cmd.arg("--fixed-iterations").arg(iters.to_string());
    }

    Ok(cmd)
}

/// Applies GPU limiter configuration to the command.
fn apply_limiter_config(
    cmd: &mut Command,
    gpu_index: usize,
    utilization_limit: u64,
    memory_limit: u64,
) -> Result<(), Report<IntegrationTestError>> {
    let limiter_path = build_and_find_cuda_limiter()?;
    cmd.env("LD_PRELOAD", &limiter_path);

    // Enable mock/test mode in limiter to avoid contacting hypervisor
    cmd.env(LIMITER_MOCK_MODE, "1");

    // Convert UUID to lowercase for compatibility with limiter expectations
    let uuid_lowercase = get_gpu_uuid(gpu_index)?.to_lowercase();

    // Set memory limit as JSON
    cmd.env(
        TF_CUDA_MEM_LIMIT,
        json!({
            &uuid_lowercase: memory_limit,
        })
        .to_string(),
    );

    // Set utilization limit as JSON
    cmd.env(
        TF_CUDA_UP_LIMIT,
        json!({
            &uuid_lowercase: utilization_limit,
        })
        .to_string(),
    );

    // Set deterministic shared memory identifier for tests
    let shm_identifier =
        std::env::var(TF_SHM_IDENTIFIER).unwrap_or_else(|_| DEFAULT_SHM_IDENTIFIER.to_string());
    cmd.env(TF_SHM_IDENTIFIER, shm_identifier);

    Ok(())
}

/// Spawns the CUDA process and returns the process ID and a wait function.
fn spawn_cuda_process(mut cmd: Command) -> CudaTestResult {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let child = cmd
        .spawn()
        .change_context(IntegrationTestError::ProcessSpawnFailed)
        .attach_printable("Failed to spawn CUDA test program process")?;

    let child_id = child.id();
    let wait_fn = Box::new(move || {
        child
            .wait_with_output()
            .change_context(IntegrationTestError::ProcessOutputFailed)
            .attach_printable("Failed to get output from CUDA test program")
    });

    Ok((child_id, wait_fn))
}

/// Searches for the library file in the given directories.
fn find_library_in_dirs(dirs: &[PathBuf], names: &[&str]) -> Option<String> {
    for dir in dirs {
        for name in names {
            let path = dir.join(name);
            if path.exists() {
                return Some(path.display().to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod limiter_tests {
    use super::*;
    use error_stack::Report;

    // Test tolerance constants
    const MIN_SCALING_FACTOR_40_VS_80: f64 = 1.10;
    const MIN_SCALING_FACTOR_20_VS_40: f64 = 1.15;
    const BASELINE_TOLERANCE_FACTOR: f64 = 0.95;

    // Test configuration
    const TEST_MEMORY_MB: u64 = 512 * 1024 * 1024; // 512MB
    const TEST_ITERATIONS_SCALING: u64 = 600;
    const TEST_ITERATIONS_BASELINE: u64 = 500;

    /// Test configuration parameters for different workload types.
    const TEST_CONFIGS: &[(u64, u64, &str)] =
        &[(TEST_MEMORY_MB, TEST_ITERATIONS_SCALING, "standard_workload")];

    struct CleanupGuard;
    impl Drop for CleanupGuard {
        fn drop(&mut self) {
            std::env::remove_var(TF_SHM_IDENTIFIER);
        }
    }

    /// Checks if GPU testing prerequisites are met.
    fn check_test_prerequisites() -> Result<(), Report<IntegrationTestError>> {
        // Check if any GPU is available
        let nvml = global_nvml();
        if nvml.device_count().unwrap_or(0) == 0 {
            eprintln!("No GPU devices available, skipping GPU tests");
            return Err(Report::new(IntegrationTestError::GpuNotFound { index: 0 })
                .attach_printable("No GPU devices available for testing"));
        }

        // Check if CUDA test program exists
        let cuda_program_path = env!("CUDA_TEST_PROGRAM_PATH");
        if !std::path::Path::new(cuda_program_path).exists() {
            eprintln!("CUDA test program not found at: {cuda_program_path}");
            return Err(Report::new(IntegrationTestError::CudaProgramNotFound)
                .attach_printable(format!("CUDA test program missing: {cuda_program_path}")));
        }

        Ok(())
    }

    /// Helper function to run a test with detailed error reporting.
    fn run_test_with_detailed_output(
        description: &str,
        utilization_limit: Option<u64>,
        memory_bytes: u64,
        gpu_index: usize,
        fixed_iterations: u64,
    ) -> Result<(u128, Output), Report<IntegrationTestError>> {
        let limit = utilization_limit.map(|u| (u, memory_bytes));
        let (elapsed_ms, output) =
            run_cuda_and_measure(memory_bytes, gpu_index, limit, fixed_iterations)?;

        if !output.status.success() {
            eprintln!("Test '{description}' failed:");
            eprintln!("Exit status: {:?}", output.status);
            eprintln!("STDOUT: {}", String::from_utf8_lossy(&output.stdout));
            eprintln!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
            return Err(
                Report::new(IntegrationTestError::ProcessOutputFailed).attach_printable(format!(
                    "Test '{}' process failed with status {:?}",
                    description, output.status
                )),
            );
        }

        println!("Test '{description}' completed in {elapsed_ms}ms");
        Ok((elapsed_ms, output))
    }

    /// Tests that lower utilization limits should increase execution time proportionally.
    ///
    /// This test verifies the core functionality of the GPU limiter by running the same
    /// workload with different utilization limits and ensuring the execution times scale
    /// as expected.
    #[test]
    #[serial]
    fn lower_utilization_limits_should_increase_execution_time(
    ) -> Result<(), Report<IntegrationTestError>> {
        // Skip test if prerequisites aren't met
        if check_test_prerequisites().is_err() {
            println!("Skipping test: prerequisites not met");
            return Ok(());
        }

        let gpu_index = 0;

        // Set up test-specific shared memory identifier
        let test_shm_id = "tf_shm_test_scaling";
        std::env::set_var(TF_SHM_IDENTIFIER, test_shm_id);

        let _guard = CleanupGuard;

        // Run tests with different utilization limits
        let (t80, _) = run_test_with_detailed_output(
            "80% utilization limit",
            Some(80),
            TEST_MEMORY_MB,
            gpu_index,
            TEST_ITERATIONS_SCALING,
        )?;

        let (t40, _) = run_test_with_detailed_output(
            "40% utilization limit",
            Some(40),
            TEST_MEMORY_MB,
            gpu_index,
            TEST_ITERATIONS_SCALING,
        )?;

        let (t20, _) = run_test_with_detailed_output(
            "20% utilization limit",
            Some(20),
            TEST_MEMORY_MB,
            gpu_index,
            TEST_ITERATIONS_SCALING,
        )?;

        // Verify runtime scaling with meaningful tolerances
        let scaling_40vs80 = t40 as f64 / t80 as f64;
        let scaling_20vs40 = t20 as f64 / t40 as f64;

        assert!(
            scaling_40vs80 >= MIN_SCALING_FACTOR_40_VS_80,
            "40% utilization limit should be at least {MIN_SCALING_FACTOR_40_VS_80:.2}x slower than 80%, but got {scaling_40vs80:.2}x (t40={t40}ms, t80={t80}ms)"
        );

        assert!(
            scaling_20vs40 >= MIN_SCALING_FACTOR_20_VS_40,
            "20% utilization limit should be at least {MIN_SCALING_FACTOR_20_VS_40:.2}x slower than 40%, but got {scaling_20vs40:.2}x (t20={t20}ms, t40={t40}ms)"
        );

        println!(
            "✅ Scaling test passed: 40% is {scaling_40vs80:.2}x slower than 80%, 20% is {scaling_20vs40:.2}x slower than 40%"
        );

        Ok(())
    }

    /// Tests that unlimited baseline should be fastest execution mode.
    ///
    /// This test ensures that the GPU limiter doesn't introduce significant overhead
    /// when no limits are applied, and that limited execution is indeed slower.
    #[test]
    #[serial]
    fn unlimited_baseline_should_be_fastest_execution() -> Result<(), Report<IntegrationTestError>>
    {
        // Skip test if prerequisites aren't met
        if check_test_prerequisites().is_err() {
            println!("Skipping test: prerequisites not met");
            return Ok(());
        }

        let gpu_index = 0;

        // Set up test-specific shared memory identifier
        let test_shm_id = "tf_shm_test_baseline";
        std::env::set_var(TF_SHM_IDENTIFIER, test_shm_id);

        let _guard = CleanupGuard;

        // Test unlimited (baseline) performance
        let (t_unlimited, _) = run_test_with_detailed_output(
            "unlimited baseline",
            None,
            TEST_MEMORY_MB,
            gpu_index,
            TEST_ITERATIONS_BASELINE,
        )?;

        // Test 80% limited performance
        let (t80, _) = run_test_with_detailed_output(
            "80% utilization limit",
            Some(80),
            TEST_MEMORY_MB,
            gpu_index,
            TEST_ITERATIONS_BASELINE,
        )?;

        // Verify that 80% limited is not significantly faster than unlimited
        let performance_ratio = t80 as f64 / t_unlimited as f64;

        assert!(
            performance_ratio >= BASELINE_TOLERANCE_FACTOR,
            "80% limited execution should not be significantly faster than unlimited baseline. \
             Expected ratio >= {BASELINE_TOLERANCE_FACTOR:.2}, got {performance_ratio:.2} (t80={t80}ms, unlimited={t_unlimited}ms)"
        );

        println!(
            "✅ Baseline test passed: 80% limited is {performance_ratio:.2}x the time of unlimited ({t80}ms vs {t_unlimited}ms)"
        );

        Ok(())
    }

    /// Tests performance with different workload configurations.
    ///
    /// This parameterized test runs the same scaling verification across
    /// different memory sizes and iteration counts to ensure robustness.
    #[test]
    #[serial]
    fn performance_scaling_across_workload_configurations(
    ) -> Result<(), Report<IntegrationTestError>> {
        if check_test_prerequisites().is_err() {
            println!("Skipping test: prerequisites not met");
            return Ok(());
        }

        for &(memory_bytes, iterations, description) in TEST_CONFIGS {
            println!("Testing scaling with {description} configuration...");

            let gpu_index = 0;
            let test_shm_id = format!("tf_shm_test_config_{description}");
            std::env::set_var(TF_SHM_IDENTIFIER, &test_shm_id);

            let _guard = CleanupGuard;

            // Test with 80% and 40% limits for this configuration
            let (t80, _) = run_test_with_detailed_output(
                &format!("{description} - 80% limit"),
                Some(80),
                memory_bytes,
                gpu_index,
                iterations,
            )?;

            let (t40, _) = run_test_with_detailed_output(
                &format!("{description} - 40% limit"),
                Some(40),
                memory_bytes,
                gpu_index,
                iterations,
            )?;

            // Verify scaling for this configuration
            let scaling_ratio = t40 as f64 / t80 as f64;
            assert!(
                scaling_ratio >= MIN_SCALING_FACTOR_40_VS_80,
                "Configuration '{description}' should show {MIN_SCALING_FACTOR_40_VS_80:.2}x scaling, got {scaling_ratio:.2}x"
            );

            println!("✅ {description} scaling: {scaling_ratio:.2}x");
        }

        Ok(())
    }
}
