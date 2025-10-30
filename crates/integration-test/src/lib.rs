use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

use api_types::WorkerInfo;
use error_stack::{Report, ResultExt};
use hypervisor::pod_management::coordinator::{CoordinatorConfig, LimiterCoordinator};
use hypervisor::pod_management::device_info::calculate_device_limits_from_gpu_info;
use hypervisor::pod_management::pod_state_store::PodStateStore;
use hypervisor::pod_management::sampler::{NvmlDeviceSampler, SystemClock};
use nvml_wrapper::Nvml;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;
use utils::shared_memory::handle::SHM_PATH_SUFFIX;
use utils::shared_memory::manager::MemoryManager;
use utils::shared_memory::{DeviceConfig, PodIdentifier};

// Type aliases for complex return types
type CudaTestResult = Result<
    (
        u32,
        Box<dyn FnOnce() -> Result<Output, Report<IntegrationTestError>> + Send>,
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
const LIMITER_MOCK_MODE: &str = "CUDA_LIMITER_MOCK_MODE";
const TF_SHM_FILE: &str = "TF_SHM_FILE";

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

pub fn run_cuda_test_program(
    memory_bytes: u64,
    gpu_index: usize,
    gpu_uuid: &str,
    is_limiter_enabled: bool,
    iterations: Option<u64>,
) -> CudaTestResult {
    let mut cmd = setup_cuda_command(memory_bytes, gpu_index, iterations)?;
    if is_limiter_enabled {
        apply_limiter_config(&mut cmd, gpu_uuid)?;
    }
    spawn_cuda_process(cmd)
}

pub async fn run_cuda_and_measure<F, Fut>(
    memory_bytes: u64,
    gpu_index: usize,
    gpu_uuid: &str,
    iterations: u64,
    is_limiter_enabled: bool,
    post_start: F,
) -> Result<(u128, Output), Report<IntegrationTestError>>
where
    F: Fn(u32) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let start = std::time::Instant::now();
    let (pid, wait) = run_cuda_test_program(
        memory_bytes,
        gpu_index,
        gpu_uuid,
        is_limiter_enabled,
        Some(iterations),
    )?;
    post_start(pid).await;
    let output = tokio::task::spawn_blocking(wait).await.unwrap()?;
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
        if Path::new(&explicit).exists() {
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
        .attach("Failed to invoke cargo build for cuda-limiter")?;

    if !status.success() {
        return Err(Report::new(IntegrationTestError::CudaLimiterBuildFailed)
            .attach("Cargo build command returned non-zero exit status"));
    }

    // Search again after building
    find_library_in_dirs(&candidate_dirs, &candidate_names).ok_or_else(|| {
        Report::new(IntegrationTestError::CudaLimiterBuildFailed)
            .attach("Library not found in target directories after build")
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
        .attach(format!("NVML device lookup failed for index {gpu_index}"))?;
    let uuid = device
        .uuid()
        .change_context(IntegrationTestError::GpuUuidFailed)
        .attach(format!("Failed to get UUID for GPU device {gpu_index}"))?;
    Ok(uuid)
}

/// Sets up the base CUDA command with standard arguments.
fn setup_cuda_command(
    memory_bytes: u64,
    gpu_index: usize,
    iterations: Option<u64>,
) -> Result<Command, Report<IntegrationTestError>> {
    let cuda_program_path = env!("CUDA_TEST_PROGRAM_PATH");

    // Verify CUDA program exists
    if !Path::new(cuda_program_path).exists() {
        return Err(
            Report::new(IntegrationTestError::CudaProgramNotFound).attach(format!(
                "CUDA test program not found at: {cuda_program_path}"
            )),
        );
    }

    let mut cmd = Command::new(cuda_program_path);
    cmd.arg("--memory-bytes")
        .arg(memory_bytes.to_string())
        .arg("--gpu-index")
        .arg(gpu_index.to_string());

    if let Some(iters) = iterations {
        cmd.arg("--iterations").arg(iters.to_string());
    }

    Ok(cmd)
}

/// Applies GPU limiter configuration to the command.
fn apply_limiter_config(
    cmd: &mut Command,
    gpu_uuid: &str,
) -> Result<(), Report<IntegrationTestError>> {
    let limiter_path = build_and_find_cuda_limiter()?;
    cmd.env("LD_PRELOAD", &limiter_path);

    // Enable mock/test mode in limiter to avoid contacting hypervisor
    cmd.env(LIMITER_MOCK_MODE, "1");

    // Set shared memory file environment variable
    // In mock mode, the limiter expects TF_SHM_FILE to contain the shared memory path
    if let Ok(shm_file) = std::env::var(TF_SHM_FILE) {
        cmd.env(TF_SHM_FILE, shm_file);
    }

    cmd.env("TF_VISIBLE_DEVICES", gpu_uuid);
    Ok(())
}

/// Spawns the CUDA process and returns the process ID and a wait function.
fn spawn_cuda_process(mut cmd: Command) -> CudaTestResult {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let child = cmd
        .spawn()
        .change_context(IntegrationTestError::ProcessSpawnFailed)
        .attach("Failed to spawn CUDA test program process")?;

    let child_id = child.id();
    let wait_fn = Box::new(move || {
        child
            .wait_with_output()
            .change_context(IntegrationTestError::ProcessOutputFailed)
            .attach("Failed to get output from CUDA test program")
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

/// Helper to start a mock coordinator for resource management during tests
pub async fn mock_coordinator(
    test_shm_id: &str,
    _gpu_index: usize,
) -> (
    Arc<LimiterCoordinator<MemoryManager, PodStateStore, NvmlDeviceSampler, SystemClock>>,
    Arc<PodStateStore>,
) {
    // Create mock dependencies
    let tmpdir = tempdir().expect("Failed to create temporary directory");
    let shared_memory = Arc::new(MemoryManager::new());
    let base_path = tmpdir.path().to_path_buf();
    let pod_state = Arc::new(PodStateStore::new(base_path.clone()));
    let snapshot = Arc::new(NvmlDeviceSampler::init().unwrap());
    let time = Arc::new(SystemClock::new());

    // Create coordinator with mock dependencies
    // Create default ERL config for testing
    let erl_config = hypervisor::config::ErlConfig {
        update_interval_ms: 100,
        rate_min: 10.0,
        rate_max: 5_000.0,
        kp: 0.5,
        ki: 0.1,
        kd: 0.05,
        filter_alpha: 0.3,
        burst_window: 2.0,
        capacity_min: 100.0,
        capacity_max: 10_000.0,
    };

    let config = CoordinatorConfig {
        watch_interval: Duration::from_millis(50), // Fast monitoring for tests
        device_count: 1,                           // Single GPU
        shared_memory_glob_pattern: format!("{test_shm_id}*"),
        base_path,
        erl_config,
    };

    let coordinator = Arc::new(LimiterCoordinator::new(
        config,
        shared_memory,
        pod_state.clone(),
        snapshot,
        time,
    ));

    (coordinator, pod_state)
}

// Test constants
const TEST_MEMORY: u64 = 256 * 1024 * 1024; // 256MB

/// Test coordinator manager that handles a single coordinator with multiple pods
pub struct TestCoordinatorManager {
    coordinator:
        Arc<LimiterCoordinator<MemoryManager, PodStateStore, NvmlDeviceSampler, SystemClock>>,
    pod_state: Arc<PodStateStore>,
    cancellation_token: CancellationToken,
    gpu_index: usize,
    registered_pods: HashSet<PodIdentifier>,
}

/// A registered pod within the test coordinator
pub struct TestPod {
    pub pod_id: PodIdentifier,
    device_config: DeviceConfig,
    _cleanup_guard: PodCleanupGuard,
}

struct PodCleanupGuard {
    base_path: PathBuf,
    pod_id: PodIdentifier,
}

impl Drop for PodCleanupGuard {
    fn drop(&mut self) {
        let pod_path = self.pod_id.to_path(&self.base_path);
        // Clean up shared memory file for this specific pod
        let _ = std::fs::remove_file(pod_path.join(SHM_PATH_SUFFIX));
        // Remove environment variable if it matches this pod
        if std::env::var(TF_SHM_FILE).unwrap_or_default() == pod_path.to_string_lossy() {
            std::env::remove_var(TF_SHM_FILE);
        }
    }
}

impl TestCoordinatorManager {
    /// Creates a new test coordinator manager
    pub async fn new(gpu_index: usize) -> Self {
        let base_shm_id = format!("tf_shm_coordinator_{gpu_index}");

        // Create coordinator and pod state
        let (coordinator, pod_state) = mock_coordinator(&base_shm_id, gpu_index).await;
        let cancellation_token = CancellationToken::new();

        // Start coordinator in background
        tokio::spawn({
            let coordinator = coordinator.clone();
            let cancellation_token = cancellation_token.clone();
            async move {
                coordinator.run(cancellation_token).await;
            }
        });

        // Wait for coordinator to initialize
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        Self {
            coordinator,
            pod_state,
            cancellation_token,
            gpu_index,
            registered_pods: HashSet::new(),
        }
    }

    /// Register a new pod with custom up_limit
    pub async fn register_pod_with_limit(
        &mut self,
        pod_name: &str,
        up_limit: u32,
    ) -> Result<TestPod, Report<IntegrationTestError>> {
        let nvml = global_nvml();
        let device_uuid = get_gpu_uuid(self.gpu_index)?;

        let (
            total_cuda_cores,
            sm_count,
            max_thread_per_sm,
            _calculated_up_limit,
            _calculated_mem_limit,
        ) = calculate_device_limits_from_gpu_info(
            nvml,
            self.gpu_index as u32,
            None,                   // tflops_limit - custom up_limit
            Some(TEST_MEMORY * 10), // vram_limit
            None,                   // tflops_capacity - not needed for tests
        )
        .map_err(|e| {
            Report::new(IntegrationTestError::GpuUuidFailed)
                .attach(format!("Failed to calculate device limits: {e}"))
        })?;

        let device_config = DeviceConfig {
            device_idx: self.gpu_index as u32,
            device_uuid,
            up_limit, // custom up_limit
            mem_limit: TEST_MEMORY * 10,
            total_cuda_cores: (total_cuda_cores as f64 * (up_limit as f64 / 100.0)).round() as u32,
            sm_count,
            max_thread_per_sm,
        };

        self.register_pod_with_config(pod_name, device_config).await
    }

    /// Register a new pod with custom device configuration
    pub async fn register_pod_with_config(
        &mut self,
        pod_name: &str,
        device_config: DeviceConfig,
    ) -> Result<TestPod, Report<IntegrationTestError>> {
        let pod_identifier = PodIdentifier::new("test", pod_name);

        // Register the pod
        self.pod_state
            .register_pod(
                &pod_identifier,
                WorkerInfo::default(),
                vec![device_config.clone()],
            )
            .map_err(|e| {
                Report::new(IntegrationTestError::ProcessSpawnFailed)
                    .attach(format!("Failed to register pod '{pod_identifier}': {e}"))
            })?;

        self.coordinator
            .ensure_pod_registered(&pod_identifier, &[device_config.clone()])
            .await
            .map_err(|e| {
                Report::new(IntegrationTestError::ProcessSpawnFailed)
                    .attach(format!("Failed to register pod '{pod_identifier}': {e}"))
            })?;

        self.registered_pods.insert(pod_identifier.clone());
        let cleanup_guard = PodCleanupGuard {
            base_path: self.pod_state.base_path().clone(),
            pod_id: pod_identifier.clone(),
        };

        Ok(TestPod {
            pod_id: pod_identifier,
            device_config,
            _cleanup_guard: cleanup_guard,
        })
    }

    /// Register a process with a specific pod
    pub async fn register_process(
        &self,
        pod: &TestPod,
        pid: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.pod_state.register_process(&pod.pod_id, pid)?;
        self.coordinator.register_process(&pod.pod_id, pid).await?;
        Ok(())
    }

    /// Run a test with detailed output and automatic process registration
    pub async fn run_test_with_pod(
        &self,
        pod: &TestPod,
        description: &str,
        iterations: u64,
        memory_bytes: u64,
        is_limiter_enabled: bool,
    ) -> Result<(u128, Output), Report<IntegrationTestError>> {
        // Set environment variable for this pod - use the full path
        let pod_path = self.pod_state.pod_path(&pod.pod_id).join(SHM_PATH_SUFFIX);
        std::env::set_var(TF_SHM_FILE, pod_path.to_string_lossy().as_ref());

        self.run_test_with_detailed_output(
            description,
            iterations,
            memory_bytes,
            is_limiter_enabled,
            |pid| async move {
                if let Err(e) = self.register_process(pod, pid).await {
                    eprintln!(
                        "Failed to register process {} for pod {}: {}",
                        pid, pod.pod_id, e
                    );
                }
            },
        )
        .await
    }

    /// Internal method that integrates run_test_with_detailed_output functionality
    async fn run_test_with_detailed_output<F, Fut>(
        &self,
        description: &str,
        iterations: u64,
        memory_bytes: u64,
        is_limiter_enabled: bool,
        post_start: F,
    ) -> Result<(u128, Output), Report<IntegrationTestError>>
    where
        F: Fn(u32) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let device_uuid = get_gpu_uuid(self.gpu_index)?;
        let (elapsed_ms, output) = run_cuda_and_measure(
            memory_bytes,
            self.gpu_index,
            &device_uuid,
            iterations,
            is_limiter_enabled,
            post_start,
        )
        .await?;

        if !output.status.success() {
            eprintln!("Test '{description}' failed:");
            eprintln!("Exit status: {:?}", output.status);
            eprintln!("STDOUT: {}", String::from_utf8_lossy(&output.stdout));
            eprintln!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
            return Err(
                Report::new(IntegrationTestError::ProcessOutputFailed).attach(format!(
                    "Test '{}' process failed with status {:?}",
                    description, output.status
                )),
            );
        }

        println!("Test '{description}' completed in {elapsed_ms}ms");
        Ok((elapsed_ms, output))
    }

    /// Get the GPU index
    pub fn gpu_index(&self) -> usize {
        self.gpu_index
    }
}

impl Drop for TestCoordinatorManager {
    fn drop(&mut self) {
        // Cancel the coordinator
        self.cancellation_token.cancel();

        // Clean up any remaining shared memory files using proper paths
        for pod_id in &self.registered_pods {
            let pod_path = self.pod_state.pod_path(pod_id).join(SHM_PATH_SUFFIX);
            let _ = std::fs::remove_file(&pod_path);
        }
    }
}

impl TestPod {
    /// Get the pod ID as a display string
    pub fn pod_id(&self) -> String {
        format!("{}", self.pod_id)
    }

    /// Get the device configuration
    pub fn device_config(&self) -> &DeviceConfig {
        &self.device_config
    }
}

#[cfg(test)]
mod limiter_tests {
    use std::path::Path;

    use super::*;
    use error_stack::Report;

    // Test tolerance constants
    const MIN_SCALING_FACTOR_40_VS_80: f64 = 1.10;
    const MIN_SCALING_FACTOR_20_VS_40: f64 = 1.15;

    /// Checks if GPU testing prerequisites are met.
    fn check_test_prerequisites() -> Result<(), Report<IntegrationTestError>> {
        // Check if CUDA is available on the system
        if Command::new("nvidia-smi").output().is_err() {
            return Err(Report::new(IntegrationTestError::CudaProgramNotFound)
                .attach("nvidia-smi not found"));
        }
        // Check if CUDA test program exists
        let cuda_program_path = env!("CUDA_TEST_PROGRAM_PATH");
        if !Path::new(cuda_program_path).exists() {
            eprintln!("CUDA test program not found at: {cuda_program_path}");
            return Err(Report::new(IntegrationTestError::CudaProgramNotFound)
                .attach(format!("CUDA test program missing: {cuda_program_path}")));
        }

        Ok(())
    }

    /// Tests that lower utilization limits should increase execution time proportionally.
    ///
    /// This test verifies the core functionality of the GPU limiter by running the same
    /// workload with different utilization limits and ensuring the execution times scale
    /// as expected.
    #[tokio::test]
    #[serial]
    async fn lower_utilization_limits_should_increase_execution_time(
    ) -> Result<(), Report<IntegrationTestError>> {
        // Skip test if prerequisites aren't met
        if check_test_prerequisites().is_err() {
            println!("Skipping test: prerequisites not met");
            return Ok(());
        }
        const TEST_ITERATIONS_SCALING: u64 = 30;

        let gpu_index = 0;

        // Create a single coordinator manager
        let mut coordinator_manager = TestCoordinatorManager::new(gpu_index).await;

        // Register different pods with different limits
        let pod_80 = coordinator_manager
            .register_pod_with_limit("scaling_80", 80)
            .await?;
        let pod_40 = coordinator_manager
            .register_pod_with_limit("scaling_40", 40)
            .await?;
        let pod_20 = coordinator_manager
            .register_pod_with_limit("scaling_20", 20)
            .await?;

        // Run tests with different utilization limits
        let (t80, _) = coordinator_manager
            .run_test_with_pod(
                &pod_80,
                "80% utilization limit",
                TEST_ITERATIONS_SCALING,
                TEST_MEMORY,
                true,
            )
            .await?;

        let (t40, _) = coordinator_manager
            .run_test_with_pod(
                &pod_40,
                "40% utilization limit",
                TEST_ITERATIONS_SCALING,
                TEST_MEMORY,
                true,
            )
            .await?;

        let (t20, _) = coordinator_manager
            .run_test_with_pod(
                &pod_20,
                "20% utilization limit",
                TEST_ITERATIONS_SCALING,
                TEST_MEMORY,
                true,
            )
            .await?;

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
}
