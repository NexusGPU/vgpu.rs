use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::process::Output;
use std::process::Stdio;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;
use serde_json::json;
use tempfile::TempDir;
use tracing::info;
use tracing::warn;

use crate::test_setup;

/// QoS levels for test processes
#[derive(Debug, Clone, Copy)]
pub enum QosLevel {
    Low,
    Medium,
    High,
    #[allow(dead_code)]
    Critical,
}

impl QosLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            QosLevel::Low => "Low",
            QosLevel::Medium => "Medium",
            QosLevel::High => "High",
            QosLevel::Critical => "Critical",
        }
    }
}

/// Memory allocation patterns for test clients
#[derive(Debug, Clone)]
pub enum MemoryPattern {
    /// Single large allocation
    SingleLargeAlloc(u64),
    /// Repeated allocations
    #[allow(dead_code)]
    RepeatedAlloc { size: u64, count: u32 },
    /// Gradually increasing allocation
    #[allow(dead_code)]
    GradualIncrease { start: u64, step: u64, max: u64 },
}

/// Configuration for test client
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub memory_pattern: MemoryPattern,
    pub duration: Duration,
    #[allow(dead_code)]
    pub gpu_index: usize,
}

/// Information about a running test client
#[derive(Debug)]
pub struct TestClient {
    pub pid: u32,
    pub process: Child,
}

/// Main integration test setup and management
pub struct IntegrationTestSetup {
    /// Hypervisor process
    hypervisor_process: Option<Child>,
    /// Running worker processes
    worker_processes: Vec<Child>,
    /// Running test clients
    test_clients: Vec<TestClient>,
    /// Temporary directories for test artifacts
    temp_dirs: Vec<TempDir>,
    /// IPC path for trap communication
    ipc_path: PathBuf,
    /// Socket path for worker discovery
    sock_path: PathBuf,
    /// Paths to tensor-fusion components
    tensor_fusion_paths: TensorFusionPaths,
    /// Next available port for workers
    next_port: u16,
    /// Optional path for GPU metrics file
    gpu_metrics_file: Option<PathBuf>,
    /// Temporary directory for metrics file
    _metrics_temp_dir: Option<TempDir>,
}

/// Paths to tensor-fusion components
#[derive(Debug, Clone)]
struct TensorFusionPaths {
    worker_binary: PathBuf,
    libcuda: PathBuf,
    libteleport: PathBuf,
    libnvidia_ml: PathBuf,
    cuda_limiter: PathBuf,
    hypervisor_binary: PathBuf,
}

impl IntegrationTestSetup {
    /// Create a new test setup
    pub fn new() -> Result<Self> {
        let temp_dir = TempDir::new().context("Failed to create temp directory")?;
        let sock_path = temp_dir.path().join("workers");
        let ipc_path = temp_dir.path().join("ipc_server");

        // Create necessary directories
        fs::create_dir_all(&sock_path)?;

        let tensor_fusion_paths = Self::get_tensor_fusion_paths()?;

        Ok(Self {
            hypervisor_process: None,
            worker_processes: Vec::new(),
            test_clients: Vec::new(),
            temp_dirs: vec![temp_dir],
            ipc_path,
            sock_path,
            tensor_fusion_paths,
            next_port: 50000,
            gpu_metrics_file: None,
            _metrics_temp_dir: None,
        })
    }

    /// Enable metrics logging to a temporary file
    pub fn enable_metrics_logging(&mut self) -> Result<PathBuf> {
        let temp_dir = TempDir::new().context("Failed to create temp dir for metrics")?;
        let file_path = temp_dir.path().join("metrics.log");
        self.gpu_metrics_file = Some(file_path.clone());
        self._metrics_temp_dir = Some(temp_dir);
        Ok(file_path)
    }

    /// Get paths to tensor-fusion components from environment variables
    fn get_tensor_fusion_paths() -> Result<TensorFusionPaths> {
        let worker_binary = PathBuf::from(env!("TENSOR_FUSION_WORKER_PATH"));

        let libcuda = PathBuf::from(env!("TENSOR_FUSION_LIBCUDA_PATH"));

        let libteleport = PathBuf::from(env!("TENSOR_FUSION_LIBTELEPORT_PATH"));

        let libnvidia_ml = PathBuf::from(env!("TENSOR_FUSION_LIBNVML_PATH"));

        // Build cuda-limiter.so and hypervisor binary
        let cuda_limiter = build_and_find_cuda_limiter()?;
        let hypervisor_binary = build_and_find_hypervisor()?;

        Ok(TensorFusionPaths {
            worker_binary,
            libcuda,
            libteleport,
            libnvidia_ml,
            cuda_limiter,
            hypervisor_binary,
        })
    }

    /// Start the hypervisor process
    pub fn start_hypervisor(&mut self) -> Result<()> {
        if self.hypervisor_process.is_some() {
            return Err(anyhow::anyhow!("Hypervisor already running"));
        }

        let mut command = Command::new(&self.tensor_fusion_paths.hypervisor_binary);

        command
            .arg("--sock-path")
            .arg(&self.sock_path)
            .arg("--ipc-path")
            .arg(&self.ipc_path)
            .arg("--metrics-batch-size")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(metrics_file) = &self.gpu_metrics_file {
            command.arg("--gpu-metrics-file").arg(metrics_file);
        }

        let child = command.spawn().context("Failed to start hypervisor")?;

        self.hypervisor_process = Some(child);

        // Give hypervisor time to start up
        std::thread::sleep(Duration::from_secs(2));

        Ok(())
    }

    /// Start a worker process with specified configuration
    pub fn start_worker(&mut self, qos: QosLevel, mem_limit: u64, tflop_limit: u32) -> Result<u32> {
        let port = self.next_port;
        self.next_port += 1;

        // Get GPU UUID (assume first GPU for now)
        let gpu_uuid = get_gpu_uuid(0)?;

        // Create memory and compute limit configurations
        let mem_limit_config = json!({
            gpu_uuid.clone(): mem_limit
        });
        let tflop_limit_config = json!({
            gpu_uuid.clone(): tflop_limit
        });

        let mut env_vars = HashMap::new();
        env_vars.insert(
            "TENSOR_FUSION_IPC_SERVER_PATH".to_string(),
            self.ipc_path.to_string_lossy().to_string(),
        );
        env_vars.insert(
            "LD_PRELOAD".to_string(),
            self.tensor_fusion_paths
                .cuda_limiter
                .to_string_lossy()
                .to_string(),
        );
        env_vars.insert(
            "TENSOR_FUSION_CUDA_MEM_LIMIT".to_string(),
            mem_limit_config.to_string(),
        );
        env_vars.insert(
            "TENSOR_FUSION_CUDA_UP_LIMIT_TFLOPS".to_string(),
            tflop_limit_config.to_string(),
        );
        env_vars.insert(
            "TENSOR_FUSION_QOS_LEVEL".to_string(),
            qos.as_str().to_string(),
        );
        env_vars.insert("NVIDIA_VISIBLE_DEVICES".to_string(), gpu_uuid);
        env_vars.insert("POD_NAME".to_string(), format!("test-worker-{port}"));
        env_vars.insert(
            "TENSOR_FUSION_WORKLOAD_NAME".to_string(),
            format!("test-workload-{port}"),
        );

        let mut command = Command::new(&self.tensor_fusion_paths.worker_binary);
        command
            .arg("-n")
            .arg("native")
            .arg("-p")
            .arg(port.to_string())
            .arg("-s")
            .arg(&self.sock_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Set environment variables
        for (key, value) in env_vars {
            command.env(key, value);
        }

        let child = command.spawn().context("Failed to start worker")?;

        let pid = child.id();
        self.worker_processes.push(child);

        // Give worker time to start and register
        std::thread::sleep(Duration::from_secs(3));

        info!("Worker started with PID: {}", pid);
        Ok(pid)
    }

    /// Start a test client with specified configuration
    pub fn start_client(&mut self, config: ClientConfig) -> Result<u32> {
        // Find an available worker port (simplified - just use the last started worker)
        let port = self.next_port - 1;

        info!("Starting client with config: {:?}", config);

        // Prepare LD_PRELOAD
        let ld_preload = format!(
            "{}:{}:{}",
            self.tensor_fusion_paths.libcuda.display(),
            self.tensor_fusion_paths.libteleport.display(),
            self.tensor_fusion_paths.libnvidia_ml.display()
        );

        let connection_info = format!("native+127.0.0.1+{port}+{port}");

        let mut env_vars = HashMap::new();
        env_vars.insert("LD_PRELOAD".to_string(), ld_preload);
        env_vars.insert(
            "TENSOR_FUSION_OPERATOR_GET_CONNECTION_DEBUG_INFO".to_string(),
            connection_info,
        );

        // Build command for our CUDA test program
        let cuda_test_program = env!("CUDA_TEST_PROGRAM_PATH");

        let mut command = Command::new(cuda_test_program);

        // Add arguments based on memory pattern
        match &config.memory_pattern {
            MemoryPattern::SingleLargeAlloc(size) => {
                command.args([
                    "--memory-bytes",
                    &size.to_string(),
                    "--duration",
                    &config.duration.as_secs().to_string(),
                    "--pattern",
                    "single",
                ]);
            }
            MemoryPattern::RepeatedAlloc { size, count } => {
                command.args([
                    "--memory-bytes",
                    &size.to_string(),
                    "--duration",
                    &config.duration.as_secs().to_string(),
                    "--pattern",
                    "repeated",
                    "--count",
                    &count.to_string(),
                ]);
            }
            MemoryPattern::GradualIncrease { start, step, max } => {
                command.args([
                    "--memory-bytes",
                    &start.to_string(),
                    "--duration",
                    &config.duration.as_secs().to_string(),
                    "--pattern",
                    "gradual",
                    "--step",
                    &step.to_string(),
                    "--max",
                    &max.to_string(),
                ]);
            }
        }

        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        // Set environment variables
        for (key, value) in env_vars {
            command.env(key, value);
        }

        let child = command.spawn().context("Failed to start test client")?;

        let pid = child.id();
        let test_client = TestClient {
            pid,
            process: child,
        };

        self.test_clients.push(test_client);

        info!("Test client started with PID: {}", pid);
        Ok(pid)
    }

    /// Wait for a specific client to complete
    pub fn wait_for_client(&mut self, pid: u32, timeout: Duration) -> Result<std::process::Output> {
        let start_time = Instant::now();

        let client_index = self
            .test_clients
            .iter()
            .position(|c| c.pid == pid)
            .ok_or_else(|| anyhow::anyhow!("Client with PID {} not found", pid))?;

        let mut client = self.test_clients.remove(client_index);

        // Wait for the process with timeout
        loop {
            match client.process.try_wait()? {
                Some(status) => {
                    let output = std::process::Output {
                        status,
                        stdout: Vec::new(), // We'll need to capture this separately
                        stderr: Vec::new(),
                    };
                    return Ok(output);
                }
                None => {
                    if start_time.elapsed() > timeout {
                        // Kill the process
                        let _ = client.process.kill();
                        return Err(anyhow::anyhow!("Client timed out after {:?}", timeout));
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }

    /// Check if hypervisor is still running
    pub fn is_hypervisor_running(&mut self) -> bool {
        if let Some(ref mut process) = self.hypervisor_process {
            match process.try_wait() {
                Ok(Some(_)) => false, // Process has exited
                Ok(None) => true,     // Process is still running
                Err(_) => false,      // Error checking process
            }
        } else {
            false
        }
    }

    /// Get the number of running workers
    pub fn running_worker_count(&mut self) -> usize {
        self.worker_processes.retain(|_child| {
            // This is a bit hacky since we can't easily check child process status
            // In a real implementation, we might track this differently
            true
        });
        self.worker_processes.len()
    }

    /// Stop a specific worker by killing its process
    pub fn stop_worker(&mut self, index: usize) -> Result<()> {
        if index >= self.worker_processes.len() {
            return Err(anyhow::anyhow!("Worker index {} out of range", index));
        }

        let mut worker = self.worker_processes.remove(index);
        // We must wait on the child to avoid zombies.
        // If we can't kill it, it might be already dead, so waiting should be fine.
        let kill_res = worker.kill();
        let wait_res = worker.wait();

        // Report kill error first if it happened.
        kill_res.context("Failed to kill worker process")?;
        wait_res.context("Failed to wait for worker process")?;

        info!("Worker {} stopped", index);
        Ok(())
    }

    /// Cleanup all processes and temporary files
    pub fn cleanup(&mut self) {
        // Kill all test clients
        for mut client in self.test_clients.drain(..) {
            if let Err(e) = client.process.kill() {
                warn!("Failed to kill test client {}: {}", client.pid, e);
            }
        }

        // Kill all workers
        for mut worker in self.worker_processes.drain(..) {
            if let Err(e) = worker.kill() {
                warn!("Failed to kill worker: {}", e);
            }
        }

        // Kill hypervisor
        if let Some(mut hypervisor) = self.hypervisor_process.take() {
            if let Err(e) = hypervisor.kill() {
                warn!("Failed to kill hypervisor: {}", e);
            }
        }

        // Clean up temp directories
        self.temp_dirs.clear();
    }
}

/// Build hypervisor binary
fn build_and_find_hypervisor() -> Result<PathBuf> {
    let output = Command::new("cargo")
        .args(["build", "--package", "hypervisor", "--release"])
        .output()
        .context("Failed to build hypervisor")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("Failed to build hypervisor: {}", stderr));
    }

    // Find the built binary - try different possible paths
    let possible_paths = [
        PathBuf::from("target/release/hypervisor"),
        PathBuf::from("../../target/release/hypervisor"),
        env::current_dir()?.join("target/release/hypervisor"),
        env::current_dir()?.join("../../target/release/hypervisor"),
        PathBuf::from("target/debug/hypervisor"),
        PathBuf::from("../../target/debug/hypervisor"),
        env::current_dir()?.join("target/debug/hypervisor"),
        env::current_dir()?.join("../../target/debug/hypervisor"),
    ];

    for binary_path in &possible_paths {
        if binary_path.exists() {
            return Ok(binary_path.clone());
        }
    }
    Err(anyhow::anyhow!(
        "hypervisor binary not found at any of the expected paths. Tried: {:?}",
        possible_paths,
    ))
}

/// Build and find cuda-limiter.so
pub fn build_and_find_cuda_limiter() -> Result<PathBuf> {
    let output = Command::new("cargo")
        .args(["build", "--package", "cuda-limiter", "--release"])
        .output()
        .context("Failed to build cuda-limiter")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("Failed to build cuda-limiter: {}", stderr));
    }

    // Find the built library - try different possible paths
    let possible_paths = [
        PathBuf::from("target/release/libcuda_limiter.so"),
        PathBuf::from("../../target/release/libcuda_limiter.so"),
        env::current_dir()?.join("target/release/libcuda_limiter.so"),
        env::current_dir()?.join("../../target/release/libcuda_limiter.so"),
        PathBuf::from("target/debug/libcuda_limiter.so"),
        PathBuf::from("../../target/debug/libcuda_limiter.so"),
        env::current_dir()?.join("target/debug/libcuda_limiter.so"),
        env::current_dir()?.join("../../target/debug/libcuda_limiter.so"),
    ];

    for lib_path in &possible_paths {
        if lib_path.exists() {
            return Ok(lib_path.clone());
        }
    }
    Err(anyhow::anyhow!(
        "cuda-limiter library not found at any of the expected paths. Tried: {:?}",
        possible_paths,
    ))
}

impl Drop for IntegrationTestSetup {
    fn drop(&mut self) {
        self.cleanup();
    }
}

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
        let limiter_path = build_and_find_cuda_limiter()?;
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
