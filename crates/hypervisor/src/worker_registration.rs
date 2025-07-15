use std::sync::Arc;

use anyhow::Result;
use nvml_wrapper::Nvml;
use utils::shared_memory::DeviceConfig;

use crate::limiter_coordinator::LimiterCoordinator;
use crate::process::worker::TensorFusionWorker;
use crate::process::GpuProcess;

// Configuration constant for CUDA cores calculation
const FACTOR: u32 = 64;

/// Registers a worker with the limiter coordinator.
pub async fn register_worker_to_limiter_coordinator(
    limiter_coordinator: &LimiterCoordinator,
    worker: &Arc<TensorFusionWorker>,
    container_name: &str,
    container_pid: u32,
    host_pid: u32,
    nvml: &Nvml,
) -> Result<()> {
    // Get pod info from the worker.
    let pod_identifier = &format!("{}_{}", worker.namespace, worker.pod_name);

    // Create device config based on the worker's GPU info.
    let device_config = create_device_config_from_worker(worker, nvml).await?;

    // Register with the limiter coordinator.
    limiter_coordinator.register_device(
        pod_identifier,
        container_name,
        container_pid,
        host_pid,
        device_config,
    )?;

    tracing::info!(
        "Registered worker {} (container_name: {}, container_pid: {}, host_pid: {}) to limiter coordinator",
        worker.name(),
        container_name,
        container_pid,
        host_pid
    );

    Ok(())
}

/// Unregisters a worker from the limiter coordinator.
pub async fn unregister_worker_from_limiter_coordinator(
    limiter_coordinator: &LimiterCoordinator,
    pod_name: &str,
    pod_namespace: &str,
    container_name: &str,
    container_pid: u32,
) -> Result<()> {
    let pod_identifier = &format!("{}_{}", pod_namespace, pod_name);
    limiter_coordinator.unregister_device(pod_identifier, container_name, container_pid)?;

    tracing::info!(
        "Unregistered worker (pod_identifier: {}, container_name: {}, container_pid: {}) from limiter coordinator",
        pod_identifier,
        container_name,
        container_pid
    );

    Ok(())
}

/// Creates a device config from a worker's GPU info.
async fn create_device_config_from_worker(
    worker: &Arc<TensorFusionWorker>,
    nvml: &Nvml,
) -> Result<DeviceConfig> {
    // Get the first GPU UUID from the worker (assuming single GPU for now).
    let gpu_uuid = worker
        .gpu_uuids()
        .first()
        .ok_or_else(|| anyhow::anyhow!("Worker has no GPU UUIDs"))?;

    // Map the GPU UUID to a device index.
    let device_count = nvml.device_count()?;
    let mut device_idx = 0;
    let mut found = false;

    for i in 0..device_count {
        let device = nvml.device_by_index(i)?;
        let uuid = device.uuid()?.to_lowercase();
        if uuid == gpu_uuid.to_lowercase() {
            device_idx = i;
            found = true;
            break;
        }
    }

    if !found {
        return Err(anyhow::anyhow!("GPU UUID {} not found in system", gpu_uuid));
    }

    // Get actual GPU hardware information and calculate total CUDA cores
    let (total_cuda_cores, up_limit, mem_limit) =
        calculate_device_limits_from_gpu_info(nvml, device_idx)?;

    Ok(DeviceConfig {
        device_idx,
        up_limit,
        mem_limit,
        total_cuda_cores,
    })
}

/// Calculate device limits from actual GPU hardware information
fn calculate_device_limits_from_gpu_info(nvml: &Nvml, device_idx: u32) -> Result<(u32, u32, u64)> {
    let device = nvml.device_by_index(device_idx)?;

    // Get number of streaming multiprocessors (SM cores)
    let sm_count = device.num_cores()?;

    // For max threads per SM, we use a fixed value based on modern GPU architectures
    // Most modern GPUs support 2048 threads per SM (32 warps * 64 threads per warp)
    let max_thread_per_sm = 2048u32;

    // Calculate total CUDA cores using the formula: sm_count * max_thread_per_sm * FACTOR
    let total_cuda_cores = sm_count * max_thread_per_sm * FACTOR;

    // Get memory information
    let memory_info = device.memory_info()?;
    let total_memory = memory_info.total;

    // Use default resource limits (these could be made configurable)
    let tflops_limit = 10.0; // Default 10 TFLOPS
    let vram_limit = (total_memory / 2).min(8 * 1024 * 1024 * 1024); // Use half of total memory or 8GB, whichever is smaller

    // Convert TFLOPS to utilization percentage (simplified calculation)
    let up_limit = (tflops_limit * 8.0) as u32; // Rough conversion, needs refinement
    let up_limit = up_limit.min(100); // Cap at 100%

    tracing::debug!(
        device_idx = device_idx,
        sm_count = sm_count,
        max_thread_per_sm = max_thread_per_sm,
        factor = FACTOR,
        total_cuda_cores = total_cuda_cores,
        total_memory = total_memory,
        vram_limit = vram_limit,
        up_limit = up_limit,
        "Calculated device limits from GPU hardware info"
    );

    Ok((total_cuda_cores, up_limit, vram_limit))
}
