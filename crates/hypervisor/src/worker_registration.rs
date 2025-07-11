use std::sync::Arc;

use anyhow::Result;
use nvml_wrapper::Nvml;
use utils::shared_memory::DeviceConfig;

use crate::limiter_coordinator::LimiterCoordinator;
use crate::process::worker::TensorFusionWorker;
use crate::process::GpuProcess;

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
    let pod_name = &worker.pod_name;

    // Create device config based on the worker's GPU info.
    let device_config = create_device_config_from_worker(worker, nvml).await?;

    // Register with the limiter coordinator.
    limiter_coordinator.register_device(
        pod_name,
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
    container_name: &str,
    container_pid: u32,
) -> Result<()> {
    limiter_coordinator.unregister_device(pod_name, container_name, container_pid)?;

    tracing::info!(
        "Unregistered worker (pod_name: {}, container_name: {}, container_pid: {}) from limiter coordinator",
        pod_name,
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

    // Use default resource limits since we don't have access to the registry here.
    // These could be made configurable or passed as parameters.
    let tflops_limit = 10.0; // Default 10 TFLOPS
    let vram_limit = 8 * 1024 * 1024 * 1024; // Default 8GB

    // Convert TFLOPS to utilization percentage (simplified calculation).
    let up_limit = (tflops_limit * 8.0) as u32; // Rough conversion, needs refinement.
    let up_limit = up_limit.min(100); // Cap at 100%

    // Calculate total CUDA cores (simplified calculation).
    let total_cuda_cores = 2048; // Default value, can be calculated from GPU info.

    Ok(DeviceConfig {
        device_idx,
        up_limit,
        mem_limit: vram_limit,
        total_cuda_cores,
    })
}
