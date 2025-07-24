//! Worker and pod registration logic

use super::coordinator::LimiterCoordinator;
use crate::config::GPU_CAPACITY_MAP;
use crate::process::worker::TensorFusionWorker;
use crate::process::GpuProcess;
use anyhow::Result;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::CudaContext;
use nvml_wrapper::Nvml;
use std::sync::Arc;
use utils::shared_memory::DeviceConfig;

// Configuration constant for CUDA cores calculation
const FACTOR: u32 = 64;

/// Registers a worker process with the limiter coordinator.
pub async fn register_worker_to_limiter_coordinator(
    pod_identifier: &str,
    limiter_coordinator: &LimiterCoordinator,
    worker: &Arc<TensorFusionWorker>,
    container_name: &str,
    container_pid: u32,
    host_pid: u32,
) -> Result<()> {
    // Register process with the limiter coordinator.
    limiter_coordinator.register_process(
        pod_identifier,
        container_name,
        container_pid,
        host_pid,
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

/// Creates device configs from WorkerInfo (pod metadata) for pod-level registration
pub async fn create_device_configs_from_worker_info(
    worker_info: &api_types::WorkerInfo,
    nvml: &Nvml,
) -> Result<Vec<DeviceConfig>> {
    let gpu_uuids = worker_info.gpu_uuids.as_deref().unwrap_or(&[]);
    let mut device_configs = Vec::new();

    for gpu_uuid in gpu_uuids {
        let device = nvml.device_by_uuid(gpu_uuid.as_str())?;
        let device_idx = device.index()?;

        let tflops_capacity = *GPU_CAPACITY_MAP
            .read()
            .expect("poisoned")
            .get(gpu_uuid.as_str())
            .unwrap_or(&0.0);
        let (total_cuda_cores, sm_count, max_thread_per_sm, up_limit, mem_limit) =
            calculate_device_limits_from_gpu_info(
                nvml,
                device_idx,
                worker_info.tflops_limit,
                worker_info.vram_limit,
                if tflops_capacity > 0.0 {
                    Some(tflops_capacity)
                } else {
                    None
                },
            )?;

        device_configs.push(DeviceConfig {
            device_idx,
            device_uuid: gpu_uuid.to_string(),
            up_limit,
            mem_limit,
            total_cuda_cores,
            sm_count,
            max_thread_per_sm,
        });
    }

    Ok(device_configs)
}

/// Calculate device limits from actual GPU hardware information
fn calculate_device_limits_from_gpu_info(
    nvml: &Nvml,
    device_idx: u32,
    tflops_limit: Option<f64>,
    vram_limit: Option<u64>,
    tflops_capacity: Option<f64>,
) -> Result<(u32, u32, u32, u32, u64)> {
    let device = nvml.device_by_index(device_idx)?;
    let ctx = CudaContext::new(device_idx as usize)?;
    let sm_count =
        ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
    let max_thread_per_sm = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?
        as u32;

    // Calculate total CUDA cores using the formula: sm_count * max_thread_per_sm * FACTOR
    let total_cuda_cores = sm_count * max_thread_per_sm * FACTOR;

    // Get memory information
    let memory_info = device.memory_info()?;
    let total_memory = memory_info.total;

    let up_limit = match (tflops_limit, tflops_capacity) {
        (Some(tflops_limit), Some(tflops_capacity)) if tflops_capacity > 0.0 => {
            (((tflops_limit / tflops_capacity) * 100.0).round() as u32).min(100)
        }
        _ => 100,
    };
    let mem_limit = vram_limit.unwrap_or(total_memory);

    tracing::debug!(
        device_idx = device_idx,
        sm_count = sm_count,
        max_thread_per_sm = max_thread_per_sm,
        factor = FACTOR,
        total_cuda_cores = total_cuda_cores,
        total_memory = total_memory,
        up_limit = up_limit,
        mem_limit = mem_limit,
        "Calculated device limits from GPU hardware info"
    );

    Ok((
        total_cuda_cores,
        sm_count,
        max_thread_per_sm,
        up_limit,
        mem_limit,
    ))
}
