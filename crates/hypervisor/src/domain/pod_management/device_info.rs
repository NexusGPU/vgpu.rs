//! Worker and pod registration logic

use crate::config::GPU_CAPACITY_MAP;
use anyhow::Result;
use api_types::WorkerInfo;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::CudaContext;
use nvml_wrapper::Nvml;
use utils::shared_memory::DeviceConfig;

// Configuration constant for CUDA cores calculation
const FACTOR: u32 = 64;

/// Creates device configs from WorkerInfo (pod metadata) for pod-level registration
pub async fn create_device_configs_from_worker_info(
    worker_info: &WorkerInfo,
    nvml: &Nvml,
) -> Result<Vec<DeviceConfig>> {
    let gpu_uuids = worker_info.gpu_uuids.as_deref().unwrap_or(&[]);
    let mut device_configs = Vec::new();

    tracing::info!(
        pod_name = %worker_info.pod_name,
        namespace = %worker_info.namespace,
        tflops_limit = ?worker_info.tflops_limit,
        vram_limit = ?worker_info.vram_limit,
        gpu_uuids = ?gpu_uuids,
        "Creating device configs from WorkerInfo"
    );

    for gpu_uuid in gpu_uuids {
        let device = nvml.device_by_uuid(gpu_uuid.as_str())?;
        let device_idx = device.index()?;

        let tflops_capacity = tokio::task::spawn_blocking({
            let gpu_uuid = gpu_uuid.clone();
            move || {
                *GPU_CAPACITY_MAP
                    .read()
                    .expect("poisoned")
                    .get(gpu_uuid.to_lowercase().as_str())
                    .unwrap_or(&0.0)
            }
        })
        .await
        .expect("spawn_blocking failed");

        tracing::debug!(
            gpu_uuid = %gpu_uuid,
            device_idx = device_idx,
            tflops_capacity = tflops_capacity,
            "Retrieved TFLOPS capacity from GPU_CAPACITY_MAP"
        );

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

    // Add detailed logging for up_limit calculation
    tracing::debug!(
        device_idx = device_idx,
        tflops_limit = ?tflops_limit,
        tflops_capacity = ?tflops_capacity,
        "Input parameters for up_limit calculation"
    );

    let up_limit = match (tflops_limit, tflops_capacity) {
        (Some(tflops_limit), Some(tflops_capacity)) if tflops_capacity > 0.0 => {
            let percentage = (tflops_limit / tflops_capacity) * 100.0;
            let rounded_percentage = percentage.round() as u32;

            rounded_percentage.min(100)
        }
        _ => {
            tracing::warn!(
                device_idx = device_idx,
                tflops_limit = ?tflops_limit,
                tflops_capacity = ?tflops_capacity,
                "Using default up_limit=100 because tflops_limit or tflops_capacity is missing/invalid"
            );
            100
        }
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
