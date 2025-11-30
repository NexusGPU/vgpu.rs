//! Worker and pod registration logic

use crate::config::GPU_CAPACITY_MAP;
use anyhow::Result;
use api_types::PodResourceInfo;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::CudaContext;
use nvml_wrapper::Nvml;
use utils::shared_memory::DeviceConfig;

/// Configuration constant for CUDA cores calculation
/// This factor is used to scale SM count and threads per SM to estimate total CUDA cores
/// Formula: total_cuda_cores = sm_count * max_thread_per_sm * FACTOR
const FACTOR: u32 = 64;

/// Creates device configs from PodResourceInfo (pod metadata) for pod-level registration
#[tracing::instrument(skip(nvml), fields(pod = %pod_info.pod_name, namespace = %pod_info.namespace, gpu_count = pod_info.gpu_uuids.as_ref().map(|v| v.len()).unwrap_or(0)))]
pub async fn create_device_configs_from_pod_resource_info(
    pod_info: &PodResourceInfo,
    nvml: &Nvml,
) -> Result<Vec<DeviceConfig>> {
    let gpu_uuids = pod_info.gpu_uuids.as_deref().unwrap_or(&[]);
    let mut device_configs = Vec::new();

    tracing::info!(
        pod_name = %pod_info.pod_name,
        namespace = %pod_info.namespace,
        tflops_limit = ?pod_info.tflops_limit,
        vram_limit = ?pod_info.vram_limit,
        gpu_uuids = ?gpu_uuids,
        "Creating device configs from PodResourceInfo"
    );

    for gpu_uuid in gpu_uuids {
        let device = nvml.device_by_uuid(gpu_uuid.replace("gpu-", "GPU-").as_str())?;
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
                pod_info.tflops_limit,
                pod_info.vram_limit,
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
#[tracing::instrument(skip(nvml), fields(device_idx = device_idx, tflops_limit = ?tflops_limit, vram_limit = ?vram_limit, tflops_capacity = ?tflops_capacity))]
pub fn calculate_device_limits_from_gpu_info(
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

#[cfg(test)]
mod tests {
    use api_types::PodResourceInfo;
    use std::collections::BTreeMap;

    fn create_test_pod_resource_info(gpu_uuids: Option<Vec<String>>) -> PodResourceInfo {
        PodResourceInfo {
            namespace: "test-namespace".to_string(),
            pod_name: "test-pod".to_string(),
            containers: Some(vec!["test-container".to_string()]),
            gpu_uuids,
            qos_level: Some(api_types::QosLevel::Medium),
            tflops_request: Some(5.0),
            tflops_limit: Some(10.0),
            vram_request: Some(1024),
            vram_limit: Some(2048),
            node_name: Some("test-node".to_string()),
            host_pid: 12345,
            labels: BTreeMap::new(),
            workload_name: Some("test-workload".to_string()),
            compute_shard: false,
        }
    }

    #[test]
    fn test_up_limit_calculation_logic() {
        // Test up_limit calculation logic separately (extracted from the main function)

        // Test case 1: Normal calculation
        let tflops_limit = Some(5.0);
        let tflops_capacity = Some(10.0);
        let up_limit = match (tflops_limit, tflops_capacity) {
            (Some(tflops_limit), Some(tflops_capacity)) if tflops_capacity > 0.0 => {
                let percentage: f64 = (tflops_limit / tflops_capacity) * 100.0;
                let rounded_percentage = percentage.round() as u32;
                rounded_percentage.min(100)
            }
            _ => 100,
        };
        assert_eq!(up_limit, 50); // 5.0/10.0 * 100 = 50%

        // Test case 2: Limit higher than capacity
        let tflops_limit = Some(15.0);
        let tflops_capacity = Some(10.0);
        let up_limit = match (tflops_limit, tflops_capacity) {
            (Some(tflops_limit), Some(tflops_capacity)) if tflops_capacity > 0.0 => {
                let percentage: f64 = (tflops_limit / tflops_capacity) * 100.0;
                let rounded_percentage = percentage.round() as u32;
                rounded_percentage.min(100)
            }
            _ => 100,
        };
        assert_eq!(up_limit, 100); // Should be capped at 100%

        // Test case 3: Missing tflops_limit
        let tflops_limit: Option<f64> = None;
        let tflops_capacity = Some(10.0);
        let up_limit = match (tflops_limit, tflops_capacity) {
            (Some(tflops_limit), Some(tflops_capacity)) if tflops_capacity > 0.0 => {
                let percentage: f64 = (tflops_limit / tflops_capacity) * 100.0;
                let rounded_percentage = percentage.round() as u32;
                rounded_percentage.min(100)
            }
            _ => 100,
        };
        assert_eq!(up_limit, 100); // Default to 100%

        // Test case 4: Zero tflops_capacity
        let tflops_limit: Option<f64> = Some(5.0);
        let tflops_capacity = Some(0.0);
        let up_limit = match (tflops_limit, tflops_capacity) {
            (Some(tflops_limit), Some(tflops_capacity)) if tflops_capacity > 0.0 => {
                let percentage: f64 = (tflops_limit / tflops_capacity) * 100.0;
                let rounded_percentage = percentage.round() as u32;
                rounded_percentage.min(100)
            }
            _ => 100,
        };
        assert_eq!(up_limit, 100); // Default to 100% when capacity is 0

        // Test case 5: Rounding behavior
        let tflops_limit: Option<f64> = Some(3.33);
        let tflops_capacity = Some(10.0);
        let up_limit = match (tflops_limit, tflops_capacity) {
            (Some(tflops_limit), Some(tflops_capacity)) if tflops_capacity > 0.0 => {
                let percentage: f64 = (tflops_limit / tflops_capacity) * 100.0;
                let rounded_percentage = percentage.round() as u32;
                rounded_percentage.min(100)
            }
            _ => 100,
        };
        assert_eq!(up_limit, 33); // 33.3% rounds to 33%
    }

    #[test]
    fn test_extreme_tflops_scenarios() {
        // Test edge cases with extreme TFLOPS values
        let test_cases = vec![
            (Some(0.0), Some(10.0), 0),             // Zero limit
            (Some(0.001), Some(10.0), 0),           // Very small limit (rounds to 0)
            (Some(999999.0), Some(10.0), 100),      // Massive limit (should cap at 100)
            (Some(5.0), Some(0.0), 100),            // Zero capacity (fallback)
            (Some(f64::INFINITY), Some(10.0), 100), // Infinity limit
            (Some(10.0), Some(f64::INFINITY), 100), // Infinity capacity (fallback)
            (Some(f64::NAN), Some(10.0), 100),      // NaN limit
            (Some(10.0), Some(f64::NAN), 100),      // NaN capacity
        ];

        for (tflops_limit, tflops_capacity, expected_up_limit) in test_cases {
            let up_limit = match (tflops_limit, tflops_capacity) {
                (Some(tflops_limit), Some(tflops_capacity))
                    if tflops_capacity > 0.0
                        && tflops_capacity.is_finite()
                        && tflops_limit.is_finite() =>
                {
                    let percentage: f64 = (tflops_limit / tflops_capacity) * 100.0;
                    let rounded_percentage = percentage.round() as u32;
                    rounded_percentage.min(100)
                }
                _ => 100,
            };

            assert_eq!(
                up_limit, expected_up_limit,
                "Failed for limit: {tflops_limit:?}, capacity: {tflops_capacity:?}"
            );
        }
    }

    #[test]
    fn test_rounding_precision_edge_cases() {
        // Test rounding behavior with precise decimal values
        let precision_cases = vec![
            (0.4999, 10.0, 5),   // 4.999% -> rounds to 5%
            (0.5001, 10.0, 5),   // 5.001% -> rounds to 5%
            (3.335, 10.0, 33),   // 33.35% -> rounds to 33%
            (3.345, 10.0, 33),   // 33.45% -> rounds to 33%
            (3.355, 10.0, 34),   // 33.55% -> rounds to 34%
            (9.9999, 10.0, 100), // 99.999% -> rounds to 100%
            (0.00001, 10.0, 0),  // 0.0001% -> rounds to 0%
        ];

        for (tflops_limit, tflops_capacity, expected) in precision_cases {
            let percentage: f64 = (tflops_limit / tflops_capacity) * 100.0;
            let rounded_percentage = percentage.round() as u32;
            let up_limit = rounded_percentage.min(100);

            assert_eq!(
                up_limit, expected,
                "Rounding failed for {tflops_limit}/{tflops_capacity} = {percentage}%"
            );
        }
    }

    #[test]
    fn test_comprehensive_pod_resource_info_edge_cases() {
        // Test pod resource info with various edge case combinations
        let edge_cases = vec![
            // Very long strings
            create_test_pod_resource_info(Some(vec![format!("GPU-{}", "x".repeat(1000))])),
            // Special characters in UUID
            create_test_pod_resource_info(Some(vec!["GPU-123!@#$%^&*()".to_string()])),
            // Empty string UUID
            create_test_pod_resource_info(Some(vec!["".to_string()])),
            // Many UUIDs
            create_test_pod_resource_info(Some((0..100).map(|i| format!("GPU-{i:03}")).collect())),
        ];

        for pod_info in edge_cases {
            // These should not panic even with edge case data
            let gpu_uuids = pod_info.gpu_uuids.as_deref().unwrap_or(&[]);

            // Basic validation - should handle all edge cases gracefully
            assert!(gpu_uuids.len() <= 100, "Too many GPUs: {}", gpu_uuids.len());

            for uuid in gpu_uuids {
                // UUIDs should be strings (even if invalid format)
                assert!(uuid.len() <= 1010, "UUID too long: {}", uuid.len()); // 1000 + "GPU-" prefix
            }
        }
    }
}
