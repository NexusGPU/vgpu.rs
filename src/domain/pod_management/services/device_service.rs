//! Device allocation and management service

use std::sync::Arc;
use api_types::WorkerInfo;
use nvml_wrapper::Nvml;
use cudarc::driver::{CudaContext, sys::CUdevice_attribute};
use utils::shared_memory::{DeviceConfig, handle::SharedMemoryHandle, manager::ThreadSafeSharedMemoryManager};

use crate::domain::pod_management::{
    core::error::{PodManagementError, Result},
    types::{DeviceAllocation, DeviceUsage, DeviceQuota, PodId},
};
use crate::config::GPU_CAPACITY_MAP;

// Configuration constant for CUDA cores calculation
const FACTOR: u32 = 64;

/// Service for managing device allocation and monitoring
#[derive(Debug)]
pub struct DeviceService {
    nvml: Arc<Nvml>,
    shared_memory_manager: Arc<ThreadSafeSharedMemoryManager>,
    glob_pattern: String,
}

impl DeviceService {
    /// Create a new device service
    pub fn new(
        nvml: Arc<Nvml>,
        shared_memory_manager: Arc<ThreadSafeSharedMemoryManager>,
        glob_pattern: String,
    ) -> Self {
        Self {
            nvml,
            shared_memory_manager,
            glob_pattern,
        }
    }

    /// Create device allocation from worker info
    pub async fn create_allocation(&self, worker_info: &WorkerInfo) -> Result<DeviceAllocation> {
        let configs = self.create_device_configs(worker_info).await?;
        Ok(DeviceAllocation::new(configs))
    }

    /// Register device allocation for a pod
    pub fn register_pod_allocation(&self, pod_id: &PodId, allocation: &DeviceAllocation) -> Result<()> {
        self.shared_memory_manager
            .register_pod(pod_id.as_str(), allocation.configs.clone())
            .map_err(|e| PodManagementError::SharedMemoryError(e.to_string()))
    }

    /// Unregister device allocation for a pod
    pub fn unregister_pod_allocation(&self, pod_id: &PodId) -> Result<()> {
        self.shared_memory_manager
            .unregister_pod(pod_id.as_str())
            .map_err(|e| PodManagementError::SharedMemoryError(e.to_string()))
    }

    /// Register a process for resource limiting
    pub fn register_process(
        &self,
        pod_id: &PodId,
        container_name: &str,
        container_pid: u32,
        host_pid: u32,
    ) -> Result<()> {
        self.shared_memory_manager
            .register_process(pod_id.as_str(), container_name, container_pid, host_pid)
            .map_err(|e| PodManagementError::SharedMemoryError(e.to_string()))
    }

    /// Unregister a process
    pub fn unregister_process(&self, pod_id: &PodId, host_pid: u32) -> Result<()> {
        self.shared_memory_manager
            .unregister_process(pod_id.as_str(), host_pid)
            .map_err(|e| PodManagementError::SharedMemoryError(e.to_string()))
    }

    /// Get current device usage for monitoring
    pub async fn get_device_usage(&self, device_idx: u32) -> Result<DeviceUsage> {
        let device = self.nvml.device_by_index(device_idx)?;
        let uuid = device.uuid()?;
        
        // Get current utilization
        let utilization = device.utilization_rates()?;
        let memory_info = device.memory_info()?;
        
        // Count active processes
        let processes = device.running_compute_processes()?;
        
        Ok(DeviceUsage {
            device_idx,
            device_uuid: uuid,
            cuda_usage: utilization.gpu,
            memory_usage: memory_info.used,
            active_processes: processes.len() as u32,
        })
    }

    /// Check if device usage exceeds allocation
    pub async fn check_resource_violations(
        &self,
        allocation: &DeviceAllocation,
    ) -> Result<Vec<(u32, DeviceUsage)>> {
        let mut violations = Vec::new();
        
        for config in &allocation.configs {
            let usage = self.get_device_usage(config.device_idx).await?;
            let quota = allocation.get_quota(config.device_idx)
                .ok_or_else(|| PodManagementError::InvalidConfiguration(
                    format!("No quota found for device {}", config.device_idx)
                ))?;
            
            if usage.exceeds_quota(quota) {
                violations.push((config.device_idx, usage));
            }
        }
        
        Ok(violations)
    }

    /// Create device configs from worker info
    async fn create_device_configs(&self, worker_info: &WorkerInfo) -> Result<Vec<DeviceConfig>> {
        let gpu_uuids = worker_info.gpu_uuids.as_deref().unwrap_or(&[]);
        let mut device_configs = Vec::new();

        for gpu_uuid in gpu_uuids {
            let device = self.nvml.device_by_uuid(gpu_uuid.as_str())?;
            let device_idx = device.index()?;

            let tflops_capacity = *GPU_CAPACITY_MAP
                .read()
                .expect("poisoned")
                .get(gpu_uuid.as_str())
                .unwrap_or(&0.0);

            let (total_cuda_cores, sm_count, max_thread_per_sm, up_limit, mem_limit) =
                self.calculate_device_limits(
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
    fn calculate_device_limits(
        &self,
        device_idx: u32,
        tflops_limit: Option<f64>,
        vram_limit: Option<u64>,
        tflops_capacity: Option<f64>,
    ) -> Result<(u32, u32, u32, u32, u64)> {
        let device = self.nvml.device_by_index(device_idx)?;
        let ctx = CudaContext::new(device_idx as usize)?;
        
        let sm_count = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        let max_thread_per_sm = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)? as u32;

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
}